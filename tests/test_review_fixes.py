"""Regression tests for the code-review fixes (items 1-7, a-f and the statistics list). One section per item."""
import dataclasses
import os
import sys

import numpy as np
import pytest

from arena import Scenario
from arena.controllers import Controller, controller_factory, register_controller
from arena.eval import (CAL_SEEDS, DEMO_SEEDS, TEST_SEEDS, TRAIN_SEEDS, EvalResult, Trace, compute_metrics, evaluate,
                        evaluate_grid, metrics_table, paired_table, run_episode)
from arena.eval.protocol import FAILURE_ERR
from arena.eval.stats import NonFiniteError, bootstrap_ci, holm_adjust, mean_ci, paired_diff, summarize
from arena.perturb import BallRestitution, Delay, PayloadError, PoleLength, SensorNoise, TorquePulses
from arena.robots import RobotSpec
from arena.tasks import Task, register_task

SHORT = {"duration": 0.5}


# ---------------------------------------------------------------- helper task / controllers
@register_task("_rf_event")
class EventTask(Task):
    """No error() at all: an event task (succeeds at step 50) with a ball whose restitution is a task parameter."""
    name, duration = "_rf_event", 1.0
    metrics = ("sat", "energy", "alive_frac")
    primary_metric = "success"
    restitution = 0.5

    def tip_xml(self, robot):
        return f'<geom name="ball" type="sphere" size="0.02" mass="0.05" solref="0.02 {self.restitution!r}"/>'

    def info(self):
        return dict(restitution=self.restitution)

    def reset(self, seed, env):
        env.data.qpos[:2] = [0.1, 1.0]

    def reference(self, k):
        return dict(q=np.array([0.1, 1.0]), qd=np.zeros(2), qdd=np.zeros(2), ee=np.zeros(2))

    def succeeded(self, env):
        return env.k >= 50

    def extra_metrics(self, trace):
        return dict(success=float(trace.outcome == "success"))


@register_controller("_rf_rand")
class RandCtl(Controller):
    def act(self, obs):
        return self.rng.normal(0, 1.0, 2)


@register_controller("_rf_ctxrand")
class CtxRandCtl(Controller):
    def act(self, obs):
        return self.ctx.rng.normal(0, 1.0, 2)


@register_controller("_rf_nan")
class NanCtl(Controller):
    def act(self, obs):
        return np.full(2, np.nan) if obs.k >= 10 else np.zeros(2)


@register_controller("_rf_notimpl_init")
class NotImplInit(Controller):
    def __init__(self, ctx):
        raise NotImplementedError("unsupported task")

    def act(self, obs):
        return np.zeros(2)


@register_controller("_rf_notimpl_act")
class NotImplAct(Controller):
    def act(self, obs):
        raise NotImplementedError("bug inside act")


def _hand(seeds, **metrics):
    return EvalResult("sc", "c", np.asarray(seeds), {k: np.asarray(v, float) for k, v in metrics.items()})


# ---------------------------------------------------------------- 1. make_env must not share perturbation state
def test_two_envs_from_one_scenario_do_not_share_perturbation_state():
    sc = Scenario("tracking", {"duration": 1.0}).with_perturb(SensorNoise(q=0.002, qd=0.02), Delay(2), TorquePulses())
    fac = controller_factory("ctc_lqr")

    def roll(envs, seeds, order, n=60):
        ctrls = [fac(e.ctx) for e in envs]
        obs = [e.reset(s) for e, s in zip(envs, seeds)]
        for c, s in zip(ctrls, seeds):
            c.reset(s)
        qs = [[] for _ in envs]
        for _ in range(n):
            for i in order:
                obs[i], *_ = envs[i].step(ctrls[i].act(obs[i]))
                qs[i].append(envs[i].data.qpos[:2].copy())
        return qs

    inter = roll([sc.make_env(), sc.make_env()], [5000, 5001], order=(0, 1))
    alone0 = roll([sc.make_env()], [5000], order=(0,))[0]
    alone1 = roll([sc.make_env()], [5001], order=(0,))[0]
    assert np.array_equal(inter[0], alone0) and np.array_equal(inter[1], alone1)


# ---------------------------------------------------------------- 2. early termination must not flatter energy / sat
def _tr(T, tau, sat, N=800, **kw):
    return Trace(dt=0.005, n_steps=N, t_end=T * 0.005, terminated=T < N, err=np.zeros((T, 1)),
                 tau_cmd=np.zeros((T, 2)), tau_applied=np.full((T, 2), tau), sat=np.full(T, sat, bool),
                 pulse=np.zeros(T, bool), **kw)


def test_energy_is_per_step_mean_and_sat_is_a_fraction_of_executed_steps():
    full, early = compute_metrics(_tr(800, 3.0, True)), compute_metrics(_tr(100, 3.0, True))
    assert full["energy"] == pytest.approx(early["energy"]) == pytest.approx(18.0)    # sum_j tau_j^2, mean over steps
    assert full["sat"] == early["sat"] == 1.0


def test_paired_table_does_not_judge_energy_when_survival_differs():
    seeds = np.arange(20)
    base_e = np.linspace(10, 20, 20)
    kw = dict(rmse=np.linspace(1, 2, 20), ss_rmse=np.linspace(1, 2, 20))
    base = _hand(seeds, energy=base_e, alive_frac=np.ones(20), **kw)
    same = _hand(seeds, energy=base_e * 0.5, alive_frac=np.ones(20), **kw)
    diff = _hand(seeds, energy=base_e * 0.5, alive_frac=np.r_[np.full(10, 0.4), np.ones(10)], **kw)
    row = lambda t: [ln for ln in t.splitlines() if "| energy |" in ln][0]
    t_same = paired_table({("sc", "base"): base, ("sc", "c"): same}, "base", ["energy"])
    t_diff = paired_table({("sc", "base"): base, ("sc", "c"): diff}, "base", ["energy"])
    assert "YES" in row(t_same)
    assert "YES" not in row(t_diff) and "survival" in row(t_diff)


# ---------------------------------------------------------------- 3. NaN is flagged, never silently dropped
def test_stats_refuse_nonfinite_values():
    with pytest.raises(NonFiniteError):
        mean_ci([1.0, np.nan, 3.0])
    with pytest.raises(NonFiniteError):
        mean_ci([1.0, np.inf])
    with pytest.raises(NonFiniteError):
        paired_diff([1.0, np.nan], [1.0, 2.0])
    s = summarize([1.0, np.nan, 3.0])
    assert s["n"] == 3 and s["n_valid"] == 2 and s["mean"] == 2.0
    with pytest.raises(NonFiniteError):
        summarize([1.0, np.inf])


def test_table_n_column_shows_valid_count():
    r = _hand(np.arange(4), rmse=[1, 2, 3, 4], recovery_time=[0.1, np.nan, 0.3, 0.2])
    line = [ln for ln in metrics_table({("sc", "c"): r}).splitlines() if ln.startswith("| sc | c")][0]
    assert line.split("|")[3].strip() == "3/4"
    none = _hand(np.arange(4), rmse=[1, 2, 3, 4], recovery_time=[np.nan] * 4)         # undefined by design (no pulses)
    line = [ln for ln in metrics_table({("sc", "c"): none}).splitlines() if ln.startswith("| sc | c")][0]
    assert line.split("|")[3].strip() == "4" and "n/a" in line


def test_controller_nan_output_is_sanitised_and_flagged():
    tr = run_episode(Scenario("tracking", SHORT), "_rf_nan", 5000)
    assert tr.diverged and tr.outcome == "failure" and np.isfinite(tr.tau_applied).all()
    m = compute_metrics(tr)
    assert all(np.isfinite(v) for k, v in m.items() if k != "recovery_time")


# ---------------------------------------------------------------- 4. pole error is theta only (rad), tip error separate
def test_pole_error_is_theta_only_and_tip_error_is_its_own_channel():
    env = Scenario("pole_balance").make_env()
    env.reset(5000)
    assert env.error().shape == (1,)
    tr = run_episode(Scenario("pole_balance"), "ctc_lqr", 5000)
    assert tr.err.shape[1] == 1 and tr.channels["tip_err"].shape[1:] == (2,)
    m = compute_metrics(tr, env.task)
    assert {"tip_rmse", "tip_settle_time", "survival_time"} <= set(m)
    assert env.task.primary_metric == "survival_time"
    assert m["tip_rmse"] < 0.05 and m["rmse"] < 0.05


# ---------------------------------------------------------------- 5. outcomes, terminal state, fixed failure penalty
def test_trace_records_terminal_state_and_outcome():
    sc = Scenario("pole_balance")
    tr = run_episode(sc, lambda ctx: type("Z", (), {"reset": lambda s, seed=None: None, "act": lambda s, o: np.zeros(2)})(),
                     5000)
    assert tr.outcome == "failure" and tr.failure and not tr.success and not tr.timeout
    assert abs(tr.final["err"][0]) > 0.6                              # the state that triggered the termination
    series = tr.series("err", FAILURE_ERR)
    assert series.shape == (tr.n_steps, 1) and np.all(series[len(tr.err) + 1:] == FAILURE_ERR)
    assert series[len(tr.err), 0] == tr.final["err"][0]
    ok = run_episode(sc, "ctc_lqr", 5000)
    assert ok.outcome == "timeout" and ok.timeout


def test_success_outcome_and_event_task_without_error():
    sc = Scenario("_rf_event")
    env = sc.make_env()
    assert env.error() is None
    tr = run_episode(sc, lambda ctx: type("Z", (), {"reset": lambda s, seed=None: None,
                                                  "act": lambda s, o: np.zeros(2)})(), 0)
    assert tr.outcome == "success" and tr.success and tr.t_end == pytest.approx(50 * env.dt)
    m = compute_metrics(tr, env.task)
    assert set(m) == {"sat", "energy", "alive_frac", "success"} and m["success"] == 1.0


def test_pole_survival_is_primary_and_rmse_can_be_compared_on_common_survivors_only():
    seeds = np.arange(4)
    base = _hand(seeds, rmse=[1, 1, 1, 1], survival_time=[5, 5, 5, 5], alive_frac=[1, 1, 1, 1])
    cand = _hand(seeds, rmse=[0.5, 0.5, 9.0, 0.5], survival_time=[5, 5, 1, 5], alive_frac=[1, 1, 0.2, 1])
    tbl = paired_table({("sc", "b"): base, ("sc", "c"): cand}, "b", ["rmse"], common_survivors=True)
    assert [ln for ln in tbl.splitlines() if "| rmse |" in ln][0].split("|")[4].strip() == "3"
    tbl_all = paired_table({("sc", "b"): base, ("sc", "c"): cand}, "b", ["rmse"])
    assert [ln for ln in tbl_all.splitlines() if "| rmse |" in ln][0].split("|")[4].strip() == "4"


# ---------------------------------------------------------------- 6. stochastic controllers are seeded
@pytest.mark.parametrize("name", ["_rf_rand", "_rf_ctxrand"])
def test_random_controller_is_reproducible_per_episode_seed(name):
    sc = Scenario("tracking", SHORT)
    a, b = run_episode(sc, name, 5000), run_episode(sc, name, 5000)
    c = run_episode(sc, name, 5001)
    assert np.array_equal(a.tau_cmd, b.tau_cmd) and not np.array_equal(a.tau_cmd, c.tau_cmd)
    reused = evaluate(sc, name, [5001, 5000])                          # env reused, other order
    alone = [evaluate(sc, name, [s]) for s in (5001, 5000)]
    assert np.array_equal(reused.metrics["rmse"], np.concatenate([x.metrics["rmse"] for x in alone]))


def test_controller_reset_receives_the_episode_seed():
    seen = []

    class Rec(Controller):
        def reset(self, seed=None):
            super().reset(seed)
            seen.append(seed)

        def act(self, obs):
            return np.zeros(2)

    run_episode(Scenario("tracking", SHORT), Rec, 4321)
    assert seen == [4321]


# ---------------------------------------------------------------- 7. TRAIN / CAL / TEST separation
def test_seed_splits_are_disjoint():
    sets = {"train": set(TRAIN_SEEDS), "cal": set(CAL_SEEDS), "test": set(TEST_SEEDS), "demo": set(DEMO_SEEDS)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), (a, b)
    assert len(TRAIN_SEEDS) >= 100


def test_test_seed_access_is_logged_and_splits_are_enforced(tmp_path, monkeypatch):
    log = tmp_path / "access.log"
    monkeypatch.setenv("ARENA_TEST_LOG", str(log))
    sc = Scenario("tracking", SHORT)
    evaluate(sc, "pid", DEMO_SEEDS[:1])
    assert not log.exists()
    evaluate(sc, "pid", TEST_SEEDS[:2], split="test")
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1 and "pid" in lines[0] and sc.label in lines[0] and os.path.basename(sys.argv[0]) in lines[0]
    with pytest.raises(ValueError):
        evaluate(sc, "pid", CAL_SEEDS[:1], split="train")
    with pytest.raises(ValueError):
        evaluate(sc, "pid", DEMO_SEEDS[:1], split="test")
    with pytest.raises(ValueError):
        evaluate(sc, "pid", np.r_[TEST_SEEDS[:1], CAL_SEEDS[:1]])       # mixing splits
    evaluate(sc, "pid", TRAIN_SEEDS[:1], split="train")
    assert len(log.read_text().strip().splitlines()) == 1


# ---------------------------------------------------------------- a. step API + gym wrapper
def test_step_returns_gym_style_tuple():
    env = Scenario("tracking", {"duration": 0.05}).make_env()
    env.reset(5000)
    out = env.step(np.zeros(2))
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float) and reward <= 0 and terminated is False and truncated is False
    for _ in range(env.n_steps - 1):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert truncated and not terminated and info["outcome"] == "timeout"
    pole = Scenario("pole_balance").make_env()
    pole.reset(5000)
    for _ in range(pole.n_steps):
        _, _, terminated, truncated, info = pole.step(np.zeros(2))
        if terminated or truncated:
            break
    assert terminated and not truncated and info["outcome"] == "failure"
    ev = Scenario("_rf_event").make_env()
    ev.reset(0)
    for _ in range(ev.n_steps):
        _, _, terminated, truncated, info = ev.step(np.zeros(2))
        if terminated or truncated:
            break
    assert terminated and info["outcome"] == "success"


def test_gym_wrapper():
    from arena.gym_env import ArenaGymEnv
    env = ArenaGymEnv(Scenario("tracking", {"duration": 0.05}))
    obs, info = env.reset(seed=3)
    assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
    assert env.action_space.shape == (2,) and float(env.action_space.high[0]) == 1.0
    obs, r, term, trunc, info = env.step(np.ones(2))
    assert np.allclose(info["applied"], env.unwrapped_env.ctx.tau_limit)             # action 1.0 -> tau_limit
    _, info = env.reset()                                                              # no seed: drawn from TRAIN_SEEDS
    assert info["episode_seed"] in set(TRAIN_SEEDS)
    done = False
    for _ in range(100):
        _, _, term, trunc, _ = env.step(np.zeros(2))
        if term or trunc:
            done = True
            break
    assert done and trunc and not term


# ---------------------------------------------------------------- b. reference preview
def test_reference_preview():
    env = Scenario("tracking").make_env()
    obs = env.reset(5000)
    p = obs.ref_preview(5)
    assert p["q"].shape == (5, 2) and np.array_equal(p["q"][0], obs.ref["q"])
    assert np.array_equal(p["q"][3], env.task.reference(3)["q"]) and np.array_equal(env.ctx.ref_fn(3)["q"], p["q"][3])
    env.k = env.n_steps - 2                                                            # clamped at the end of the horizon
    last = env._observe().ref_preview(5)["q"]
    assert np.array_equal(last[-1], env.task.reference(env.n_steps - 1)["q"])


# ---------------------------------------------------------------- c. nominal model for MPC
def test_ctx_make_model_is_nominal_and_includes_tool_and_pole():
    sc = Scenario("pole_balance", perturbations=(PayloadError(rel=1.0), PoleLength(rel=0.5)))
    env = sc.make_env()
    nominal = env.ctx.make_model()
    assert nominal.nq == env.model.nq == 3 and nominal.nv == 3
    assert nominal.body_mass.sum() < env.model.body_mass.sum()                         # true plant is heavier
    assert nominal.geom_pos[:, 2].max() == pytest.approx(0.35) and env.model.geom_pos[:, 2].max() == pytest.approx(0.525)
    paddle = Scenario("tracking", robot=RobotSpec.two_link(tool="paddle")).make_env()
    assert paddle.ctx.make_model().ngeom == paddle.model.ngeom
    info = Scenario("pole_balance").make_env().ctx.task_info
    assert {"pole_length", "bob_mass", "hold", "gravity", "fall_angle", "q_hold", "duration"} <= set(info)


# ---------------------------------------------------------------- d. task-parameter perturbations
def test_pole_length_perturbation_changes_true_task_only():
    env = Scenario("pole_balance", perturbations=(PoleLength(rel=-0.2),)).make_env()
    assert env.task.L == pytest.approx(0.28) and env.nominal_task.L == 0.35 and env.ctx.task_info["pole_length"] == 0.35
    assert PoleLength(length=0.5).modify_task(Scenario("pole_balance").make_env().task).L == 0.5
    with pytest.raises(ValueError):
        PoleLength()


def test_ball_restitution_perturbation():
    env = Scenario("_rf_event", perturbations=(BallRestitution(0.9),)).make_env()
    assert env.task.restitution == 0.9 and env.ctx.task_info["restitution"] == 0.5
    assert env.model.geom_solref[-1, 1] == pytest.approx(0.9) and env.ctx.make_model().geom_solref[-1, 1] == pytest.approx(0.5)
    with pytest.raises(AttributeError):
        BallRestitution(0.5).modify_task(Scenario("tracking").make_env().task)


# ---------------------------------------------------------------- e. history, previous action, multi-rate
def test_history_and_previous_action():
    env = Scenario("tracking", history=3).make_env()
    o0 = env.reset(5000)
    d = len(o0.to_vector())
    assert o0.history.shape == (3, d) and np.allclose(o0.history, o0.to_vector()) and np.all(o0.prev_action == 0)
    u = np.array([1.0, -2.0])
    o1, *_ = env.step(u)
    assert np.array_equal(o1.prev_action, u) and np.allclose(o1.history[-1], o0.to_vector())
    o2, *_ = env.step(u)
    assert np.allclose(o2.history[-1], o1.to_vector()) and np.allclose(o2.history[-2], o0.to_vector())
    assert Scenario("tracking").make_env().reset(0).history.shape[0] == 0


def test_hold_wrapper_calls_base_every_n_steps_and_supports_chunks():
    env = Scenario("tracking").make_env()
    obs = env.reset(5000)
    calls = []

    class Count(Controller):
        def act(self, o):
            calls.append(o.k)
            return np.full(2, float(len(calls)))

    hold = controller_factory("hold", base=Count, every=4)(env.ctx)
    hold.reset(0)
    outs = [hold.act(obs.replace(k=k)) for k in range(10)]
    assert calls == [0, 4, 8] and [o[0] for o in outs] == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]

    class Chunk(Controller):
        def act(self, o):
            return np.arange(4)[:, None] * np.ones((4, 2))

    ch = controller_factory("hold", base=Chunk, every=4, chunk=True)(env.ctx)
    ch.reset(0)
    assert [ch.act(obs.replace(k=k))[0] for k in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]


# ---------------------------------------------------------------- statistics
def test_binary_metrics_use_bootstrap_and_are_deterministic():
    x = np.r_[np.ones(8), np.zeros(42)]
    m, lo, hi = bootstrap_ci(x)
    assert m == pytest.approx(0.16) and 0.0 < lo < m < hi < 0.35 and (m, lo, hi) == bootstrap_ci(x)
    r = _hand(np.arange(50), fell=x)
    cell = [ln for ln in metrics_table({("sc", "c"): r}).splitlines() if ln.startswith("| sc | c")][0]
    assert "[" in cell and "±" not in cell.split("|")[-2]


def test_paired_diff_with_zero_base_mean_uses_absolute_difference():
    base = np.zeros(20)
    p = paired_diff(np.full(20, 5.0) + np.linspace(0, 1, 20), base, higher_is_better=True)
    assert np.isnan(p["rel"]) and p["win"]
    assert not paired_diff(np.zeros(20), base, higher_is_better=True)["win"]
    assert not paired_diff(np.full(20, 5.0), base, higher_is_better=False)["win"]
    pb = paired_diff(np.ones(20), np.zeros(20), method="bootstrap")                  # binary metric, worse
    assert not pb["win"] and pb["lo"] > 0


def test_paired_table_requires_identical_seeds():
    a, b = _hand(np.arange(5), rmse=np.ones(5)), _hand(np.arange(1, 6), rmse=np.ones(5))
    with pytest.raises(ValueError, match="seeds"):
        paired_table({("sc", "b"): a, ("sc", "c"): b}, "b", ["rmse"])


def test_holm_adjustment_and_table_option():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    seeds = np.arange(30)
    base = _hand(seeds, rmse=np.linspace(1, 2, 30), ss_rmse=np.linspace(1, 2, 30))
    cand = _hand(seeds, rmse=np.linspace(1, 2, 30) * 0.5, ss_rmse=np.linspace(1, 2, 30) * 0.99)
    res = {("sc", "b"): base, ("sc", "c"): cand}
    assert "Holm" not in paired_table(res, "b", ["rmse", "ss_rmse"])
    t = paired_table(res, "b", ["rmse", "ss_rmse"], holm=True)
    assert "Holm" in t and "YES" in [ln for ln in t.splitlines() if "| rmse |" in ln][0]


def test_grid_skips_only_construction_time_not_implemented_and_lists_skips():
    sc = Scenario("tracking", SHORT)
    res = evaluate_grid([sc], {"pid": "pid", "bad": "_rf_notimpl_init"}, DEMO_SEEDS[:1])
    assert set(k[1] for k in res) == {"pid"} and res.skipped == [(sc.label, "bad")]
    assert "bad" in metrics_table(res) and "skipped" in metrics_table(res).lower()
    with pytest.raises(NotImplementedError):
        evaluate_grid([sc], {"x": "_rf_notimpl_act"}, DEMO_SEEDS[:1])


def test_reference_is_read_only():
    env = Scenario("tracking").make_env()
    obs = env.reset(5000)
    for ref in (obs.ref, env.task.reference(0), Scenario("pole_balance").make_env().task.reference(0)):
        with pytest.raises(ValueError):
            ref["q"][0] = 9.0


def test_delay_from_ms_uses_actual_dt_and_observation_delay():
    assert Delay.from_ms(40, dt=0.004).steps == 10
    sc = Scenario("tracking", robot=RobotSpec.two_link(dt_sim=0.002), perturbations=(Delay.from_ms(40),))
    env = sc.make_env()
    env.reset(0)
    assert env.perts[0].steps == 10
    env = Scenario("tracking", perturbations=(Delay(3, where="obs"),)).make_env()
    obs = env.reset(5000)
    true, seen, applied = [env.data.qpos[:2].copy()], [obs.q.copy()], []
    for i in range(8):
        obs, _, _, _, info = env.step(np.full(2, 5.0))
        true.append(env.data.qpos[:2].copy())
        seen.append(obs.q.copy())
        applied.append(info["applied"])
    for k in range(9):
        assert np.array_equal(seen[k], true[max(0, k - 3)])
    assert all(np.array_equal(a, np.full(2, 5.0)) for a in applied)                  # command side untouched
