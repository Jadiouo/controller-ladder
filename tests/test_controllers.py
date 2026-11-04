import os
import pickle

import numpy as np
import pytest

from arena import Scenario
from arena.controllers import controller_factory
from arena.controllers.residual import MLPPolicy, Residual, spike_features
from arena.eval import DEMO_SEEDS, evaluate, run_episode
from arena.perturb import PayloadError
from arena.robots import RobotSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = DEMO_SEEDS[:6]


def test_ctc_lqr_beats_pid_in_steady_state_tracking():
    pid = evaluate(Scenario("tracking"), "pid", S).metrics["ss_rmse"]
    ctc = evaluate(Scenario("tracking"), "ctc_lqr", S).metrics["ss_rmse"]
    assert ctc.mean() < 0.2 * pid.mean() and ctc.max() < 1e-3


def test_pid_gravity_compensation_holds_posture():
    env = Scenario("tracking").make_env()
    ctrl = controller_factory("pid")(env.ctx)
    from arena.robots import ik
    env.reset(0)
    env.task.Q[:] = env.task.Q[0]; env.task.Qd[:] = 0; env.task.Qdd[:] = 0     # frozen reference
    env.data.qpos[:2], env.data.qvel[:2] = env.task.Q[0], 0
    obs = env._observe()
    for _ in range(200):
        obs, *_ = env.step(ctrl.act(obs))
    assert np.abs(env.data.qpos[:2] - env.task.Q[0]).max() < 1e-3
    nog = controller_factory("pid", gravity=False)(env.ctx)
    env.data.qpos[:2], env.data.qvel[:2] = env.task.Q[0], 0
    obs = env._observe()
    for _ in range(200):
        obs, *_ = env.step(nog.act(obs))
    assert np.abs(env.data.qpos[:2] - env.task.Q[0]).max() > 5e-3         # without compensation it sags


def test_ctc_works_on_three_link_and_all_references():
    for kind in ("lissajous", "glyph", "joint_sines"):
        sc = Scenario("tracking", {"ref": kind}, robot=RobotSpec.three_link() if kind != "joint_sines" else None)
        assert evaluate(sc, "ctc_lqr", S[:2]).metrics["ss_rmse"].max() < 1e-3


def test_pole_balance_ctc_lqr_survives_and_uncontrolled_falls():
    sc = Scenario("pole_balance")
    r = evaluate(sc, "ctc_lqr", S)
    assert r.metrics["fell"].sum() == 0 and r.metrics["rmse"].max() < 0.05
    off = evaluate(sc, lambda ctx: type("Z", (), {"reset": lambda s, seed=None: None, "act": lambda s, o: np.zeros(2)})(), S)
    assert off.metrics["fell"].min() == 1.0 and off.metrics["survival_time"].max() < 1.5


def test_pole_balance_pid_balances_nominal_but_is_weaker():
    sc = Scenario("pole_balance")
    pid, ctc = evaluate(sc, "pid", S), evaluate(sc, "ctc_lqr", S)
    assert pid.metrics["survival_time"].mean() > 3.0
    # theta-only rmse (rad) no longer ranks them (the old mixed rad+m error was dominated by the tip drift);
    # the PID's weakness is the tip drifting away from the hold point:
    assert pid.metrics["tip_rmse"].mean() > 2 * ctc.metrics["tip_rmse"].mean()


def test_pole_balance_works_on_three_link():
    sc = Scenario("pole_balance", robot=RobotSpec.three_link(payload_mass=0.5))
    assert evaluate(sc, "ctc_lqr", S[:3]).metrics["fell"].sum() == 0


def test_residual_zero_policy_equals_base():
    sc = Scenario("tracking")
    a, b = run_episode(sc, "ctc_lqr", 1000), run_episode(sc, controller_factory("residual"), 1000)
    assert np.array_equal(a.tau_applied, b.tau_applied)


def test_residual_is_bounded_and_clipped():
    sc = Scenario("tracking")
    f = controller_factory("residual", policy=lambda x: np.array([5.0, -5.0]), bound=16.0)
    env = sc.make_env()
    ctrl = f(env.ctx)
    base = controller_factory("ctc_lqr")(env.ctx)
    obs = env.reset(1000)
    d = ctrl.act(obs) - base.act(obs)
    assert np.abs(d).max() <= 16.0 + 1e-9 and np.abs(d).max() > 15.0
    assert np.abs(ctrl.act(obs)).max() <= env.ctx.tau_limit


def test_mlp_policy_matches_manual_forward():
    rng = np.random.default_rng(0)
    st = {"0.weight": rng.normal(size=(8, 16)), "0.bias": rng.normal(size=8), "2.weight": rng.normal(size=(2, 8)),
          "2.bias": rng.normal(size=2)}
    x = rng.normal(size=16)
    assert np.allclose(MLPPolicy(st)(x), st["2.weight"] @ np.tanh(st["0.weight"] @ x + st["0.bias"]) + st["2.bias"])


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results/spike02/raw_rl_W0.pkl")), reason="spike02 results absent")
def test_residual_port_reproduces_stored_spike02_actor():
    """Load the trained spike02 residual-PPO actor (W0, seed 0) and re-run it through arena: per-trajectory RMSE must
    match the numbers stored by spike02 (float32 MLP: tolerance 1e-4 rad)."""
    pytest.importorskip("torch")
    d = pickle.load(open(os.path.join(ROOT, "results/spike02/raw_rl_W0.pkl"), "rb"))
    rec = d[("rl", 0)]
    sc = Scenario("tracking", {"ref": "joint_sines"})
    f = controller_factory("residual", policy=MLPPolicy(rec["state"]), bound=16.0, features=spike_features)
    seeds = np.arange(5)                                           # already-reported test trajectories
    got = evaluate(sc, f, seeds, split="test").metrics["rmse"]
    assert np.abs(got - rec["clean"]["rmse"][:5]).max() < 1e-4
