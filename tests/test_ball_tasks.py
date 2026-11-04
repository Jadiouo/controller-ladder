"""catch / toss: interface contracts, reproducibility, true-vs-nominal perturbations, event metrics on hand-made traces,
event logic in the simulator, and the teachers' success rate on the demo seeds (no perturbation)."""
import numpy as np
import pytest

from arena import Scenario
from arena.controllers import CONTROLLERS
from arena.controllers.base import controller_factory
from arena.eval import DEMO_SEEDS, Trace, compute_metrics, evaluate, run_episode
from arena.perturb import (AirDrag, BallObsDelay, BallObsNoise, BallRestitution, Friction, ObjectMass, ReleaseDelay)
from arena.robots import RobotSpec, build_mjcf, fk
from arena.tasks import TASKS, make_task

TEACHERS = {"catch": "catch_teacher", "toss": "toss_teacher"}


# ---- registries / contracts --------------------------------------------------------------------------
def test_registered():
    assert {"catch", "toss"} <= set(TASKS.names())
    assert {"catch_teacher", "toss_teacher"} <= set(CONTROLLERS.names())


@pytest.mark.parametrize("task", ["catch", "toss"])
def test_env_contract(task):
    env = Scenario(task).make_env()
    obs = env.reset(int(DEMO_SEEDS[0]))
    assert obs.q.shape == (3,) and obs.qd.shape == (3,) and obs.k == 0
    assert obs.extra["ball_pos"].shape == (2,)
    if task == "toss":
        assert obs.ref["basket"].shape == (2,)
    assert env.error() is None and env.task.primary_metric == "success"
    assert env.ctx.task == task and "drag_rate" in env.ctx.task_info and callable(env.ctx.make_model)
    env.ctx.make_model()                                              # the nominal model builds (ball, basket, constraint)
    o2, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert o2.k == 1 and isinstance(reward, float) and info["outcome"] in (None, "failure", "success", "timeout")
    assert obs.to_vector().dtype == np.float32


@pytest.mark.parametrize("task", ["catch", "toss"])
def test_teacher_contract(task):
    env = Scenario(task).make_env()
    ctrl = controller_factory(TEACHERS[task])(env.ctx)
    obs = env.reset(7)
    ctrl.reset(7)
    tau = ctrl.act(obs)
    assert tau.shape == (3,) and np.all(np.isfinite(tau)) and np.all(np.abs(tau) <= env.ctx.tau_limit)
    ctrl.reset(7)
    assert np.array_equal(ctrl.act(obs), tau)


def test_teachers_refuse_other_tasks():
    env = Scenario("tracking").make_env()
    for name in TEACHERS.values():
        with pytest.raises(NotImplementedError):
            controller_factory(name)(env.ctx)
    with pytest.raises(NotImplementedError):
        controller_factory("ctc_lqr")(Scenario("catch").make_env().ctx)


def test_tools_and_sections_in_mjcf():
    import mujoco
    for tool in ("net", "claw"):
        m = mujoco.MjModel.from_xml_string(build_mjcf(RobotSpec.three_link(tool=tool, payload_mass=0.0)))
        assert m.nv == 3
    plain = build_mjcf(RobotSpec.two_link())
    assert "<equality>" not in plain and plain.count("<worldbody>") == 1
    with pytest.raises(ValueError):
        build_mjcf(RobotSpec.two_link(), "<!--@nonsense-->")


# ---- reproducibility ---------------------------------------------------------------------------------
@pytest.mark.parametrize("task", ["catch", "toss"])
def test_same_seed_same_episode_different_seed_different_task(task):
    sc = Scenario(task)
    a, b = run_episode(sc, TEACHERS[task], 5003), run_episode(sc, TEACHERS[task], 5003)
    assert np.array_equal(a.tau_cmd, b.tau_cmd) and a.outcome == b.outcome
    ea, eb = sc.make_env(), sc.make_env()
    ea.reset(1), eb.reset(2)
    if task == "catch":
        assert not np.allclose(ea.task.launch["v0"], eb.task.launch["v0"])
    else:
        assert not np.allclose(ea.task.basket, eb.task.basket)


def test_toss_has_unreachable_baskets():
    xs = []
    for s in DEMO_SEEDS:
        env = Scenario("toss").make_env()
        env.reset(int(s))
        xs.append(env.task.basket[0])
    assert 0.3 < np.mean(np.array(xs) > env.task.L) < 0.95          # a good share of the baskets is beyond the arm's reach


def test_catch_launch_comes_from_outside_the_workspace():
    env = Scenario("catch").make_env()
    for s in DEMO_SEEDS[:8]:
        env.reset(int(s))
        assert env.task.launch["p0"][0] > sum(env.robot.link_lengths) + 0.2           # starts beyond the reach of the arm
        assert env.task.launch["v0"][0] < 0                           # and flies toward it


# ---- perturbations change the TRUE task only ---------------------------------------------------------------
def test_param_perturbations_change_true_task_not_nominal():
    sc = Scenario("catch").with_perturb(AirDrag(1.0), ObjectMass(0.5), BallRestitution(0.8))
    env = sc.make_env()
    nom = env.nominal_task
    assert env.task.drag == pytest.approx(2 * nom.drag) and env.task.ball_mass == pytest.approx(1.5 * nom.ball_mass)
    assert env.task.restitution == 0.8 and nom.restitution != 0.8
    info = env.ctx.task_info
    assert info["drag"] == nom.drag and info["ball_mass"] == nom.ball_mass and info["drag_rate"] == nom.drag / nom.ball_mass
    # the real MuJoCo ball has the true damping / mass, the nominal model (what make_model gives) the nominal ones
    env.reset(5000)
    assert env.model.dof_damping[3] == pytest.approx(2 * nom.drag) and env.model.body("ball").mass[0] == pytest.approx(1.5 * nom.ball_mass)
    nm = env.ctx.make_model()
    assert nm.dof_damping[3] == pytest.approx(nom.drag) and nm.body("ball").mass[0] == pytest.approx(nom.ball_mass)
    assert env.model.geom("ball").solref[1] != nm.geom("ball").solref[1]


def test_release_delay_only_for_toss_and_launch_is_nominal_based():
    with pytest.raises(AttributeError):
        Scenario("catch").with_perturb(ReleaseDelay(30)).make_env()
    # the launch of the ball depends on the nominal parameters only, so the same seed launches the same ball under a drag error
    a, b = Scenario("catch").make_env(), Scenario("catch").with_perturb(AirDrag(1.0)).make_env()
    a.reset(5004), b.reset(5004)
    assert np.allclose(a.task.launch["v0"], b.task.launch["v0"])
    assert b.task.drag > a.task.drag


def test_ball_observation_perturbations_touch_only_the_measurement():
    base = Scenario("catch").make_env()
    noisy = Scenario("catch").with_perturb(BallObsNoise(0.02)).make_env()
    o0, o1 = base.reset(5001), noisy.reset(5001)
    assert np.array_equal(o0.q, o1.q) and not np.array_equal(o0.extra["ball_pos"], o1.extra["ball_pos"])
    assert np.allclose(noisy.task.measure(noisy)["ball_pos"], o0.extra["ball_pos"])       # the truth is untouched
    d_env, u_env = Scenario("catch").with_perturb(BallObsDelay(3)).make_env(), Scenario("catch").make_env()
    d, u = [d_env.reset(5001).extra["ball_pos"]], [u_env.reset(5001).extra["ball_pos"]]
    for _ in range(6):
        d.append(d_env.step(np.zeros(3))[0].extra["ball_pos"])
        u.append(u_env.step(np.zeros(3))[0].extra["ball_pos"])
    assert np.allclose(d[6], u[3]) and np.allclose(d[2], u[0])        # 3 steps old; the first value is held while the buffer fills
    assert not np.allclose(d[6], u[6])


def test_noise_is_reproducible_per_seed():
    sc = Scenario("catch").with_perturb(BallObsNoise(0.02))
    a, b = sc.make_env(), sc.make_env()
    assert np.array_equal(a.reset(9).extra["ball_pos"], b.reset(9).extra["ball_pos"])


# ---- event logic inside the simulator ----------------------------------------------------------------------
def _hold_with_gravity_comp(env, steps):
    """Run `steps` control steps with a gravity-compensating torque; returns (last info, last obs)."""
    from arena.controllers._arm import ArmModel
    m = ArmModel(env.nominal)
    info, obs = None, env._obs
    for _ in range(steps):
        obs, _, term, trunc, info = env.step(m.gravity(obs.q))
        if term or trunc:
            break
    return info, obs


def test_catch_dwell_needs_half_a_second():
    env = Scenario("catch").make_env()
    env.reset(5000)
    n = env.n
    mujoco_sid = env.model.site("tip").id
    import mujoco
    # put the ball at rest inside the basket, arm static
    env.data.qpos[:n] = env.task.q_home
    env.data.qvel[:n] = 0
    mujoco.mj_forward(env.model, env.data)
    tip = env.data.site_xpos[mujoco_sid]
    env.data.qpos[n:n + 2] = [tip[0], tip[2] + 0.034]
    env.data.qvel[n:n + 2] = 0
    env.task._dwell_n = 0
    k0 = env.k
    info, _ = _hold_with_gravity_comp(env, 200)
    assert info["outcome"] == "success"
    assert 0.5 / env.dt - 3 <= env.k - k0 <= 0.5 / env.dt + 6          # success only after ~0.5 s, not before
    # a ball far away from the basket never counts
    env.reset(5000)
    info, _ = _hold_with_gravity_comp(env, 40)
    assert info["outcome"] is None and env.task._dwell_n == 0


def test_catch_fall_is_a_failure():
    env = Scenario("catch").make_env()
    env.reset(5000)
    env.data.qpos[env.n + 1] = -1.0
    _, _, term, _, info = env.step(np.zeros(3))
    assert term and info["outcome"] == "failure"


def test_toss_release_schedule_and_delay():
    def release_step(delay_ms):
        sc = Scenario("toss").with_perturb(ReleaseDelay(delay_ms)) if delay_ms else Scenario("toss")
        from arena.controllers._arm import ArmModel
        env = sc.make_env()
        obs, m = env.reset(5000), ArmModel(env.nominal)
        for _ in range(env.n_steps):
            obs, _, term, _, _ = env.step(m.gravity(obs.q))
            if not env.data.eq_active[0]:
                return env.k
            if term:
                break
        return None
    k0 = release_step(0)
    assert k0 == int(round(0.8 / 0.005))
    assert release_step(40) == k0 + 8                                 # 40 ms = 8 control steps later


def test_toss_object_follows_tip_until_release_then_flies():
    env = Scenario("toss").make_env()
    obs = env.reset(5000)
    rng = np.random.default_rng(0)
    err = 0.0
    while env.k * env.dt < 0.75:
        obs, *_ = env.step(rng.uniform(-30, 30, 3))
        err = max(err, float(np.linalg.norm(obs.extra["ball_pos"] - fk(env.robot, obs.q))))
    assert err < 5e-3                                                 # held rigidly
    for _ in range(40):
        obs, _, term, *_ = env.step(np.zeros(3))
    assert np.linalg.norm(obs.extra["ball_pos"] - fk(env.robot, obs.q)) > 0.05     # released: separate from the tip


def test_toss_success_event_is_entering_the_basket():
    env = Scenario("toss").make_env()
    env.reset(5003)
    xb, zb = env.task.basket
    n = env.n
    env.data.eq_active[0] = 0
    env.task._released = True
    env.data.qpos[n:n + 2] = [xb + 0.05 - env.task.L, zb + 0.1]       # inside the walls, below the rim
    env.data.qvel[n:n + 2] = 0
    _, _, term, _, info = env.step(np.zeros(3))
    assert term and info["outcome"] == "success"
    env.reset(5003)
    env.data.eq_active[0] = 0
    env.data.qpos[n:n + 2] = [xb + 0.4 - env.task.L, zb + 0.1]        # beside the basket
    env.data.qvel[n:n + 2] = 0
    _, _, term, _, info = env.step(np.zeros(3))
    assert not term and info["outcome"] is None


# ---- event metrics on hand-made traces ---------------------------------------------------------------------
def _trace(T, outcome, channels, final=None):
    z = np.zeros((T, 3))
    return Trace(dt=0.005, n_steps=T, t_end=T * 0.005, terminated=outcome != "timeout", err=np.zeros((T, 0)), tau_cmd=z,
                 tau_applied=z, sat=np.zeros(T, bool), pulse=np.zeros(T, bool), channels=channels, outcome=outcome, final=final)


def test_catch_metrics_on_constructed_trace():
    T = 40
    ball_vel = np.tile([-1.0, -3.0], (T, 1))
    tip_vel = np.tile([0.0, -1.0], (T, 1))
    tip_vel[10:] = [0.0, 0.0]                                          # (after the impact the basket is slower, ignored)
    ball_vel[10:] = [0.0, 0.0]
    contact = np.zeros(T)
    contact[10:] = 1.0                                                 # the contact first shows up at row 10 -> use row 9
    ball = np.column_stack([np.linspace(1, 0.5, T), np.linspace(0.5, 0.1, T)])
    tip = np.column_stack([np.full(T, 0.5), np.full(T, 0.1)])
    tr = _trace(T, "success", dict(ball=ball, ball_vel=ball_vel, tip=tip, tip_vel=tip_vel, contact=contact))
    task = make_task("catch")
    m = compute_metrics(tr, task)
    assert m["success"] == 1.0
    assert m["impact_speed"] == pytest.approx(np.linalg.norm([-1.0, -2.0]))          # |(-1,-3) - (0,-1)|
    assert m["min_dist"] == pytest.approx(np.linalg.norm(ball - tip, axis=1).min())
    assert {"energy", "sat", "alive_frac"} <= set(m)
    miss = _trace(T, "failure", dict(ball=ball, ball_vel=ball_vel, tip=tip, tip_vel=tip_vel, contact=np.zeros(T)))
    mm = compute_metrics(miss, task)
    assert mm["success"] == 0.0 and np.isnan(mm["impact_speed"])


def test_toss_landing_error_on_constructed_trace():
    task = make_task("toss")
    basket = np.array([1.5, -0.3])
    z_rim = -0.3 + task.basket_h
    T = 6
    # the object descends through the rim plane between rows 3 and 4 (x from 1.40 to 1.60 -> crossing at the half way point)
    obj = np.array([[0.5, 0.3], [0.9, 0.5], [1.2, 0.3], [1.4, z_rim + 0.05], [1.6, z_rim - 0.05], [1.65, -0.1]])
    released = np.array([0, 1, 1, 1, 1, 1.0])
    ch = dict(obj=obj[:T - 1], released=released[:T - 1], basket=np.tile(basket, (T - 1, 1)))
    final = dict(err=np.zeros(0), q=np.zeros(3), channels=dict(obj=obj[-1], released=1.0, basket=basket))
    tr = _trace(T - 1, "failure", ch, final)
    m = compute_metrics(tr, task)
    assert m["success"] == 0.0 and m["landing_err"] == pytest.approx(abs(1.5 - 1.5), abs=1e-9)    # x at the rim plane = 1.5
    # a crossing BEFORE the release must not count (the arm swings through the rim plane while holding it)
    obj2 = obj.copy()
    rel2 = np.zeros(T)
    ch2 = dict(obj=obj2[:T - 1], released=rel2[:T - 1], basket=np.tile(basket, (T - 1, 1)))
    final2 = dict(err=np.zeros(0), q=np.zeros(3), channels=dict(obj=obj2[-1], released=0.0, basket=basket))
    m2 = compute_metrics(_trace(T - 1, "failure", ch2, final2), task)
    assert m2["landing_err"] == pytest.approx(abs(obj2[-1, 0] - 1.5))                           # fallback: where it ended
    ok = compute_metrics(_trace(T - 1, "success", ch, final), task)
    assert ok["success"] == 1.0


# ---- the teachers do their job (demo seeds, no perturbation) and the perturbations hurt ---------------------
@pytest.fixture(scope="module")
def clean_results():
    return {t: evaluate(Scenario(t), TEACHERS[t], DEMO_SEEDS) for t in TEACHERS}


@pytest.mark.parametrize("task", ["catch", "toss"])
def test_teacher_reaches_threshold_without_errors(clean_results, task):
    assert clean_results[task].metrics["success"].mean() >= 0.8


def test_catch_is_soft_when_it_works(clean_results):
    r = clean_results["catch"]
    ok = r.metrics["success"] > 0.5
    assert np.nanmedian(r.metrics["impact_speed"][ok]) < 1.0           # median relative speed at the first contact (m/s)


def test_observation_delay_hurts_the_catch_teacher(clean_results):
    r = evaluate(Scenario("catch").with_perturb(BallObsDelay(2)), "catch_teacher", DEMO_SEEDS)
    assert r.metrics["success"].mean() <= clean_results["catch"].metrics["success"].mean() - 0.25


def test_release_delay_hurts_the_toss_teacher(clean_results):
    r = evaluate(Scenario("toss").with_perturb(ReleaseDelay(60)), "toss_teacher", DEMO_SEEDS)
    assert r.metrics["success"].mean() <= clean_results["toss"].metrics["success"].mean() - 0.2
