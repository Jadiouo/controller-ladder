"""whip / push_t: interface contracts, reproducibility, true-vs-nominal perturbations, event metrics on hand-made traces, event logic in
the simulator, the quasi-static push model, and the teachers' success rates on the demo seeds (recorded values minus a tolerance)."""
import numpy as np
import pytest

from arena import Scenario
from arena.controllers import CONTROLLERS
from arena.controllers._pusht_plan import PushModel, PushPlanner, approach_path, pose_cost
from arena.controllers.base import controller_factory
from arena.eval import DEMO_SEEDS, Trace, compute_metrics, evaluate, run_episode
from arena.perturb import ChainDamping, ChainLength, ChainMass, TComOffset, TFriction, TMass, TPoseNoise
from arena.robots import fk
from arena.tasks import TASKS, make_task
from arena.tasks import _pusht_geom as G

FAST_WHIP = dict(generations=6, lam=12, restarts=1, n_rescore=3)       # a cheap planner for the contract tests


# ---- registries / contracts ----------------------------------------------------------------------------------
def test_registered():
    assert {"whip", "push_t"} <= set(TASKS.names())
    assert {"whip_teacher", "whip_random", "push_t_teacher", "push_t_random"} <= set(CONTROLLERS.names())


@pytest.mark.parametrize("task,n", [("whip", 3), ("push_t", 2)])
def test_env_contract(task, n):
    env = Scenario(task).make_env()
    obs = env.reset(int(DEMO_SEEDS[0]))
    assert obs.q.shape == (n,) and obs.qd.shape == (n,) and obs.k == 0
    assert env.error() is None and env.task.primary_metric == "success" and "success" in env.task.binary_metrics
    assert env.ctx.task == task and callable(env.ctx.make_model)
    env.ctx.make_model()                                              # the nominal model builds
    if task == "whip":
        assert obs.ref["target"].shape == (2,)
        d = np.linalg.norm(obs.ref["target"])
        assert env.task.L_arm < d < env.task.L_arm + env.task.n_links * 0.10     # outside the reach, inside reach + chain
    else:
        assert obs.ref["goal"].shape == (3,) and obs.extra["t_pose"].shape == (3,)
        assert env.ctx.task_info["gravity"] == 0.0 and np.all(env.model.opt.gravity == 0.0)
    o2, reward, terminated, truncated, info = env.step(np.zeros(n))
    assert o2.k == 1 and isinstance(reward, float) and info["outcome"] in (None, "failure", "success", "timeout")
    assert obs.to_vector().dtype == np.float32


@pytest.mark.parametrize("task,name,kw", [("whip", "whip_teacher", FAST_WHIP), ("push_t", "push_t_teacher", {})])
def test_teacher_contract(task, name, kw):
    env = Scenario(task).make_env()
    ctrl = controller_factory(name, **kw)(env.ctx)
    obs = env.reset(7)
    ctrl.reset(7)
    tau = ctrl.act(obs)
    assert tau.shape == (env.n,) and np.all(np.isfinite(tau)) and np.all(np.abs(tau) <= env.ctx.tau_limit)
    ctrl.reset(7)
    assert np.array_equal(ctrl.act(obs), tau)


def test_teachers_refuse_other_tasks():
    env = Scenario("tracking").make_env()
    for name in ("whip_teacher", "whip_random", "push_t_teacher", "push_t_random"):
        with pytest.raises(NotImplementedError):
            controller_factory(name)(env.ctx)
    for task in ("whip", "push_t"):
        with pytest.raises(NotImplementedError):
            controller_factory("ctc_lqr")(Scenario(task).make_env().ctx)


def test_whip_model_has_free_chain_and_site():
    env = Scenario("whip").make_env()
    m = env.model
    assert m.nq == 3 + env.task.n_links and m.nu == 3 and m.site("whip_tip").id >= 0
    obs = env.reset(5000)
    assert np.allclose(env.task.chain_tip(env), fk(env.robot, obs.q) + [0.0, -env.task.n_links * 0.10], atol=0.02)   # chain hangs straight down


def test_push_t_model_dofs_and_gravity_off():
    env = Scenario("push_t").make_env()
    m = env.model
    assert m.nq == 2 + 3 and m.nu == 2
    assert np.allclose(m.dof_damping[2:4], env.task.table_b) and m.dof_damping[4] == pytest.approx(env.task.table_b * G.RHO2)
    obs = env.reset(5000)
    for _ in range(100):
        obs, *_ = env.step(np.zeros(2))
    assert np.allclose(obs.extra["t_pose"], env.task.t0_pose)          # no gravity: nothing moves without a push


# ---- reproducibility ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("task,n", [("whip", 3), ("push_t", 2)])
def test_same_seed_same_world_different_seed_differs(task, n):
    def rollout(seed):
        env = Scenario(task).make_env()
        obs = env.reset(seed)
        first = obs.to_vector().copy()
        rng = np.random.default_rng(3)
        for _ in range(60):
            obs, *_ = env.step(rng.uniform(-5, 5, n))
        return first, obs.to_vector().copy(), env
    a, b, c = rollout(5001), rollout(5001), rollout(5002)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_reused_env_equals_fresh_env():
    for task, name, kw in (("push_t", "push_t_random", {}), ("whip", "whip_random", {})):
        sc, seeds = Scenario(task), [5003, 5004]
        env = sc.make_env()
        for s in seeds[::-1]:                                          # dirty the env first
            run_episode(sc, controller_factory(name, **kw), s, env=env)
        for s in seeds:
            a = run_episode(sc, controller_factory(name, **kw), s, env=env)
            b = run_episode(sc, controller_factory(name, **kw), s)
            assert np.array_equal(a.tau_applied, b.tau_applied) and a.outcome == b.outcome


def test_push_t_teacher_episode_is_reproducible():
    sc = Scenario("push_t")
    a, b = run_episode(sc, "push_t_teacher", 5001), run_episode(sc, "push_t_teacher", 5001)
    assert np.array_equal(a.tau_applied, b.tau_applied) and a.outcome == b.outcome


def test_whip_teacher_episode_is_reproducible():
    sc = Scenario("whip")
    a = run_episode(sc, controller_factory("whip_teacher", **FAST_WHIP), 5001)
    b = run_episode(sc, controller_factory("whip_teacher", **FAST_WHIP), 5001)
    assert np.array_equal(a.tau_applied, b.tau_applied) and a.outcome == b.outcome


# ---- perturbations change the TRUE task only -------------------------------------------------------------------
def test_chain_perturbations_change_true_chain_not_nominal():
    sc = Scenario("whip").with_perturb(ChainMass(0.5), ChainDamping(2.0), ChainLength(0.1))
    env = sc.make_env()
    nom, true = env.ctx.make_model(), env.model
    for i in range(1, 11):
        assert true.body(f"c{i}").mass[0] == pytest.approx(1.5 * nom.body(f"c{i}").mass[0])
        assert true.dof_damping[2 + i] == pytest.approx(3.0 * nom.dof_damping[2 + i])
    assert true.body("c2").pos[0] == pytest.approx(1.1 * nom.body("c2").pos[0])
    info = env.ctx.task_info
    assert np.allclose(info["chain_mass"], 0.015) and np.allclose(info["chain_len"], 0.10) and np.allclose(info["chain_damp"], 0.0004)
    assert env.task.chain_mass[0] == pytest.approx(0.0225) and env.nominal_task.chain_mass[0] == 0.015
    # per-link spread: reproducible, uneven, mean of the logs ~ rel
    s1 = Scenario("whip").with_perturb(ChainMass(0.0, spread=0.3, pattern=2)).make_env().task.chain_mass
    s2 = Scenario("whip").with_perturb(ChainMass(0.0, spread=0.3, pattern=2)).make_env().task.chain_mass
    assert np.array_equal(s1, s2) and s1.std() > 0
    with pytest.raises(AttributeError):
        Scenario("push_t").with_perturb(ChainMass(0.5)).make_env()


def test_whip_target_does_not_depend_on_the_perturbation():
    a, b = Scenario("whip").make_env(), Scenario("whip").with_perturb(ChainLength(0.2), ChainMass(1.0)).make_env()
    assert np.array_equal(a.reset(5006).ref["target"], b.reset(5006).ref["target"])


def test_push_t_perturbations_change_true_block_not_nominal():
    sc = Scenario("push_t").with_perturb(TFriction(1.0), TMass(1.0), TComOffset(0.03, -0.02))
    env = sc.make_env()
    nom = env.ctx.make_model()
    t = env.task
    assert t.mu == pytest.approx(2 * env.nominal_task.mu) and t.t_mass == pytest.approx(2 * env.nominal_task.t_mass)
    assert env.model.body("tbody").mass[0] == pytest.approx(2 * nom.body("tbody").mass[0])
    assert env.model.geom("tbar").friction[0] == pytest.approx(2 * nom.geom("tbar").friction[0])
    assert env.model.dof_damping[2] == pytest.approx(2 * nom.dof_damping[2])         # friction x2 (mass alone does not change b)
    assert env.model.dof_frictionloss[2] == pytest.approx(4 * nom.dof_frictionloss[2])        # dry table force = mu m g: x2 mu, x2 mass
    assert nom.dof_frictionloss[2] > 0 and nom.dof_damping[2] > 0                               # mixed viscous + dry friction
    assert np.allclose(env.model.geom("tbar").pos - nom.geom("tbar").pos, [-0.03, 0.0, 0.02])    # geometry moves against the COM offset
    info = env.ctx.task_info
    assert info["mu"] == 0.5 and info["t_mass"] == 0.4
    # mass alone leaves the table friction coefficient b alone; friction alone scales it
    e2 = Scenario("push_t").with_perturb(TMass(1.0)).make_env()
    assert e2.model.dof_damping[2] == pytest.approx(nom.dof_damping[2])
    with pytest.raises(AttributeError):
        Scenario("whip").with_perturb(TMass(1.0)).make_env()


def test_com_offset_is_invisible_in_the_observation_of_the_outline():
    a, b = Scenario("push_t").make_env(), Scenario("push_t").with_perturb(TComOffset(0.04, 0.03)).make_env()
    oa, ob = a.reset(5002), b.reset(5002)
    assert np.allclose(oa.extra["t_pose"], ob.extra["t_pose"], atol=1e-12)         # same outline pose
    assert not np.allclose(a.data.qpos[2:4], b.data.qpos[2:4])                     # but the body (COM) frame is elsewhere
    # world position of the outline's corner is the same in both models
    for env in (a, b):
        import mujoco
        mujoco.mj_forward(env.model, env.data)
    ga, gb = a.data.geom_xpos[a.model.geom("tbar").id], b.data.geom_xpos[b.model.geom("tbar").id]
    assert np.allclose(ga, gb, atol=1e-9)


# ---- Push-T geometry / coverage ----------------------------------------------------------------------------------
def test_coverage_on_known_poses():
    goal = (0.55, 0.0, np.pi / 4)
    assert G.coverage(goal, goal) == pytest.approx(1.0)
    assert G.coverage((0.55 + 0.9, 0.0, np.pi / 4), goal) == 0.0                   # far away
    d = 0.02                                                                        # shift along the bar axis (u): bar and stem overlap shrink linearly
    assert G.coverage((0.55 + d, 0.0, 0.0), (0.55, 0.0, 0.0)) == pytest.approx(((G.W - d) * G.B + (G.S - d) * G.L) / G.AREA)
    d = 0.06                                                                        # shift beyond the stem width (0.05): only the bar overlaps
    assert G.coverage((0.55 + d, 0.0, 0.0), (0.55, 0.0, 0.0)) == pytest.approx((G.W - d) * G.B / G.AREA)
    # rotated by pi about the centroid: the bar of one lies on the stem side of the other
    cov = G.coverage((0.55, 0.0, np.pi), (0.55, 0.0, 0.0))
    assert 0.0 < cov < 0.5
    # symmetric in its arguments, rotation by 2 pi is the identity
    assert G.overlap_area((0.5, 0.01, 0.4), (0.52, -0.01, 0.1)) == pytest.approx(G.overlap_area((0.52, -0.01, 0.1), (0.5, 0.01, 0.4)))
    assert G.coverage((0.5, 0.0, 0.3 + 2 * np.pi), (0.5, 0.0, 0.3)) == pytest.approx(1.0)


def test_coverage_matches_a_raster_estimate():
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.3, 0.3, (400000, 2))

    def inside(pose):
        return G.inside_polygon((P - np.array(pose[:2])) @ G.rot(pose[2]), G.POLY)
    for a, b in [((0.0, 0.01, 0.4), (0.02, -0.01, 0.1)), ((0.05, 0.0, 2.0), (0.0, 0.02, -1.0)), ((0.0, 0.0, 0.0), (0.04, 0.03, 0.2))]:
        raster = (inside(a) & inside(b)).mean() * 0.36 / G.AREA
        assert G.coverage(a, b) == pytest.approx(raster, abs=0.01)


def test_push_t_extra_metrics_on_constructed_trace():
    task = make_task("push_t")
    T = 8
    cov = np.array([0.1, 0.2, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8])
    tip = np.column_stack([np.arange(T) * 0.1, np.zeros(T)])
    z = np.zeros((T, 2))
    ch = dict(cov=cov, tip=tip, t_pose=np.zeros((T, 3)))
    final = dict(err=np.zeros(0), q=np.zeros(2), channels=dict(cov=0.93, tip=np.array([0.8, 0.0]), t_pose=np.zeros(3)))
    tr = Trace(dt=0.005, n_steps=100, t_end=0.04, terminated=True, err=np.zeros((T, 0)), tau_cmd=z, tau_applied=z,
               sat=np.zeros(T, bool), pulse=np.zeros(T, bool), channels=ch, outcome="success", final=final)
    m = compute_metrics(tr, task)
    assert m["success"] == 1.0 and m["coverage"] == pytest.approx(0.93) and m["max_cov"] == pytest.approx(0.93)
    assert m["time_to_success"] == pytest.approx(0.04) and m["path_len"] == pytest.approx(0.8)
    tr2 = Trace(dt=0.005, n_steps=T, t_end=0.04, terminated=False, err=np.zeros((T, 0)), tau_cmd=z, tau_applied=z,
                sat=np.zeros(T, bool), pulse=np.zeros(T, bool), channels=ch, outcome="timeout")
    m2 = compute_metrics(tr2, task)
    assert m2["success"] == 0.0 and m2["coverage"] == pytest.approx(0.8) and m2["max_cov"] == pytest.approx(0.8)
    assert np.isnan(m2["time_to_success"]) and m2["path_len"] == pytest.approx(0.7)


def test_whip_extra_metrics_on_constructed_trace():
    task = make_task("whip")
    target = np.array([1.5, 0.0])
    T = 6
    tip = np.array([[1.0, 0.5], [1.2, 0.3], [1.4, 0.1], [1.45, 0.02], [1.6, 0.3], [1.7, 0.5]])      # row 3 is the first within r_hit = 0.08
    z = np.zeros((T, 3))
    ch = dict(whip_tip=tip[:5], target=np.tile(target, (5, 1)))
    final = dict(err=np.zeros(0), q=np.zeros(3), channels=dict(whip_tip=tip[5], target=target))
    tr = Trace(dt=0.01, n_steps=100, t_end=0.06, terminated=True, err=np.zeros((5, 0)), tau_cmd=z[:5], tau_applied=z[:5],
               sat=np.zeros(5, bool), pulse=np.zeros(5, bool), channels=ch, outcome="success", final=final)
    m = compute_metrics(tr, task)
    assert m["success"] == 1.0
    assert m["min_dist"] == pytest.approx(np.linalg.norm(tip - target, axis=1).min())
    assert m["hit_speed"] == pytest.approx(np.linalg.norm(tip[3] - tip[2]) / 0.01)          # speed at the first hit (step before -> hit)
    far = dict(ch, whip_tip=np.array([[1.0, 0.5]] * 5))
    fin2 = dict(final, channels=dict(whip_tip=np.array([1.0, 0.5]), target=target))
    miss = Trace(dt=0.01, n_steps=100, t_end=0.06, terminated=False, err=np.zeros((5, 0)), tau_cmd=z[:5], tau_applied=z[:5],
                 sat=np.zeros(5, bool), pulse=np.zeros(5, bool), channels=far, outcome="timeout", final=fin2)
    mm = compute_metrics(miss, task)
    assert mm["success"] == 0.0 and np.isnan(mm["hit_speed"]) and mm["min_dist"] == pytest.approx(np.linalg.norm([-0.5, 0.5]))


# ---- event logic inside the simulator --------------------------------------------------------------------------------
def test_whip_success_is_the_chain_tip_within_r():
    env = Scenario("whip").make_env()
    env.reset(5000)
    tip = env.task.chain_tip(env)
    env.task.target = tip + np.array([0.0, env.task.r_hit * 0.5])
    assert env.task.succeeded(env)
    env.task.target = tip + np.array([0.0, env.task.r_hit * 2.0])
    assert not env.task.succeeded(env)
    env.reset(5000)
    env.task.target = env.task.chain_tip(env) + np.array([0.0, -0.03])             # the next step (arm barely moves) is still inside r
    _, _, term, _, info = env.step(np.zeros(3))
    assert term and info["outcome"] == "success"


def test_push_t_success_and_failure_events():
    env = Scenario("push_t").make_env()
    env.reset(5000)
    n, goal = env.n, env.task.goal
    env.data.qpos[n:n + 2], env.data.qpos[n + 2] = goal[:2], goal[2]                # T exactly on the goal
    _, _, term, _, info = env.step(np.zeros(2))
    assert term and info["outcome"] == "success"
    env.reset(5000)
    env.data.qpos[n:n + 2] = goal[:2] + [0.01, 0.0]                                 # 1 cm off but upright: coverage < 1, still near 0.9?
    env.data.qpos[n + 2] = goal[2]
    cov = G.coverage((goal[0] + 0.01, goal[1], goal[2]), goal)
    _, _, term, _, info = env.step(np.zeros(2))
    assert (info["outcome"] == "success") == (cov >= 0.9)
    env.reset(5000)
    env.data.qpos[n:n + 2] = [1.5, 0.0]                                             # pushed off the table
    _, _, term, _, info = env.step(np.zeros(2))
    assert term and info["outcome"] == "failure"
    env.reset(5000)
    _, _, term, trunc, info = env.step(np.zeros(2))
    assert not term and info["outcome"] is None


def test_push_t_initial_states_are_valid_and_vary():
    covs, ths = [], []
    for s in DEMO_SEEDS:
        env = Scenario("push_t").make_env()
        env.reset(int(s))
        t = env.task
        covs.append(t.coverage_now(env))
        ths.append(float(G.wrap(t.t0_pose[2] - t.goal[2])))
        _, d = G.closest_on_polygon(t.p0[None], G.poly_world(t.t0_pose))
        assert d[0] > 0.10 and covs[-1] <= 0.3
    assert max(covs) < 0.3 and np.ptp(ths) > 1.5


# ---- quasi-static push model -------------------------------------------------------------------------------------
MODEL = PushModel(mu=0.5, rho2=G.RHO2, r_p=0.025)


def _push(pose, start, direction, dist, ds=0.002):
    pose = np.array([pose], float)
    P = np.array([start], float)
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    for _ in range(int(round(dist / ds))):
        pose, P = MODEL.step(pose, P, (u * ds)[None])
    return pose[0], P[0]


def test_quasi_static_central_push_translates_and_off_centre_push_rotates():
    # push the stem end straight up (along the symmetry axis through the COM): pure translation, T moves exactly as far as the pusher
    zb = G.POLY[:, 1].min()
    pose, P = _push((0.0, 0.0, 0.0), (0.0, zb - 0.025 - 0.003), (0.0, 1.0), 0.04)
    assert pose[2] == pytest.approx(0.0, abs=1e-9) and pose[0] == pytest.approx(0.0, abs=1e-9)
    assert pose[1] == pytest.approx(0.04 - 0.003, abs=0.004)
    # push the bar's left end to the right (+x) at its mid height: the contact is far above the COM -> rotation, sense = r x f
    zt = G.V_BAR
    pose, _ = _push((0.0, 0.0, 0.0), (-G.W / 2 - 0.025 - 0.003, zt), (1.0, 0.0), 0.03)
    assert pose[0] > 0.0 and abs(pose[2]) > 0.02
    # same push from the other end gives the opposite rotation
    pose2, _ = _push((0.0, 0.0, 0.0), (G.W / 2 + 0.025 + 0.003, zt), (-1.0, 0.0), 0.03)
    assert np.sign(pose2[2]) == -np.sign(pose[2])


def test_quasi_static_no_motion_when_pusher_moves_away_or_is_far():
    pose0 = np.array([[0.0, 0.0, 0.3]])
    P = pose0[:, :2] + np.array([[0.5, 0.0]])
    new, _ = MODEL.step(pose0, P, np.array([[0.002, 0.0]]))
    assert np.array_equal(new, pose0)
    zb = G.POLY[:, 1].min()
    pose, _ = _push((0.0, 0.0, 0.0), (0.0, zb - 0.025 - 0.001), (0.0, -1.0), 0.05)          # moving away from the block
    assert np.array_equal(pose, [0.0, 0.0, 0.0])


def test_quasi_static_sliding_when_outside_the_friction_cone():
    # push the right face of the bar with a direction 80 deg away from the normal: far outside the cone (atan 0.5 = 27 deg): it slides
    xs = G.W / 2
    ang = np.radians(80)
    d = np.array([-np.cos(ang), np.sin(ang)])
    pose, P = _push((0.0, 0.0, 0.0), (xs + 0.025 + 0.002, 0.0), d, 0.03)
    assert 0.0 < np.linalg.norm(pose[:2]) < 0.03 * np.cos(ang) + 0.01                      # moves with the normal component only (plus friction)
    stick, _ = _push((0.0, 0.0, 0.0), (xs + 0.025 + 0.002, 0.0), (-1.0, 0.0), 0.03)
    assert np.linalg.norm(stick[:2]) > np.linalg.norm(pose[:2])


def test_planner_prefers_pushes_that_reduce_the_pose_error():
    goal = np.array([0.55, 0.0, np.pi / 4])
    pl = PushPlanner(0.5, G.RHO2, 0.025, goal, depth=2, beam=6)
    pose = np.array([0.60, 0.04, goal[2] + 0.5])
    ranked = pl.plan(pose)
    assert ranked[0][1] < pose_cost(pose, goal) + 0.3 + 1e-9             # cost incl. 0.15 per primitive: the best leaf improves the pose
    worst = pl._expand(pose[None])[0]
    assert ranked[0][1] < np.max(pose_cost(worst, goal)) + 0.3
    # the exact goal needs no push: any push makes it worse
    assert pl.plan(goal.copy())[0][1] > pose_cost(goal, goal) + 0.15 - 1e-9


def test_approach_path_goes_around_the_block():
    pose = np.array([0.55, 0.0, 0.0])
    tip = np.array([0.55, 0.30])                                          # above the bar
    goal_pt = np.array([0.55, -0.132 - 0.028])                            # below the stem end
    gate = goal_pt + np.array([0.0, -0.06])
    path = approach_path(tip, goal_pt, gate, pose, 0.025)
    assert path is not None and len(path) >= 4
    poly = G.poly_world(pose)
    for a, b in zip(path[:-2], path[1:-1]):                               # every leg except the final straight approach keeps the clearance
        pts = a + np.linspace(0, 1, 30)[:, None] * (b - a)
        assert G.closest_on_polygon(pts, poly)[1].min() > 0.025
    assert path[0] is not None and np.allclose(path[0], tip) and np.allclose(path[-1], goal_pt)


# ---- model fidelity: the first push of the teacher does what the quasi-static model predicts ----------------------------
def test_first_push_matches_the_quasi_static_prediction():
    env = Scenario("push_t").make_env()
    ctrl = controller_factory("push_t_teacher")(env.ctx)
    obs = env.reset(5001)
    ctrl.reset(5001)
    for _ in range(int(10.0 / env.dt)):
        obs, *_ = env.step(ctrl.act(obs))
        if len(ctrl.plan_log) == 1 and not ctrl.legs and ctrl.hold > 0:        # first push finished, hold phase
            break
    log = ctrl.plan_log[0]
    for _ in range(int(0.4 / env.dt)):                                         # let the block settle
        obs, *_ = env.step(ctrl.act(obs))
        if len(ctrl.plan_log) > 1:
            break
    assert len(ctrl.plan_log) >= 1
    pred = ctrl.planner._expand(log["pose"][None])[0][log["prim"]]
    act = np.asarray(env.task.shape_pose(env))
    # compare at the moment the plan ended: use the pose measured at the end of the hold
    assert np.linalg.norm(act[:2] - pred[:2]) < 0.02 and abs(G.wrap(act[2] - pred[2])) < 0.12
    assert np.linalg.norm(pred[:2] - log["pose"][:2]) > 0.002 or abs(pred[2] - log["pose"][2]) > 0.01    # the push really moved the block


def test_pose_noise_touches_only_the_measurement():
    a = Scenario("push_t").make_env()
    b = Scenario("push_t").with_perturb(TPoseNoise(0.01, 0.1)).make_env()
    oa, ob = a.reset(5001), b.reset(5001)
    assert np.array_equal(oa.q, ob.q) and not np.array_equal(oa.extra["t_pose"], ob.extra["t_pose"])
    assert np.allclose(b.task.shape_pose(b), oa.extra["t_pose"])                        # the truth is untouched
    c = Scenario("push_t").with_perturb(TPoseNoise(0.01, 0.1)).make_env()
    assert np.array_equal(c.reset(5001).extra["t_pose"], ob.extra["t_pose"])           # reproducible per seed
    d = [b.step(np.zeros(2))[0].extra["t_pose"] for _ in range(20)]
    assert np.std([x[0] for x in d]) > 0.002


# ---- the teachers do their job (demo seeds, no perturbation), beat a no-search baseline, and the perturbations / latency hurt -----------
# Recorded values (results/whip_pusht/table.md, DEMO seeds 5000-5019, clean teacher): whip 20/20 = 100%, push_t 20/20 = 100% (random baselines 15% / 0%).
# The tests use the first few seeds and a tolerance (>= 75%) so that one unlucky seed does not break the suite.
N_WHIP, N_PUSHT = 8, 5


@pytest.fixture(scope="module")
def whip_clean():
    return evaluate(Scenario("whip"), "whip_teacher", DEMO_SEEDS[:N_WHIP])


@pytest.fixture(scope="module")
def pusht_clean():
    return evaluate(Scenario("push_t"), "push_t_teacher", DEMO_SEEDS[:N_PUSHT])


def test_whip_teacher_reaches_threshold_without_errors(whip_clean):
    assert whip_clean.metrics["success"].mean() >= 0.75
    ok = whip_clean.metrics["success"] > 0.5
    assert np.nanmean(whip_clean.metrics["hit_speed"][ok]) > 3.0       # a fling, not a poke: the tip is fast when it hits (m/s)


def test_whip_search_beats_random_keyframes(whip_clean):
    r = evaluate(Scenario("whip"), "whip_random", DEMO_SEEDS[:N_WHIP])
    assert r.metrics["success"].mean() <= whip_clean.metrics["success"].mean() - 0.5


def test_chain_length_error_hurts_the_whip_teacher(whip_clean):
    r = evaluate(Scenario("whip").with_perturb(ChainLength(-0.10)), "whip_teacher", DEMO_SEEDS[:N_WHIP])
    assert r.metrics["success"].mean() <= whip_clean.metrics["success"].mean() - 0.25
    assert np.nanmean(r.metrics["min_dist"]) > np.nanmean(whip_clean.metrics["min_dist"])


def test_push_t_teacher_reaches_threshold_without_errors(pusht_clean):
    assert pusht_clean.metrics["success"].mean() >= 0.75
    assert np.all(pusht_clean.metrics["max_cov"] >= pusht_clean.metrics["coverage"] - 1e-12)
    assert np.all(pusht_clean.metrics["coverage"][pusht_clean.metrics["success"] > 0.5] >= 0.9)


def test_push_t_search_beats_random_primitives(pusht_clean):
    r = evaluate(Scenario("push_t"), "push_t_random", DEMO_SEEDS[:N_PUSHT])
    assert r.metrics["success"].mean() <= pusht_clean.metrics["success"].mean() - 0.5


def test_sensor_latency_hurts_the_push_t_teacher(pusht_clean):
    from arena.perturb import Delay
    r = evaluate(Scenario("push_t").with_perturb(Delay(6, where="obs")), "push_t_teacher", DEMO_SEEDS[:N_PUSHT])
    assert r.metrics["success"].mean() <= pusht_clean.metrics["success"].mean() - 0.5
