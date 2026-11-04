import mujoco
import numpy as np
import pytest

from arena import Scenario
from arena.robots import RobotSpec, build_mjcf, fk, ik, jacobian, jdot_qd
from arena.tasks import make_task


@pytest.mark.parametrize("spec", [RobotSpec.two_link(), RobotSpec.three_link(), RobotSpec.two_link(tool="paddle"),
                                  RobotSpec.three_link(tool="gripper", payload_mass=0.0)])
def test_kinematics_match_mujoco(spec):
    m = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    rng = np.random.default_rng(0)
    for _ in range(5):
        q, qd = rng.uniform(-1, 1, spec.n), rng.uniform(-1, 1, spec.n)
        d.qpos[:], d.qvel[:] = q, qd
        mujoco.mj_forward(m, d)
        assert np.allclose(d.site_xpos[sid][[0, 2]], fk(spec, q), atol=1e-12)
        Jp = np.zeros((3, spec.n))
        mujoco.mj_jacSite(m, d, Jp, None, sid)
        assert np.allclose(Jp[[0, 2]], jacobian(spec, q), atol=1e-12)
        eps = 1e-6
        d.qpos[:] = q + eps * qd
        mujoco.mj_forward(m, d)
        Jp2 = np.zeros((3, spec.n))
        mujoco.mj_jacSite(m, d, Jp2, None, sid)
        assert np.allclose((Jp2[[0, 2]] - Jp[[0, 2]]) / eps @ qd, jdot_qd(spec, q, qd), atol=1e-4)


def test_robot_spec_variants_change_mass_and_validate():
    base = mujoco.MjModel.from_xml_string(build_mjcf(RobotSpec.two_link()))
    heavy = mujoco.MjModel.from_xml_string(build_mjcf(RobotSpec.two_link(payload_mass=3.0)))
    paddle = mujoco.MjModel.from_xml_string(build_mjcf(RobotSpec.two_link(tool="paddle", tool_mass=0.3)))
    assert heavy.body_mass.sum() == pytest.approx(base.body_mass.sum() + 1.5)
    assert paddle.body_mass.sum() == pytest.approx(base.body_mass.sum() + 0.3)
    assert mujoco.MjModel.from_xml_string(build_mjcf(RobotSpec.three_link())).nv == 3
    with pytest.raises(ValueError):
        RobotSpec(n=2, link_lengths=(1.0,))
    with pytest.raises(ValueError):
        RobotSpec(tool="laser")


def test_ik_roundtrip():
    s = RobotSpec.three_link()
    tgt = np.array([0.55, 0.1])
    q = ik(s, tgt, np.array([0, 1.2, 0.4]))
    assert np.linalg.norm(fk(s, q) - tgt) < 1e-9


@pytest.mark.parametrize("kind", ["lissajous", "glyph", "joint_sines"])
@pytest.mark.parametrize("n", [2, 3])
def test_tracking_reference_is_consistent(kind, n):
    robot = RobotSpec.two_link() if n == 2 else RobotSpec.three_link()
    if kind == "joint_sines" and n == 3:
        pytest.skip("joint_sines is the 2-DoF legacy reference (extended with default ranges, not validated)")
    sc = Scenario("tracking", {"ref": kind}, robot=robot)
    env = sc.make_env()
    env.reset(1234)
    t = env.task
    # ee column is FK of q
    assert np.allclose(t.EE, [fk(robot, q) for q in t.Q])
    # velocity / acceleration match finite differences of position / velocity
    dt = robot.dt
    assert np.abs((t.Q[2:] - t.Q[:-2]) / (2 * dt) - t.Qd[1:-1]).max() < 5e-3 * (1 + np.abs(t.Qd).max())
    assert np.abs((t.Qd[2:] - t.Qd[:-2]) / (2 * dt) - t.Qdd[1:-1]).max() < 5e-2 * (1 + np.abs(t.Qdd).max())
    if kind != "joint_sines":      # a Cartesian figure stays inside the reachable annulus and moves
        r = np.linalg.norm(t.EE, axis=1)
        assert r.min() > 0.1 and r.max() < sum(robot.link_lengths) - 0.02 and np.ptp(t.EE[:, 0]) > 0.05


def test_tracking_reference_depends_only_on_seed():
    a, b = Scenario("tracking").make_env(), Scenario("tracking").make_env()
    a.reset(9); b.reset(9)
    assert np.array_equal(a.task.Q, b.task.Q)
    b.reset(10)
    assert not np.array_equal(a.task.Q, b.task.Q)


def test_pole_world_angle_matches_mujoco_geometry():
    env = Scenario("pole_balance").make_env()
    env.reset(4)
    mujoco.mj_forward(env.model, env.data)
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
    zaxis = env.data.xmat[bid].reshape(3, 3)[:, 2]               # pole axis in world
    assert np.arctan2(zaxis[0], zaxis[2]) == pytest.approx(float(env.task.measure(env)["pole_theta"]), abs=1e-12)


def test_pole_falls_like_an_inverted_pendulum():
    """Uncontrolled (gravity-compensated, static) arm: early on theta(t) = theta0 * cosh(sqrt(g/L) t)."""
    from arena.controllers import controller_factory
    env = Scenario("pole_balance").make_env()
    env.reset(0)
    th0 = 0.05
    env.data.qpos[:2] = env.task.q_hold
    env.data.qpos[2] = th0 - env.task.q_hold.sum()
    env.data.qvel[:] = 0
    ctrl = controller_factory("pid", c_i=0.0, c_p=0.0, c_d=0.0, anchor=1.0)(env.ctx)
    obs = env._observe()
    for _ in range(40):                                            # 0.2 s
        obs, *_ = env.step(ctrl.act(obs))
    th = float(env.task.measure(env)["pole_theta"])
    expect = th0 * np.cosh(np.sqrt(9.81 / env.task.L) * 0.2)
    assert th == pytest.approx(expect, rel=0.1)
