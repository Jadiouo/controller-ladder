"""Interface contracts: what every task / controller / perturbation must satisfy, plus extensibility by new files."""
import numpy as np
import pytest

import arena
from arena import ArenaEnv, ControlContext, Obs, Scenario
from arena.controllers import CONTROLLERS, Controller, register_controller
from arena.controllers.base import controller_factory
from arena.eval import DEMO_SEEDS, run_episode
from arena.robots import RobotSpec
from arena.tasks import TASKS, Task, make_task, register_task

TASK_NAMES = ["tracking", "pole_balance"]


def test_registries_populated():
    assert {"tracking", "pole_balance"} <= set(TASKS.names())
    assert {"pid", "ctc_lqr", "residual"} <= set(CONTROLLERS.names())


@pytest.mark.parametrize("task", TASK_NAMES)
@pytest.mark.parametrize("n", [2, 3])
def test_env_contract(task, n):
    robot = (RobotSpec.two_link if n == 2 else RobotSpec.three_link)(payload_mass=0.5)
    env = Scenario(task, robot=robot).make_env()
    obs = env.reset(7)
    assert isinstance(obs, Obs) and obs.q.shape == (n,) and obs.qd.shape == (n,) and obs.k == 0 and obs.t == 0.0
    assert {"q", "qd", "qdd", "ee"} <= set(obs.ref)
    assert isinstance(env.ctx, ControlContext) and env.ctx.task == task
    obs2, reward, terminated, truncated, info = env.step(np.zeros(n))
    assert obs2.k == 1 and obs2.t == pytest.approx(env.dt) and isinstance(terminated, bool) and isinstance(truncated, bool) and isinstance(reward, float)
    assert info["applied"].shape == (n,) and "saturated" in info
    assert env.error().ndim == 1
    assert obs.to_vector().dtype == np.float32


@pytest.mark.parametrize("task", TASK_NAMES)
@pytest.mark.parametrize("name", ["pid", "ctc_lqr"])
def test_controller_contract(task, name):
    env = Scenario(task).make_env()
    ctrl = controller_factory(name)(env.ctx)
    assert isinstance(ctrl, Controller)
    obs = env.reset(3)
    ctrl.reset()
    tau = ctrl.act(obs)
    assert tau.shape == (2,) and np.all(np.isfinite(tau)) and np.all(np.abs(tau) <= env.ctx.tau_limit)
    ctrl.reset()                                    # reset is idempotent / repeatable
    assert np.array_equal(ctrl.act(obs), tau)


def test_unsupported_task_is_refused_loudly():
    env = Scenario("tracking").make_env()
    bad = ControlContext(env.nominal, "no_such_task", env.dt, 80.0)
    with pytest.raises(NotImplementedError):
        controller_factory("pid")(bad)


def test_episode_ends_at_horizon():
    tr = run_episode(Scenario("tracking"), "pid", int(DEMO_SEEDS[0]))
    assert len(tr.err) == tr.n_steps == 800 and not tr.terminated


def test_new_task_and_controller_need_only_new_code():
    """A task and a controller defined in a brand-new file-like scope plug into env/eval with no core edit."""

    @register_task("_test_hold")
    class Hold(Task):
        name = "_test_hold"
        duration = 0.5

        def reset(self, seed, env):
            env.data.qpos[:2] = [0.1, 1.0]

        def reference(self, k):
            return dict(q=np.array([0.1, 1.0]), qd=np.zeros(2), qdd=np.zeros(2), ee=np.zeros(2))

        def error(self, env):
            return env.data.qpos[:2] - np.array([0.1, 1.0])

    @register_controller("_test_zero")
    class Zero(Controller):
        def act(self, obs):
            return np.zeros(2)

    from arena.eval import evaluate
    r = evaluate(Scenario("_test_hold"), "_test_zero", DEMO_SEEDS[:2])
    assert r.metrics["rmse"].shape == (2,) and np.all(r.metrics["rmse"] > 0)   # arm falls under gravity
    assert make_task("_test_hold").duration == 0.5
