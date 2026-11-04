"""Student controller: action-chunk execution order, HoldWrapper compatibility, obs-only access, determinism, timing."""
import dataclasses
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena import Scenario
from arena.controllers import controller_factory
from arena.eval import DEMO_SEEDS, TRAIN_SEEDS, run_episode
from arena.learn import collect
from arena.learn.features import FeatureSpec, layout_of
from arena.learn.train import Loaded, TrainCfg, train

H, TA = 2, 6
SC = Scenario("tracking", {"ref": "lissajous", "duration": 0.5}, history=H)


class Stub(torch.nn.Module):
    """Stands in for a trained policy: call c returns the chunk [[1000 c + j, 1000 c + j + 0.5] for j in range(TA)]."""
    chunk = TA

    def __init__(self):
        super().__init__()
        self.calls = 0

    def act_chunk(self, feat, gen=None, **kw):
        j = torch.arange(TA, dtype=torch.float32)
        out = torch.stack([1000 * self.calls + j, 1000 * self.calls + j + 0.5], -1)[None]
        self.calls += 1
        return out


def stub_loaded():
    env = SC.make_env()
    obs = env.reset(0)
    return Loaded(Stub(), FeatureSpec(layout_of(obs), H), {}, torch.device("cpu"))


def drive(ctrl, steps, seed=0):
    env = SC.make_env()
    obs = env.reset(seed)
    ctrl = ctrl(env.ctx) if callable(ctrl) else ctrl
    ctrl.reset(seed)
    out = []
    for _ in range(steps):
        a = np.asarray(ctrl.act(obs))
        out.append(a)
        obs = env.step(np.zeros(2))[0]
    return out


def test_receding_horizon_executes_chunk_in_order():
    # the stub's values are far above the torque limit -> widen the limit so that clipping does not hide the order
    acts = drive(lambda ctx: _unclipped(controller_factory("student", loaded=stub_loaded(), exec_k=3)(ctx)), 9)
    first = [a[0] for a in acts]
    assert first == [0, 1, 2, 1000, 1001, 1002, 2000, 2001, 2002]       # 3 of 6 actions per call, then re-plan
    assert all(a[1] == a[0] + 0.5 for a in acts)


def _unclipped(c):
    c.tau_limit = 1e9
    return c


def test_full_chunk_default_and_exec_k_validation():
    first = [a[0] for a in drive(lambda ctx: _unclipped(controller_factory("student", loaded=stub_loaded())(ctx)), 8)]
    assert first == [0, 1, 2, 3, 4, 5, 1000, 1001]
    with pytest.raises(ValueError):
        controller_factory("student", loaded=stub_loaded(), exec_k=TA + 1)(SC.make_env().ctx)


def test_hold_wrapper_chunk_mode_equals_internal_receding_horizon():
    k = 4

    def inner(ctx):
        return _unclipped(controller_factory("student", loaded=stub_loaded(), exec_k=k)(ctx))

    def base(ctx):
        return _unclipped(controller_factory("student", loaded=stub_loaded(), exec_k=TA, chunk_out=True)(ctx))
    hold = controller_factory("hold", base=base, every=k, chunk=True)
    a = np.array(drive(inner, 12))
    b = np.array(drive(hold, 12))
    assert np.array_equal(a, b) and a[:, 0].tolist()[:5] == [0, 1, 2, 3, 1000]


def test_student_only_gets_the_obs():
    ctx = SC.make_env().ctx

    def boom(*a, **k):
        raise AssertionError("student touched the plant / nominal model / reference function")
    blind = dataclasses.replace(ctx, make_model=boom, ref_fn=boom, rng=None)
    c = controller_factory("student", loaded=stub_loaded())(blind)
    env = SC.make_env()
    obs = env.reset(0)
    c.reset(0)
    c.act(obs.replace(ref_fn=None))
    import arena.controllers.student as mod
    src = inspect.getsource(mod)
    assert "make_model" not in src.replace("ctx.make_model", "").split("Student")[1] and "env" not in c.__dict__
    # without Obs.history the student refuses instead of silently using a wrong window
    env0 = Scenario("tracking", {"ref": "lissajous", "duration": 0.5}).make_env()
    c.reset(0)
    with pytest.raises(ValueError):
        c.act(env0.reset(0))


@pytest.fixture(scope="module")
def tiny_ckpts(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    collect(SC, "ctc_lqr", TRAIN_SEEDS[:3], d / "data", shard_episodes=3)
    cfg = TrainCfg(steps=40, batch=64, chunk=TA, width=32, depth=1, log_every=20, warmup=5)
    paths = {}
    for kind in ("bc", "mip", "diffusion", "flow"):
        paths[kind] = d / f"{kind}.pt"
        train(d / "data", kind, paths[kind], cfg, device="cpu", verbose=False)
    return paths


@pytest.mark.parametrize("kind", ["bc", "mip", "diffusion", "flow"])
def test_checkpoint_policy_runs_deterministically_and_times_itself(tiny_ckpts, kind):
    made = []
    f = controller_factory("student", ckpt=str(tiny_ckpts[kind]), exec_k=3, sample_kw={"steps": 2} if kind in ("diffusion", "flow") else {})
    tr1 = run_episode(SC, lambda ctx: made.append(f(ctx)) or made[-1], int(DEMO_SEEDS[0]))
    tr2 = run_episode(SC, f, int(DEMO_SEEDS[0]))
    assert np.array_equal(tr1.tau_cmd, tr2.tau_cmd)                     # same seed -> same torques (also for the samplers)
    t = made[0].timing()
    assert t["n"] == int(np.ceil(len(tr1.tau_cmd) / 3)) and t["mean_ms"] > 0
    if kind in ("diffusion", "flow"):
        tr3 = run_episode(SC, f, int(DEMO_SEEDS[1]))
        assert tr3.tau_cmd.shape == tr1.tau_cmd.shape
