"""MJX port vs CPU MuJoCo / arena. Never needs a GPU: the JAX parts run in subprocesses pinned to the JAX CPU backend
(CUDA_VISIBLE_DEVICES='' so no GPU is touched; the real GPU parity numbers are in results/mjx/)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sampling_matches_arena_tables():
    """ref tables, init error, pulse table = arena's own (same RNG draw order); numpy only"""
    from arena import Scenario
    from arena.perturb import TorquePulses
    from arena.robots import RobotSpec
    from arena_mjx.sampling import episode_batch
    spec = RobotSpec()
    seeds = np.array([5000, 5001, 7])
    b = episode_batch(seeds, spec, pulses=True)
    env = Scenario("tracking", {"ref": "joint_sines"}).with_perturb(TorquePulses()).make_env()
    for i, s in enumerate(seeds):
        env.reset(int(s))
        t = env.task
        assert np.array_equal(b["q"][i], t.Q) and np.array_equal(b["qd"][i], t.Qd) and np.array_equal(b["qdd"][i], t.Qdd)
        assert np.allclose(env.data.qpos[:2], t.Q[0] + b["ie"][i, :2], atol=0, rtol=0)
        assert np.array_equal(b["pulse"][i], env.perts[0].table)


def _run(x64, tmp_path):
    out = tmp_path / f"p{x64}.json"
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "mjx_parity.py"), "--x64", str(x64), "--tiny", "--out", str(out)],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(out))


@pytest.mark.parametrize("x64, tol_state, tol_rel", [(1, 1e-9, 1e-8), (0, 5e-3, 1e-3)])
def test_mjx_matches_cpu(x64, tol_state, tol_rel, tmp_path):
    """Part A: same MJCF/initial state/torque sequence -> joint trajectories; Part B: CTC+LQR rmse/ss_rmse/energy/sat per
    seed. float64 must agree to ~1e-9; float32 (the training precision) to the tolerances above."""
    res = _run(x64, tmp_path)
    for name, c in res["A_fixed_torque"]["cases"].items():
        assert c["max_dq"] < tol_state and c["max_dqd"] < tol_state * 10, (name, c)
    for name, c in res["B_ctc_lqr"].items():
        for m in ("rmse", "energy", "sat"):
            assert c[m]["max_rel"] < tol_rel or c[m]["max_abs"] < 1e-6, (name, m, c[m])
        assert c["ss_rmse"]["max_abs"] < 1e-6, (name, c["ss_rmse"])
