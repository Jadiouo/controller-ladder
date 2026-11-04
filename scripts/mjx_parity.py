"""Numerical parity MJX vs CPU MuJoCo / arena. Usage: python scripts/mjx_parity.py --x64 0|1 [--small] [--out f.json]
(run on GPU through a GPU job queue, or with JAX_PLATFORMS=cpu). Part A: same MJCF, same initial state, same torque sequence,
1000 control steps (2000 physics steps). Part B: CTC+LQR RMSE / energy / saturation per seed, arena (CPU) vs MJX."""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--x64", type=int, default=0); ap.add_argument("--small", action="store_true"); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--out", default=None)
a = ap.parse_args()
import arena_mjx
jax = arena_mjx.setup(x64=bool(a.x64))
import jax.numpy as jnp, numpy as np
from arena import Scenario
from arena.perturb import PayloadError, TorqueLimit, Delay, Friction, TorquePulses
from arena.eval import evaluate
from arena.eval.protocol import DEMO_SEEDS
from arena.robots import RobotSpec
from arena_mjx.sampling import episode_batch
from arena_mjx.env import MjxTracking, make_params
from arena_mjx.ctc import CtcLqr
from arena_mjx.rollout import rollout, replay

dtype = jnp.float64 if a.x64 else jnp.float32
spec = RobotSpec()
env = MjxTracking(spec, dtype)
seeds = DEMO_SEEDS[:2] if a.tiny else DEMO_SEEDS[:4] if a.small else DEMO_SEEDS
res = dict(device=str(jax.devices()[0]), x64=a.x64, matmul_precision=str(jax.config.jax_default_matmul_precision))

def scen(*ps): return Scenario("tracking", {"ref": "joint_sines"}).with_perturb(*ps)

CASES = {   # name -> (arena perturbations, mjx kwargs)
    "W0": ((), {}),
    "W0+pulses": ((TorquePulses(),), {}),
    "W1_cap32": ((TorqueLimit(32.0),), dict(cap=32.0)),
    "cap15_saturating": ((TorqueLimit(15.0),), dict(cap=15.0)),
    "W2_delay6": ((Delay(6),), dict(delay=6)),
    "W3_friction": ((Friction(1.5, 0.3),), dict(fc=1.5, fv=0.3)),
    "W4_all+pulses": ((TorqueLimit(32.0), Delay(6), Friction(1.5, 0.3), TorquePulses()), dict(cap=32.0, delay=6, fc=1.5, fv=0.3)),
    "payload+30%+pulses": ((PayloadError(rel=0.3), TorquePulses()), dict(payload=1.5 * 1.3)),
}
rep_j = jax.jit(lambda p, tt: replay(env, p, tt))      # jitted ONCE: cases differ only in parameter values
ro_j = jax.jit(lambda p: rollout(env, pol, p))
pulses_on = {"W0+pulses", "W4_all+pulses", "payload+30%+pulses"}

# ---------------- Part A: fixed torque sequence, 1000 control steps
T = 100 if a.tiny else 200 if a.small else 1000
A = {}
rng = np.random.default_rng(123)
t = np.arange(T)[:, None] * spec.dt
base = 12 * np.sin(2 * np.pi * np.array([0.7, 1.3]) * t + np.array([0.3, 1.1])) + 6 * np.sin(2 * np.pi * 2.9 * t)
taus = np.stack([base + rng.normal(0, 1.0, base.shape) + np.array([20.0, 8.0]) * 0 for _ in range(len(seeds))])
for name, (perts, kw) in CASES.items():
    if name in ("W1_cap32",):   # cap makes the sequence saturate: still the same chain on both sides
        pass
    sc = scen(*perts)
    cenv = sc.make_env()
    cq, cqd = [], []
    for i, s in enumerate(seeds):
        cenv.reset(int(s)); qs, vs = [], []
        for k in range(T):
            cenv.step(taus[i, k]); qs.append(cenv.data.qpos[:2].copy()); vs.append(cenv.data.qvel[:2].copy())
        cq.append(qs); cqd.append(vs)
    cq, cqd = np.array(cq), np.array(cqd)
    p = make_params(episode_batch(seeds, spec, pulses=name in pulses_on), dtype, **kw)
    jq, jqd = rep_j(p, jnp.asarray(taus[:, :T], dtype))
    jq, jqd = np.asarray(jq, np.float64), np.asarray(jqd, np.float64)
    dq, dqd = np.abs(jq - cq), np.abs(jqd - cqd)
    cks = [k for k in (25, 100, 200, 500, 1000) if k <= T]
    growth = {str(k): dict(max=float(dq[:, k - 1].max()), median=float(np.median(dq[:, k - 1].max(-1)))) for k in cks}
    A[name] = dict(max_dq=float(dq.max()), max_dqd=float(dqd.max()), final_dq=float(dq[:, -1].max()), growth=growth,
                   frac_traj_below_1e3=float((dq.max((1, 2)) < 1e-3).mean()),
                   rms_dq=float(np.sqrt((dq ** 2).mean())), q_range=float(np.abs(cq).max()))
res["A_fixed_torque"] = dict(steps=T, n_traj=len(seeds), cases=A)

# ---------------- Part B: CTC+LQR
B = {}
ctc = CtcLqr(env)
pol = lambda s, q, qd, qr, qdr, qddr: ctc.act(q, qd, qr, qdr, qddr)
for name, (perts, kw) in CASES.items():
    sc = scen(*perts)
    r = evaluate(sc, "ctc_lqr", seeds, split="demo")
    p = make_params(episode_batch(seeds, spec, pulses=name in pulses_on), dtype, **kw)
    t0 = time.time()
    out = ro_j(p)
    out = {k: np.asarray(v, np.float64) for k, v in out.items()}
    d = {}
    for m in ("rmse", "ss_rmse", "energy", "sat"):
        c = np.asarray(r.metrics[m], np.float64)
        d[m] = dict(max_abs=float(np.abs(out[m] - c).max()), max_rel=float((np.abs(out[m] - c) / np.maximum(np.abs(c), 1e-9)).max()),
                    cpu_mean=float(c.mean()), mjx_mean=float(out[m].mean()))
    B[name] = d
res["B_ctc_lqr"] = B
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
