"""Policy-level parity: a saved residual actor evaluated by MJX (mean action) vs the CPU arena `residual` controller, on
DEMO seeds (never TEST). Usage: python scripts/mjx_policy_parity.py <policy.npz> --w W3 [--scale 20] [--x64 0]
--scale multiplies the output layer so the residual is large (a stricter test than a barely-trained net)."""
import argparse, os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--w", default="W3"); ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--x64", type=int, default=0); ap.add_argument("--out", default=None)
a = ap.parse_args()
import arena_mjx
jax = arena_mjx.setup(x64=bool(a.x64))
import jax.numpy as jnp, numpy as np
from arena import Scenario
from arena.perturb import Friction, TorquePulses
from arena.eval import evaluate
from arena.eval.protocol import DEMO_SEEDS
from arena.controllers.residual import Residual, MLPPolicy
from arena.robots import RobotSpec
from arena_mjx.sampling import episode_batch
from arena_mjx.env import MjxTracking, make_params
from arena_mjx.ctc import CtcLqr
from arena_mjx.rollout import rollout
from arena_mjx.ppo import mlp, feats
z = np.load(a.npz); state = {k: z[k] for k in z.files if k.endswith(("weight", "bias"))}
state["4.weight"] = state["4.weight"] * a.scale
dtype = jnp.float64 if a.x64 else jnp.float32
layers = [(jnp.asarray(state[f"{i}.weight"], dtype), jnp.asarray(state[f"{i}.bias"], dtype)) for i in (0, 2, 4)]
spec = RobotSpec(); env = MjxTracking(spec, dtype); ctc = CtcLqr(env)
def pol(s, q, qd, qr, qdr, qddr):
    tT = ctc.act(q, qd, qr, qdr, qddr)
    return jnp.clip(tT + 16.0 * jnp.clip(mlp(layers, feats(q, qd, qr, qdr, qddr, tT).astype(dtype)), -1, 1), -80, 80)
perts = dict(W0=(), W3=(Friction(1.5, 0.3),))[a.w]; kw = dict(W0={}, W3=dict(fc=1.5, fv=0.3))[a.w]
res = {}
for cond, extra in (("clean", ()), ("dist", (TorquePulses(),))):
    sc = Scenario("tracking", {"ref": "joint_sines"}).with_perturb(*perts, *extra)
    r = evaluate(sc, lambda ctx: Residual(ctx, base="ctc_lqr", policy=MLPPolicy(state), bound=16.0), DEMO_SEEDS, split="demo")
    p = make_params(episode_batch(DEMO_SEEDS, spec, pulses=bool(extra)), dtype, **kw)
    o = jax.jit(lambda p: rollout(env, pol, p))(p)
    res[cond] = {m: dict(max_abs=float(np.abs(np.asarray(o[m], np.float64) - r.metrics[m]).max()), cpu=float(r.metrics[m].mean()), mjx=float(np.asarray(o[m]).mean())) for m in ("rmse", "energy", "sat")}
print(json.dumps(res, indent=1))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
