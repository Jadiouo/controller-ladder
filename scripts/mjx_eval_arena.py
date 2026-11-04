"""Evaluate MJX-trained residual policies on the CPU arena evaluator, TEST_SEEDS, protocol v2 (split='test', logged).
  python scripts/mjx_eval_arena.py --tag b2m --w W3 --seeds 0,1,2,3,4  -> results/mjx/eval_<tag>_<W>.pkl
Scenario = tracking(joint_sines) [+ Friction(1.5, .3) for W3] [+ TorquePulses() for the 'dist' condition]; controller = arena
`residual` (CTC+LQR base, bound 16 Nm, spike_features) with the saved actor (numpy MLP, mean action)."""
import argparse, os, pickle, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--w", required=True); ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--pdir", default=os.path.join(ROOT, "results", "mjx", "policies"))
a = ap.parse_args()
import numpy as np
from arena import Scenario
from arena.perturb import Friction, TorquePulses
from arena.eval import evaluate, TEST_SEEDS
from arena.controllers.base import controller_factory
from arena.controllers.residual import Residual, MLPPolicy

perts = dict(W0=(), W3=(Friction(1.5, 0.3),))[a.w]
conds = dict(clean=Scenario("tracking", {"ref": "joint_sines"}).with_perturb(*perts),
             dist=Scenario("tracking", {"ref": "joint_sines"}).with_perturb(*perts, TorquePulses()))
out = {}
t0 = time.time()
out[("ctc_lqr", 0)] = {c: evaluate(sc, "ctc_lqr", TEST_SEEDS, split="test").metrics for c, sc in conds.items()}
for s in [int(x) for x in a.seeds.split(",")]:
    z = np.load(os.path.join(a.pdir, f"{a.tag}_{a.w}_s{s}.npz"))
    state = {k: z[k] for k in z.files if k.endswith(("weight", "bias"))}
    made = []
    def fac(ctx, state=state):
        made.append(Residual(ctx, base="ctc_lqr", policy=MLPPolicy(state), bound=16.0)); return made[-1]
    out[("rl", s)] = {c: evaluate(sc, fac, TEST_SEEDS, label=f"{a.tag}_{a.w}_s{s}", split="test").metrics for c, sc in conds.items()}
    mx = max(m.max_residual for m in made)
    assert mx <= 16.0 + 1e-6, mx
    out[("rl", s)]["max_residual"] = mx
    r = out[("rl", s)]
    print(f"{a.tag} {a.w} s{s}: clean {r['clean']['rmse'].mean():.4f} dist {r['dist']['rmse'].mean():.4f} (ctc {out[('ctc_lqr',0)]['clean']['rmse'].mean():.4f}/{out[('ctc_lqr',0)]['dist']['rmse'].mean():.4f}) maxres {mx:.2f} t {time.time()-t0:.0f}s", flush=True)
pickle.dump(out, open(os.path.join(ROOT, "results", "mjx", f"eval_{a.tag}_{a.w}.pkl"), "wb"))
