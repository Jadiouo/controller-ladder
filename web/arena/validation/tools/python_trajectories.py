"""Closed-loop joint trajectories of the Python arena (PID, CTC+LQR; clean and payload +30% / 2-step delay), 400 control steps,
tracking glyph 'S', calibration seed 1000. Writes ../python_traj.json, which ../../validate.mjs compares against."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))   # repo root
import numpy as np
from arena import Scenario
from arena.controllers.base import as_factory
from arena.perturb import PayloadError, Delay

CASES = {"clean": [], "hard": [PayloadError(rel=0.3), Delay(2)]}
out = {"seed": 1000, "glyph": "S", "steps": 400, "cases": {}}
for cname, ps in CASES.items():
    for cn in ("pid", "ctc_lqr"):
        sc = Scenario("tracking", {"ref": "glyph", "glyph": "S"}).with_perturb(*ps)
        env = sc.make_env(); obs = env.reset(1000)
        c = as_factory(cn)(env.ctx); c.reset(1000)
        Q = [env.data.qpos[:2].copy()]; TAU = []
        for k in range(400):
            obs, *_ = env.step(c.act(obs))
            Q.append(env.data.qpos[:2].copy()); TAU.append(env.data.ctrl[:2].copy())
        out["cases"][f"{cname}/{cn}"] = dict(q=np.array(Q).tolist(), tau=np.array(TAU).tolist())
json.dump(out, open(os.path.join(HERE, "..", "python_traj.json"), "w"), separators=(",", ":"))
