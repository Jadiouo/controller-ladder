"""10-episode per-episode dump (strongest classical, real condition, CAL seeds 1000-1009). usage: local_cmp.py OUT.json  (reads results/spike16a/<task>/classical.json)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h16 import *
import numpy as np, platform, scipy, mujoco
out = dict(env=dict(numpy=np.__version__, scipy=scipy.__version__, mujoco=mujoco.__version__, host=platform.node()), tasks={})
for task in ('catch', 'toss', 'push_t'):
    C = json.load(open(os.path.join(RES, task, 'classical.json'))); rows = []
    for sd in range(1000, 1010):
        e = Ep(task, sd, C['s'], C['params'], None, False); n = 0
        while not e.done: e.step(np.zeros(e.env.n)); n += 1
        rows.append(dict(seed=sd, outcome=e.outcome, steps=n * e.rep, smooth=e.smooth(), q_end=e.env.data.qpos[:e.env.n].tolist(), qpos_all=e.env.data.qpos.tolist()))
    out['tasks'][task] = rows; print(task, [r['outcome'] for r in rows], flush=True)
jdump(out, sys.argv[1])
