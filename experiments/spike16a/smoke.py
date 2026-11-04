import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h16 import *
for task in ('catch', 'toss', 'push_t'):
    p = default_params(task); t = time.time(); out = []
    for sd in (1000, 1001, 1002):
        e = Ep(task, sd, 1.0, p, None, True); n = 0
        while not e.done: e.step(np.zeros(e.env.n)); n += 1
        out.append((e.outcome, round(e.smooth(), 3), n, len(e.x), {k: round(v, 3) for k, v in e.cond.items()}))
    print(task, 'dim', feat_dim(task), (time.time() - t) / 3, 's/ep'); [print('  ', o) for o in out]
    # extended teacher with non-default params still runs
    q = u2p(task, p2u(task, p) * 0 + 0.5)
    e = Ep(task, 1003, 1.0, q, None, True)
    while not e.done: e.step(np.zeros(e.env.n))
    print('  mid-params', e.outcome)
