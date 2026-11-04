import numpy as np, time
from sim import *
seeds = np.arange(50); P = sample_refs(seeds); ie = sample_init_err(seeds); ds = sample_dist(seeds)
for e in [0, .1, .3, -.3]:
    T = Teacher(e); t=time.time()
    r = rollout(T.act, P, ie); rd = rollout(T.act, P, ie, dist=ds)
    print(f"e={e:+.1f} rmse={r['rmse'].mean():.5f} max={r['rmse'].max():.5f} en={r['energy'].mean():.1f} | dist rmse={rd['rmse'].mean():.5f} | {time.time()-t:.1f}s")
# no-feedback-error reference: zero init err
r = rollout(Teacher(0).act, P, np.zeros_like(ie)); print("e=0 zero init err rmse", r['rmse'].mean())
