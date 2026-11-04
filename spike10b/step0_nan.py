"""Step 0: source of the MuJoCo 'Nan, Inf or huge value ... unstable' warnings and their effect on episode metrics. CAL seeds 1000-1049 only.
Finding (see README): (i) planning proxies (SwingSim rollouts, _closed_loop re-scoring) hit the instability on purpose-rejected candidates (harmless, handled by the planner);
(ii) the TRUE plant (arena/env.py step) blows up numerically in some episodes (chain tip speed ~ 500 rad/s with RK4 at dt=2.5 ms); MuJoCo auto-resets and arena does NOT flag it diverged.
The counter in c10b.INST counts true-plant steps whose clock jumps back / state non-finite. SUBF=1 (as spike09/10a) vs SUBF=2 (fix: plant dt 1.25 ms, control rate unchanged)."""
import sys, os, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10b import *
if __name__ == '__main__':
    S = list(range(1000, 1050)); pool = mp.Pool(8); out = {}
    for spec, cond in [('nom', 'abc'), ('adapt', 'abc'), ('oracle', 'abc'), ('nom', 'none'), ('adapt', 'none'), ('oracle', 'x2')]:
        r = pool.map(run10, [(spec, s, None, cond) for s in S], chunksize=1)
        out[f'{spec}|{cond}'] = dict(SUBF=SUBF, n=len(S), diverged=float(np.mean([m['diverged'] for m in r])), unstable_episodes=float(np.mean([m['inst'] > 0 for m in r])), succ=float(np.mean([m['success'] for m in r]))); print(spec, cond, out[f'{spec}|{cond}'], flush=True)
    json.dump(out, open(os.path.join(OUT, f'step0_nan_subf{SUBF}.json'), 'w'), indent=1)
