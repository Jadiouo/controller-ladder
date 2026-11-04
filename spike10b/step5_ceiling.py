"""Diagnostic (CAL 1000-1049 only, condition abc): ceiling of the 'regress parameters -> reconstruct chain state by replay -> plan' design.
'cheat-theta': the estimator returns the TRUE plant theta (perfect regression) but the probe + replay state reconstruction is kept (the L-est / T-adapt3 pipeline).
Compared with T-oracle (no probe, planner starts from the true rest state) and with the learned/fitted estimators on the same seeds. Not a method."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
class Cheat:
    def __init__(self, th): self.th = np.asarray(th, float)
    def __call__(self, ctx, q0, taus, Qm, Vm): return self.th
if __name__ == '__main__':
    S = list(range(1000, 1050)); pr = probe_of_lest(); pool = mp.Pool(8); out = {}; t0 = time.time()
    for name, mk in (('cheat-theta (true theta, probe+replay pipeline)', lambda s: ('est', f'cheat{s}', Cheat(true_theta(s, 'abc')), pr['T'], pr['amp'])), ('T-oracle', lambda s: 'oracle'), ('T-adapt3-full', lambda s: spec_of('adapt3_full')), ('L-est-full#0', lambda s: spec_of('lest_full#0'))):
        r = pool.map(run10, [(mk(s), s, None, 'abc') for s in S], chunksize=1); su = np.array([x['success'] for x in r]); out[name] = dict(succ=float(su.mean()), ci=A.boot_ci(su), min_dist=float(np.mean([x['min_dist'] for x in r])), wall=time.time() - t0); print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(OUT, 'ceiling_cal.json'), 'w'), indent=1)
