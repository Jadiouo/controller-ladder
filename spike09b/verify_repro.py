"""Reproduce T-adapt on holdout seeds 1100-1115 with a FRESH plan/estimate cache and compare with results/spike09b/holdout.json (guards against stale caches / code drift)."""
import sys, os, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import c9b; from c9b import *
c9b.PLANS = os.path.join(ROOT, '.scratch/s09b_verify'); os.makedirs(c9b.PLANS, exist_ok=True)
if __name__ == '__main__':
    HO = json.load(open(os.path.join(OUT, 'holdout.json'))); S = list(range(1100, 1116)); r = mp.Pool(8).map(run_spec, [('adapt', s, None, False) for s in S], chunksize=1)
    ds = np.array([x['min_dist'] for x in r]) - np.array(HO['adapt']['min_dist'][:16]); ss = np.array([x['success'] for x in r]) - np.array(HO['adapt']['success'][:16])
    print('max |d min_dist|', np.abs(ds).max(), 'success mismatches', int((ss != 0).sum()))
    json.dump(dict(max_abs_dmin=float(np.abs(ds).max()), success_mismatch=int((ss != 0).sum()), n=16), open(os.path.join(OUT, 'verify_repro.json'), 'w'))
