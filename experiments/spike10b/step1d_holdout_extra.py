"""Holdout 1100-1399 (n=300), condition abc, for the learned methods (secondary H11 comparison on holdout) -> holdout_extra.json. Run AFTER freeze (does not touch frozen files)."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
if __name__ == '__main__':
    S = list(range(1100, 1400)); guard(S); f = os.path.join(OUT, 'holdout_extra.json'); res = json.load(open(f)) if os.path.exists(f) else {}
    names = sys.argv[1].split(','); pool = mp.Pool(8); t0 = time.time()
    for n in names:
        if n in res: continue
        r = pool.map(run10, [(spec_of(n), s, None, 'abc') for s in S], chunksize=1)
        res[n] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], inst=[x['inst'] for x in r], diverged=[x['diverged'] for x in r], theta=[x['theta'] for x in r], wall=time.time() - t0)
        print(n, np.mean(res[n]['success']), f'{time.time()-t0:.0f}s', flush=True); json.dump(res, open(f, 'w')); t0 = time.time()
