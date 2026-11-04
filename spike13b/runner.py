"""usage: runner.py <store.json> <seed_lo> <seed_hi> <cond> <method,method,...>   (resumable per (method,cond); per-seed lists keyed by seed)"""
import sys, os, json, time, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c13 import *
if __name__ == '__main__':
    store = os.path.join(OUT13, sys.argv[1]); lo, hi, cond = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; methods = sys.argv[5].split(','); seeds = list(range(lo, hi + 1)); [guard13(s) for s in seeds]
    pool = mp.Pool(int(os.environ.get('NW', 8)))
    for meth in methods:
        res = json.load(open(store)) if os.path.exists(store) else {}; k = f'{meth}|{cond}'
        if k in res and res[k]['seeds'] == seeds: continue
        t0 = time.time(); r = pool.map(run13, [(meth, s, cond) for s in seeds], chunksize=1)
        res = json.load(open(store)) if os.path.exists(store) else {}
        res[k] = dict(seeds=seeds, success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], inst=[x['inst'] for x in r], diverged=[x['diverged'] for x in r], wall=time.time() - t0)
        json.dump(res, open(store, 'w')); print(k, f"{100*np.mean(res[k]['success']):.1f}", 'unstable', np.mean(np.array(res[k]['inst']) > 0), f'{time.time()-t0:.0f}s', flush=True)
