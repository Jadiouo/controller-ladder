"""Step 1-2: CAL 1000-1049, conditions none/a/b/c/abc x methods nom/robust/adapt/oracle_in/oracle_full. Saves results/spike10a/cal_base.json (resumable)."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10a import *
import c9b
if __name__ == '__main__':
    S = list(range(1000, 1050)); guard(S); f = os.path.join(OUT, 'cal_base.json'); res = json.load(open(f)) if os.path.exists(f) else {}
    methods = sys.argv[1].split(',') if len(sys.argv) > 1 else ['nom', 'robust', 'oracle_in', 'oracle_full', 'adapt']; conds = sys.argv[2].split(',') if len(sys.argv) > 2 else list(COND)
    pool = mp.Pool(8); t0 = time.time()
    for cond in conds:
        for m in methods:
            k = f'{m}|{cond}'
            if k in res: continue
            r = pool.map(run10, [(m, s, cond) for s in S], chunksize=1); su = np.array([x['success'] for x in r])
            res[k] = dict(succ=float(su.mean()), ci=A.boot_ci(su), min_dist=float(np.nanmean([x['min_dist'] for x in r])), diverged=float(np.mean([x['diverged'] for x in r])), per_seed=[x['success'] for x in r], wall=time.time() - t0)
            print(k, res[k]['succ'], res[k]['ci'], f"{time.time()-t0:.0f}s", flush=True); json.dump(res, open(f, 'w'), indent=1)
