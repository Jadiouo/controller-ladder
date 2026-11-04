"""Step 1c: holdout 1100-1399 (n=300; cut from 500 for wall-clock, v7 requires n>=300), condition abc, plant with integration fix. Methods from argv[1] (comma list; default nom,robust,adapt,adapt2,adapt3_grad,adapt3_full,oracle).
best-classical = highest holdout success among nom, robust, adapt, adapt2, adapt3_grad, adapt3_full (ties: lower mean min_dist). Resumable per method."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from r10b import *
if __name__ == '__main__':
    S = list(range(1100, 1400)); guard(S); f = os.path.join(OUT, "holdout1.json"); res = json.load(open(f)) if os.path.exists(f) else {}
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['nom', 'robust', 'adapt', 'adapt2', 'adapt3_grad', 'adapt3_full', 'oracle']; pool = mp.Pool(8); t0 = time.time()
    for n in names:
        if n in res: continue
        r = pool.map(run10, [(spec_of(n), s, None, 'abc') for s in S], chunksize=1); t1 = time.time() - t0
        res[n] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], inst=[x['inst'] for x in r], diverged=[x['diverged'] for x in r], theta=[x['theta'] for x in r], wall=t1)
        su = np.array(res[n]['success']); print(n, su.mean(), A.boot_ci(su), 'unstable', np.mean(np.array(res[n]['inst']) > 0), f'{t1:.0f}s', flush=True); json.dump(res, open(f, 'w')); t0 = time.time()
    cl = [n for n in ('nom', 'robust', 'adapt', 'adapt2', 'adapt3_grad', 'adapt3_full') if n in res]
    if len(cl) == 6:
        sel = max(cl, key=lambda n: (round(float(np.mean(res[n]['success'])), 6), -float(np.nanmean(res[n]['min_dist']))))
        pm = 'grad' if np.mean(res['adapt3_grad']['success']) >= np.mean(res['adapt3_full']['success']) else 'full'; cfg = jload(f'tune_{pm}.json')['chosen']
        json.dump(dict(best_classical=sel, spec=list(spec_of(sel)) if not isinstance(spec_of(sel), str) else spec_of(sel), success={n: float(np.mean(res[n]['success'])) for n in res}, n=len(S)), open(os.path.join(OUT, 'best_classical.json'), 'w'), indent=1, default=str)
        json.dump(dict(pm=pm, T=cfg['T'], amp=cfg['amp']), open(os.path.join(OUT, 'probe_for_lest.json'), 'w')); print('best-classical', sel, 'probe for L-est:', pm, cfg['T'], cfg['amp'])
