"""Step 4: holdout 1100-1599, condition abc: nom, robust, adapt, adapt2 (chosen on CAL), oracle_in, oracle_full. Resumable. best-classical = highest holdout success of the 4 classical."""
import sys, os, time, json, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c10a import *
if __name__ == '__main__':
    S = list(range(1100, 1600)); guard(S); cfg = json.load(open(os.path.join(OUT, 'adapt2_tuning.json')))['chosen']; f = os.path.join(OUT, 'holdout.json'); res = json.load(open(f)) if os.path.exists(f) else {}
    specs = {'nom': 'nom', 'robust': 'robust', 'oracle_in': 'oracle_in', 'oracle_full': 'oracle_full', 'adapt': 'adapt', 'adapt2': ('adapt2', cfg)}; pool = mp.Pool(8); t0 = time.time()
    for k, sp in specs.items():
        if k in res: continue
        r = pool.map(run10, [(sp, s, 'abc') for s in S], chunksize=1)
        res[k] = dict(success=[x['success'] for x in r], min_dist=[x['min_dist'] for x in r], chain=[x['chain'] for x in r]); su = np.array(res[k]['success']); print(k, su.mean(), A.boot_ci(su), f'{time.time()-t0:.0f}s', flush=True); json.dump(res, open(f, 'w'))
    cl = ['nom', 'robust', 'adapt', 'adapt2']; sel = max(cl, key=lambda n: (round(float(np.mean(res[n]['success'])), 6), -float(np.nanmean(res[n]['min_dist']))))
    g = lambda n: np.array(res[n]['success']); out = dict(best_classical=sel, n=len(S), adapt2_cfg=cfg, success={n: float(g(n).mean()) for n in specs}, ci={n: A.boot_ci(g(n)) for n in specs})
    for o in ('oracle_full', 'oracle_in'):
        d = g(o) - g(sel); out[f'{o}-best'] = dict(diff_pp=100 * float(d.mean()), ci_pp=[100 * x for x in A.boot_ci(d, seed=1)])
    ls = np.array([c[0] for c in res['nom']['chain']]); q = np.quantile(ls, [0.25, 0.5, 0.75]); bins = np.digitize(ls, q); cells = {}
    for b in range(4):
        m = bins == b; cells[f'ls_q{b+1}'] = dict(n=int(m.sum()), ls_range=[float(ls[m].min()), float(ls[m].max())], **{n: float(g(n)[m].mean()) for n in specs}, best_classical=float(g(sel)[m].mean()))
    out['by_chain_length_quartile'] = cells; json.dump(out, open(os.path.join(OUT, 'holdout_summary.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
