"""Step 3 (like spike08_diag expA): headroom inside error cells. 4 cells = quartiles of the true chain-length error (len_rel) (the dominant error from the pilot table).
Per cell: CMA-ES (same 5-param space as T-robust) warm-started from T-robust, tuned ONLY on the cell's CAL seeds in 1000-1049 (pop 10, gens 5 = 51 evals).
Holdout: cell seeds in 1100-1599. Paired bootstrap CI on success difference (cell-best minus T-robust; also minus T-nom). Rule: any cell +>=10 pp with CI excluding 0 -> headroom."""
import time, multiprocessing as mp; from common import *
from cma import CMA; from step2_robust import to_P, to_u, KEYS
if __name__ == '__main__':
    W = json.load(open(os.path.join(OUT, 'dist.json')))['w']; PR = json.load(open(os.path.join(OUT, 'robust.json')))['params']
    edges = [-W / 2 * 1.0 * 1, -0.0, W / 2]; edges = [-W / 2, 0.0, W / 2]         # len_rel in U(-W,W): quartile edges -W/2, 0, W/2
    def cell_of(s): l = draw_true(s, W)['len_rel']; return int(np.searchsorted(edges, l))
    fn = os.path.join(OUT, 'headroom.json'); out = json.load(open(fn)) if os.path.exists(fn) else {}; pool = mp.Pool(8); t00 = time.time()
    for c in range(4):
        name = f'len_q{c + 1}'
        if name in out: continue
        T = np.array([s for s in range(1000, 1050) if cell_of(s) == c]); H = np.array([s for s in range(1100, 1600) if cell_of(s) == c]); guard(T); guard(H); assert not set(T) & set(H)
        t0 = time.time(); nev = 0
        def ev(P):
            global nev; s, md, _ = peval(pool, T, W, P); nev += 1; return fit_of(s, md), float(s.mean())
        x0 = to_u(PR); f0, s0 = ev(to_P(x0)); best = (f0, s0, x0.copy()); es = CMA(x0, 0.2, 10, seed=1)
        for g in range(5):
            X = es.ask(); F = []
            for x in X:
                f, sc = ev(to_P(x)); F.append(f)
                if f > best[0]: best = (f, sc, x.copy())
            es.tell(X, F)
        Pc = to_P(best[2]); res = {}
        for tag, S in (('tune', T), ('hold', H)):
            sb = peval(pool, S, W, PR)[0]; sc_ = peval(pool, S, W, Pc)[0]; sn = peval(pool, S, W, None)[0]; d = sc_ - sb; dn = sc_ - sn
            res[tag] = dict(n=int(len(S)), robust=float(sb.mean()), cell=float(sc_.mean()), nom=float(sn.mean()), diff=float(d.mean()), diff_ci=boot_ci(d, seed=3), diff_vs_nom=float(dn.mean()), diff_vs_nom_ci=boot_ci(dn, seed=4),
                            robust_ci=boot_ci(sb, seed=5), cell_ci=boot_ci(sc_, seed=6), nom_ci=boot_ci(sn, seed=7))
        out[name] = dict(res=res, evals=nev, tune_fit_start=f0, tune_fit_best=best[0], params=Pc, edge_hits=[k for k, x in zip(KEYS, best[2]) if x < 0.03 or x > 0.97], wall_s=time.time() - t0,
                         len_rel_range=[(-W, edges[0]), (edges[0], edges[1]), (edges[1], edges[2]), (edges[2], W)][c]); json.dump(out, open(fn, 'w'), indent=1)
        print(name, {t: (r['n'], round(r['robust'], 3), round(r['cell'], 3), round(r['diff'], 3), [round(x, 3) for x in r['diff_ci']]) for t, r in res.items()}, f'{time.time()-t0:.0f}s', flush=True)
    print('total wall', time.time() - t00)
