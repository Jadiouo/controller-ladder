"""Experiment A: headroom inside error cells, no RL. Per cell: CMA-ES (spike05/cma.py, ROB space of calib_classical, same as spike06) warm-started from spike06 T-robust,
tuned ONLY on the cell's CAL seeds in 1000-1049; budget = 1 + 4*10 = 41 evals (spike06 robust: 81). Fitness as calib6: mean min(hits,300)/300 - 0.1*mean min(apex_err,0.5).
Holdout: (primary) cell seeds in 1050-1099; (extended, deviation, for power) cell seeds in 1100-1599. Paired bootstrap CI on per-episode success difference (cell-best minus T-robust).
Rule (pre-registered): any cell with +>=10 pp and CI excluding 0 -> headroom exists."""
import sys, os, json, time, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import *
from cma import CMA; import calib_classical as C
CELLS = {'eL_dL': lambda p: p['e'] < .585 and p['delay_ms'] < 200, 'eL_dH': lambda p: p['e'] < .585 and p['delay_ms'] >= 200,
         'eH_dL': lambda p: p['e'] >= .585 and p['delay_ms'] < 200, 'eH_dH': lambda p: p['e'] >= .585 and p['delay_ms'] >= 200, 'mH_dH': lambda p: p['mb'] >= 8.5 and p['delay_ms'] >= 200}
def cell_seeds(f, lo, hi): return np.array([s for s in range(lo, hi) if f(V.draw_params(s))])
def _fit(a):
    P, seeds, cap = a; guard(seeds); r = V.evaluate(P, np.asarray(seeds), cap=cap); ap = np.where(np.isnan(r['apex']), .5, np.minimum(r['apex'], .5))
    return float((np.minimum(r['hits'], cap) / cap).sum()), float(ap.sum()), r['steps'], float((r['hits'] >= cap).sum())
def fitness(pool, P, seeds, nw, cap=300):
    ch = [c for c in np.array_split(seeds, min(nw, len(seeds))) if len(c)]; o = pool.map(_fit, [(P, c, cap) for c in ch]); n = len(seeds)
    return sum(x[0] for x in o) / n - .1 * sum(x[1] for x in o) / n, sum(x[2] for x in o), sum(x[3] for x in o) / n
if __name__ == '__main__':
    nw = 8; pool = mp.Pool(nw); GENS = 4; POP = 10; space = C.ROB; t00 = time.time(); fn = os.path.join(OUT, 'expA.json'); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for name, f in CELLS.items():
        if name in out: continue
        T = cell_seeds(f, 1000, 1050); H1 = cell_seeds(f, 1050, 1100); H2 = cell_seeds(f, 1100, 1600); assert len(set(T) & set(H1)) == 0 and len(set(T) & set(H2)) == 0
        P0 = dict(V.P_NOM); P0['adapt'] = 0; P0.update({k: v for k, v in PROB.items() if k in space}); x0 = np.clip(C.to_u(P0, space), .02, .98); es = CMA(x0, .2, POP, seed=0)
        t0 = time.time(); nev = 1; steps = 0; f0, s0, c0 = fitness(pool, C.to_P(x0, space, False), T, nw); steps += s0; best = (f0, C.to_P(x0, space, False), c0); hist = [f0]
        for g in range(GENS):
            X = es.ask(); F = []
            for x in X:
                fv, s, fc = fitness(pool, C.to_P(x, space, False), T, nw); F.append(fv); nev += 1; steps += s
                if fv > best[0]: best = (fv, C.to_P(x, space, False), fc)
            es.tell(X, F); hist.append(best[0])
        # T-robust used for the baseline = the spike06 params exactly (PROB), not the re-clipped warm start
        res = {}
        for tag, S in (('tune', T), ('hold1', H1), ('hold2', H2)):
            hb, s1 = peval(pool, PROB, S, nw); hc, s2 = peval(pool, best[1], S, nw); steps += s1 + s2
            sb = (hb >= 300).astype(float); sc = (hc >= 300).astype(float); d = sc - sb
            res[tag] = dict(n=int(len(S)), seeds=[int(x) for x in S], robust_succ=float(sb.mean()), cell_succ=float(sc.mean()), diff=float(d.mean()), diff_ci=boot_ci(d, seed=3) if len(S) else None, robust_ci=boot_ci(sb, seed=4), cell_ci=boot_ci(sc, seed=5),
                            robust_hits=float(hb.mean()), cell_hits=float(hc.mean()), d_hits=float((hc - hb).mean()), d_hits_ci=boot_ci(hc - hb, seed=6))
        out[name] = dict(res=res, evals=nev, tune_fit_start=f0, tune_fit_best=best[0], hist=hist, params=best[1], sim_steps=steps, wall_s=time.time() - t0, kp=best[1]['kp']); json.dump(out, open(fn, 'w'), indent=1)
        print(name, {t: (r['n'], round(r['robust_succ'], 3), round(r['cell_succ'], 3), round(r['diff'], 3), [round(x, 3) for x in r['diff_ci']]) for t, r in res.items()}, f'{time.time()-t0:.0f}s', flush=True)
    print('total wall', time.time() - t00)
