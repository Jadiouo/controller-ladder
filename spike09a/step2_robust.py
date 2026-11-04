"""Step 2: T-robust = CMA-ES (spike05/cma.py, maximise, unit cube) over 5 free params (log-scale, range [default/4, default*4] each = 16x wide):
planner-assumed chain length/mass/damping scales (ls, ms, ds) and tracking gains (qp, qv). Fitness on CAL 1000-1049 over the whole dist (w from dist.json)
= success - 0.2*mean(min(min_dist,0.5)). Starts at defaults (= T-nom). Logs evals, episodes, wall, boundary hits."""
import time, multiprocessing as mp; from common import *
from cma import CMA
KEYS = ['ls', 'ms', 'ds', 'qp', 'qv']
def to_P(u): return {k: float(DEF[k] * 4.0 ** (2 * np.clip(x, 0, 1) - 1)) for k, x in zip(KEYS, u)}
def to_u(P): return np.array([(np.log(P[k] / DEF[k]) / np.log(4) + 1) / 2 for k in KEYS])
if __name__ == '__main__':
    W = json.load(open(os.path.join(OUT, 'dist.json')))['w']; S = np.arange(1000, 1050); guard(S); POP, GENS = 12, 15
    pool = mp.Pool(8); t0 = time.time(); nev = 0; neps = 0; hist = []
    def ev(P):
        global nev, neps
        s, md, _ = peval(pool, S, W, P); nev += 1; neps += len(S); return fit_of(s, md), float(s.mean())
    x0 = to_u(DEF); f0, s0 = ev(to_P(x0)); best = (f0, s0, x0.copy()); print('T-nom start', f0, s0, flush=True); es = CMA(x0, 0.3, POP, seed=0)
    for g in range(GENS):
        X = es.ask(); F = []
        for x in X:
            f, sc = ev(to_P(x)); F.append(f)
            if f > best[0]: best = (f, sc, x.copy())
        es.tell(X, F); hist.append(dict(gen=g, best_gen=float(max(F)), overall=best[0], overall_succ=best[1], sigma=float(es.sigma), evals=nev, wall=time.time() - t0)); print(hist[-1], flush=True)
        json.dump(dict(params=to_P(best[2]), u=best[2].tolist(), fitness=best[0], cal_succ=best[1], evals=nev, episodes=neps, wall_s=time.time() - t0, free_params=5, search_range='each param default*4^[-1,1] (log-uniform in unit cube)',
                       edge_hits=[k for k, x in zip(KEYS, best[2]) if x < 0.03 or x > 0.97], hist=hist, dist_w=W, T_nom_cal_succ=s0, pop=POP, gens=GENS), open(os.path.join(OUT, 'robust.json'), 'w'), indent=1)
