"""Step 2: CMA-ES retune of T-robust / T-adapt on the NEW distribution, CAL seeds 1000-1049 only. Same search spaces/CMA as spike05 (reuses spike05/cma.py, calib_classical spaces).
usage: calib6.py robust|adapt [--cap 300] [--pop 10] [--gens 14] [--workers 1] [--warm results/spike05/robust.json]
Fitness = mean min(hits,cap)/cap - 0.1*mean min(apex_err,0.5). cap=300 (the real success criterion; no saturation in the new distribution).
Warm start = spike05 tuned params for the same method (stated deviation-free: both methods get the same treatment). Logs evals / sim steps / wall."""
import sys, os, json, time, argparse, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spike05'))
import env6 as W; V = W.V; from cma import CMA
import calib_classical as C
DIST = W.use_file()
def fit_chunk(args):
    P, seeds, cap = args; assert seeds.min() >= 1000; r = V.evaluate(P, seeds, cap=cap); ap = np.where(np.isnan(r['apex']), 0.5, np.minimum(r['apex'], 0.5))
    return float((np.minimum(r['hits'], cap) / cap).sum()), float(ap.sum()), r['steps'], float((r['hits'] >= cap).sum())
def fitness(pool, P, cap, nw):
    ch = np.array_split(C.SEEDS, nw); out = pool.map(fit_chunk, [(P, c, cap) for c in ch]); n = len(C.SEEDS)
    return sum(o[0] for o in out) / n - 0.1 * sum(o[1] for o in out) / n, sum(o[2] for o in out), sum(o[3] for o in out) / n
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--cap', type=int, default=300); ap.add_argument('--pop', type=int, default=10)
    ap.add_argument('--gens', type=int, default=14); ap.add_argument('--workers', type=int, default=1); ap.add_argument('--warm', default=None); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); adapt = a.mode == 'adapt'; space = C.ADA if adapt else C.ROB
    P0 = dict(V.P_NOM); P0['adapt'] = int(adapt)
    if a.warm: P0.update({k: v for k, v in json.load(open(a.warm))['params'].items() if k in space})
    x0 = np.clip(C.to_u(P0, space), 0.02, 0.98); es = CMA(x0, 0.2, a.pop, seed=a.seed); pool = mp.Pool(a.workers)
    t0 = time.time(); nev = 0; nsteps = 0; hist = []
    f0, s0, c0 = fitness(pool, C.to_P(x0, space, adapt), a.cap, a.workers); nev += 1; nsteps += s0; best = (f0, x0.copy(), C.to_P(x0, space, adapt), c0)
    print(f'start fitness {f0:.4f} frac_cap {c0:.2f} ({time.time()-t0:.0f}s)', flush=True)
    for g in range(a.gens):
        X = es.ask(); F = []
        for x in X:
            f, s, fc = fitness(pool, C.to_P(x, space, adapt), a.cap, a.workers); F.append(f); nev += 1; nsteps += s
            if f > best[0]: best = (f, x.copy(), C.to_P(x, space, adapt), fc)
        es.tell(X, F); hist.append(dict(gen=g, best=float(max(F)), mean=float(np.mean(F)), overall_best=best[0], best_frac_cap=best[3], sigma=float(es.sigma), evals=nev, steps=nsteps, wall=time.time() - t0)); print(hist[-1], flush=True)
        json.dump(dict(mode=a.mode, fitness=best[0], cal_frac_cap=best[3], params=best[2], cap=a.cap, pop=a.pop, gens_done=g + 1, evals=nev, sim_steps=nsteps, wall_s=time.time() - t0, free_params=len(space),
                       hist=hist, warm=a.warm, dist=DIST), open(f'results/spike06/{a.mode}.json', 'w'), indent=1)
