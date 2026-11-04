"""CMA-ES tuning of the classical teacher on the CALIBRATION seeds 1000-1049 only (never the test seeds). Same random plant distribution as the test (H8).
usage: calib_classical.py robust|adapt [--cap 100] [--pop 12] [--gens 25] [--workers 4] [--warm results/spike05/robust.json]
Fitness (maximised) = mean_episodes min(hits,cap)/cap  - 0.1 * mean min(apex_err, 0.5)  (episodes with <3 hits count apex err 0.5)."""
import sys, os, json, time, argparse, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import env5 as V; from cma import CMA
ROB = {'kappa_max': (0.15, 1.2), 'kE': (0.0, 3.0), 'Kx': (0.3, 3.0), 'vxmax': (0.5, 2.5), 'kp': (300., 2500.), 'kd': (20., 150.), 'pred': (0., 30.), 'e_m': (0.3, 0.95),
       'a': (0.2, 0.95), 'b': (0.02, 0.5), 'phimax': (0.15, 0.6)}
ADA = {'kappa_max': (0.15, 1.2), 'kE': (0.0, 3.0), 'Kx': (0.3, 3.0), 'vxmax': (0.5, 2.5), 'kp': (300., 2500.), 'kd': (20., 150.), 'a': (0.2, 0.95), 'b': (0.02, 0.5), 'phimax': (0.15, 0.6),
       'e0': (0.4, 0.9), 'n0e': (1., 6.), 'bmin_e': (0.03, 0.6), 'd0': (0., 30.), 'cd': (0., 1.5), 'c0': (-5., 15.), 'thr': (0.02, 0.2)}
SEEDS = np.arange(1000, 1050)
def to_P(u, space, adapt):
    P = dict(V.P_NOM); P['adapt'] = int(adapt)
    for x, (k, (lo, hi)) in zip(u, space.items()): P[k] = float(lo + (hi - lo) * np.clip(x, 0, 1))
    return P
def to_u(P, space): return np.array([(P[k] - lo) / (hi - lo) for k, (lo, hi) in space.items()])
def fit_chunk(args):
    P, seeds, cap = args; assert seeds.min() >= 1000
    r = V.evaluate(P, seeds, cap=cap); ap = np.where(np.isnan(r['apex']), 0.5, np.minimum(r['apex'], 0.5))
    return float((np.minimum(r['hits'], cap) / cap).sum()), float(ap.sum()), r['steps']
def fitness(pool, P, cap, nw):
    ch = np.array_split(SEEDS, nw); out = pool.map(fit_chunk, [(P, c, cap) for c in ch])
    return sum(o[0] for o in out) / len(SEEDS) - 0.1 * sum(o[1] for o in out) / len(SEEDS), sum(o[2] for o in out)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--cap', type=int, default=100); ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--gens', type=int, default=25); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--warm', default=None); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); adapt = a.mode == 'adapt'; space = ADA if adapt else ROB
    P0 = dict(V.P_NOM); P0['adapt'] = int(adapt)
    if adapt: P0.update(e0=0.7, n0e=2.0, bmin_e=0.15, d0=10.0, cd=1.0, c0=0.0, thr=0.06)
    if a.warm:
        W = json.load(open(a.warm))['params']; P0.update({k: v for k, v in W.items() if k in space})
    x0 = np.clip(to_u(P0, space), 0.02, 0.98); es = CMA(x0, 0.2, a.pop, seed=a.seed); pool = mp.Pool(a.workers)
    t0 = time.time(); nev = 0; nsteps = 0; hist = []; best = (-9, None, None)
    f0, s0 = fitness(pool, to_P(x0, space, adapt), a.cap, a.workers); nev += 1; nsteps += s0; best = (f0, x0.copy(), to_P(x0, space, adapt))
    print(f'start fitness {f0:.4f} ({time.time()-t0:.0f}s)', flush=True)
    for g in range(a.gens):
        X = es.ask(); F = []
        for x in X:
            f, s = fitness(pool, to_P(x, space, adapt), a.cap, a.workers); F.append(f); nev += 1; nsteps += s
            if f > best[0]: best = (f, x.copy(), to_P(x, space, adapt))
        es.tell(X, F); hist.append(dict(gen=g, best=float(max(F)), mean=float(np.mean(F)), overall_best=best[0], sigma=float(es.sigma), evals=nev, steps=nsteps, wall=time.time() - t0))
        print(hist[-1], flush=True)
        json.dump(dict(mode=a.mode, fitness=best[0], params=best[2], cap=a.cap, pop=a.pop, gens_done=g + 1, evals=nev, sim_steps=nsteps, wall_s=time.time() - t0, free_params=len(space), hist=hist, warm=a.warm),
                  open(f'results/spike05/{a.mode}.json', 'w'), indent=1)
