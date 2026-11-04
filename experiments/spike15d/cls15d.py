"""spike15d: classical uncertainty sets from the R0 probe data only. usage: cls15d.py <cal|hold|new>  -> .scratch/s15d/cls_<set>.pkl
Per episode: 12 starts of the R0 (fit3, cfg tune_full) optimiser (4 = R0, grid 5-8, 4 prior draws), centre = R0 estimate, MS members (3 lowest-cost distinct solutions),
LAP members (9 draws from the Gauss-Newton/Laplace posterior with the generative prior). Run with cwd = .scratch/s15d/run under flock + systemd-run."""
import sys, os, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c15d import *
from scipy.optimize import least_squares
CFG = jload('tune_full.json')['chosen']
PSD_X = np.array([0.1155] + [0.60] * NL + [1.1547, 0.5774])
EPS = 0.1
def ens_fit(ctx, q0, taus, Qm, Vm, key):
    cfg = CFG; pm = 'full'; lo, hi, xs = pm_spec(pm); sq, sv = cfg['sq'], cfg['sv']; preg = cfg['preg']; n = len(taus)
    def res_T(x, T):
        try: Q, V, _ = replay3(ctx, unpack(pm, x), q0, taus[:T])
        except Exception: return np.full(T * 6 + NL + 1, 1e3)
        r = [((Q - Qm[:T]) / sq).ravel(), ((V - Vm[:T]) / sv).ravel(), preg * (x[1:11] - x[1:11].mean()) / 0.3, np.array([(x[12] - 2.0) / cfg['fr_sd']])]
        return np.nan_to_num(np.concatenate(r), nan=1e3, posinf=1e3, neginf=-1e3).clip(-1e3, 1e3)
    Ts = [max(5, int(round(f * n))) for f in cfg['prefixes']]; ng = cfg['ngrid']
    g = [np.linspace(BOX['ls'][0], BOX['ls'][1], ng), np.linspace(-1.2, 1.2, ng), np.linspace(-2.0, 2.0, ng)]
    pts = [np.array([a] + [b] * NL + [c, 2.0]) for a in g[0] for b in g[1] for c in g[2]]
    cs = np.array([0.5 * (res_T(p, Ts[0]) ** 2).sum() for p in pts]); order = np.argsort(cs)
    rng = np.random.default_rng(int(key[:8], 16)); starts = [pts[j] for j in order[:8]]
    for _ in range(4): starts.append(np.concatenate([[rng.uniform(.8, 1.2)], rng.uniform(-1, 1) + rng.uniform(-.3, .3, NL), [rng.uniform(-2, 2), rng.uniform(1, 3)]]))
    X, C, J = [], [], []
    for x in starts:
        for T in Ts:
            r = least_squares(res_T, np.clip(x, lo + 1e-9, hi - 1e-9), args=(T,), bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=cfg.get('nfev', 30), loss=cfg['loss'], f_scale=cfg.get('f_scale', 1.0)); x = r.x
        X.append(r.x); C.append(r.cost); J.append(r.jac)
    X = np.array(X); C = np.array(C); ib = int(np.argmin(C[:4])); xc = X[ib]
    # MS members
    dist = lambda a, b: np.max(np.abs((a - b) / xs)); chosen = []
    for j in np.argsort(C):
        if j == ib: continue
        if dist(X[j], xc) < EPS or any(dist(X[j], X[k]) < EPS for k in chosen): continue
        chosen.append(int(j))
        if len(chosen) == 3: break
    nuniq = len(chosen)
    while len(chosen) < 3: chosen.append(chosen[-1] if chosen else ib)
    ms = np.array([unpack(pm, X[j]) for j in chosen]); ms_cost = C[chosen] / C[ib]
    # Laplace members
    Jb = J[ib]; H = Jb.T @ Jb + np.diag(1.0 / PSD_X ** 2); Sig = np.linalg.inv(H); Sig = 0.5 * (Sig + Sig.T); L = np.linalg.cholesky(Sig + 1e-12 * np.eye(13))
    r2 = np.random.default_rng(int(key[8:16], 16)); Z = r2.standard_normal((9, 13)); XL = np.clip(xc + Z @ L.T, lo + 1e-9, hi - 1e-9); lap = np.array([unpack(pm, x) for x in XL])
    return dict(center=unpack(pm, xc), ms=ms, ms_cost=ms_cost, nuniq=nuniq, lap=lap, lap_sd=np.sqrt(np.diag(Sig)), cost_all=C, ib=ib, n_near=int((C <= 1.25 * C[ib]).sum()), n_dist_all=int(sum(dist(X[j], xc) >= EPS for j in range(12))))
_CTXD = [None]
def work(a):
    D, idx = a; ctx = nominal_ctx(); out = []; t0 = time.time()
    for i in idx: guard15d(D['seed'][i]); out.append(ens_fit(ctx, D['q0'][i].astype(float), D['taus'][i].astype(float), D['Qm'][i].astype(float), D['Vm'][i].astype(float), str(D['key'][i])))
    return out, time.time() - t0
if __name__ == '__main__':
    SET = sys.argv[1]; NW = int(os.environ.get('NW', 8)); LIM = int(os.environ.get('LIM', 10 ** 9))
    D = pickle.load(open(os.path.join(S15D if SET == 'new' else S15C, f'raw_{SET}.pkl'), 'rb')); N = min(len(D['key']), LIM); assert list(D['seed'][:N]) == SETS[SET][:N]
    t0 = time.time(); chunks = [list(range(i, min(i + 4, N))) for i in range(0, N, 4)]
    with mp.Pool(NW) as pool: r = pool.map(work, [(D, c) for c in chunks], chunksize=1)
    res = sum([a for a, b in r], []); cpu = sum(b for a, b in r); wall = time.time() - t0
    pickle.dump(dict(seeds=SETS[SET][:N], key=D['key'][:N], TH=D['TH'][:N], res=res, cpu=cpu, wall=wall), open(os.path.join(S15D, f'cls_{SET}.pkl' if N == len(D['key']) else f'cls_{SET}_lim{N}.pkl'), 'wb'))
    print('cls', SET, N, f'wall {wall:.0f}s cpu {cpu:.0f}s', flush=True)
