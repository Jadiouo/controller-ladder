"""Classical estimators for spike14a. fit4 = spike10b fit3 (T-adapt3-full) plus: (i) MAP prior from the KNOWN parameter distribution (moment matched Gaussians on the generative distribution),
(ii) multi-start with random draws from that prior, (iii) K-step-ahead (multiple-shooting style) residual: the ARM state is re-initialised to the measured (q, qd) every K control steps, chain state continues from the model.
cfg0 = frozen T-adapt3-full config (tune_full.json chosen). pw = 0 and K = 0 and nrand = 0 reproduces fit3 exactly (checked)."""
import sys, os, multiprocessing as mp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from c14 import *
from scipy.optimize import least_squares
PSD = dict(ls=0.1155, gm=0.5774, dev=0.1732, ld=1.1547, fr=0.5774)       # sd of the generative distribution (uniform ls in [.8,1.2]; log global mass U+-1; per-link U+-0.3; log damping U+-2; fric U[1,3] mm)
def replay_ms(ctx, th, q0, taus, Qm, Vm, K):
    m = model_theta(ctx, th); d = mujoco.MjData(m); n = ctx.robot.n; d.qpos[:n] = q0; d.qpos[n] = np.pi / 2 - q0.sum(); mujoco.mj_forward(m, d)
    sub = int(round(ctx.dt / m.opt.timestep)); Q = np.empty((len(taus), n)); V = np.empty((len(taus), n))
    for j, tau in enumerate(taus):
        d.ctrl[:n] = tau
        for _ in range(sub): mujoco.mj_step(m, d)
        Q[j] = d.qpos[:n]; V[j] = d.qvel[:n]
        if K > 0 and (j + 1) % K == 0: d.qpos[:n] = Qm[j]; d.qvel[:n] = Vm[j]; mujoco.mj_forward(m, d)
    return Q, V
def fit4(ctx, q0, taus, Qm, Vm, cfg, init=None):
    pm = 'full'; lo, hi, xs = pm_spec(pm); sq, sv, loss = cfg['sq'], cfg['sv'], cfg['loss']; preg = cfg.get('preg', 0.0); pw = cfg.get('pw', 0.0); K = int(cfg.get('K', 0)); n = len(taus); nr = int(cfg.get('nrand', 0))
    def res_T(x, T):
        try: Q, V = replay_ms(ctx, unpack(pm, x), q0, taus[:T], Qm[:T], Vm[:T], K)
        except Exception: return np.full(T * 6 + NL + 1 + (5 + NL if pw > 0 else 0), 1e3)
        r = [((Q - Qm[:T]) / sq).ravel(), ((V - Vm[:T]) / sv).ravel(), preg * (x[1:11] - x[1:11].mean()) / 0.3]
        r.append(np.array([(x[FRI[pm]] - 2.0) / cfg['fr_sd']]) if cfg.get('fr_sd') else np.zeros(1))
        if pw > 0:
            lmm = x[1:11].mean(); r.append(pw * np.concatenate([(x[1:11] - lmm) / PSD['dev'], [lmm / PSD['gm'], x[11] / PSD['ld'], (x[0] - 1) / PSD['ls'], (x[12] - 2.0) / PSD['fr']]]))
        return np.nan_to_num(np.concatenate(r), nan=1e3, posinf=1e3, neginf=-1e3).clip(-1e3, 1e3)
    Ts = [max(5, int(round(f * n))) for f in cfg['prefixes']]; ng = cfg['ngrid']
    g = [np.linspace(BOX['ls'][0], BOX['ls'][1], ng), np.linspace(-1.2, 1.2, ng), np.linspace(-2.0, 2.0, ng)]
    pts = [np.array([a] + [b] * NL + [c, 2.0]) for a in g[0] for b in g[1] for c in g[2]]
    if nr:
        rng = np.random.default_rng(int(hashlib.md5(np.round(Qm[:5], 9).tobytes()).hexdigest()[:8], 16))
        for _ in range(nr): pts.append(np.concatenate([[rng.uniform(.8, 1.2)], rng.uniform(-1, 1) + rng.uniform(-.3, .3, NL), [rng.uniform(-2, 2), rng.uniform(1, 3)]]))
    if init is not None:      # hybrid: single start at the (learned) estimate, no grid
        pts = [np.concatenate([[init[0]], np.log(np.maximum(init[1:11], 1e-6)), [np.log(max(init[11], 1e-6)), init[12] * 1e3]])]; cfg = {**cfg, 'nstart': 1}
    cs = np.array([0.5 * (res_T(p, Ts[0]) ** 2).sum() for p in pts]); best = None; nfev = len(pts)
    for j in np.argsort(cs)[:cfg['nstart']]:
        x = pts[j]
        for T in Ts:
            r = least_squares(res_T, np.clip(x, lo + 1e-9, hi - 1e-9), args=(T,), bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=cfg.get('nfev', 30), loss=loss, f_scale=cfg.get('f_scale', 1.0)); x = r.x; nfev += r.nfev
        if best is None or r.cost < best.cost: best = r
    fit4.nfev = nfev; return unpack(pm, best.x)
def cfg0(): return jload('tune_full.json')['chosen']
CFGS = {}
def reg(name, **kw): CFGS[name] = {**cfg0(), **kw}
reg('C0')                                         # = T-adapt3-full (frozen)
reg('C_map1', pw=1.0, fr_sd=0.0, preg=0.0); reg('C_map03', pw=0.3, fr_sd=0.0, preg=0.0); reg('C_map3', pw=3.0, fr_sd=0.0, preg=0.0)
reg('C_map1_ms8', pw=1.0, fr_sd=0.0, preg=0.0, nrand=8, nstart=8); reg('C_map1_ms', pw=1.0, fr_sd=0.0, preg=0.0, nrand=8, nstart=6); reg('C_map1_ms_K10', pw=1.0, fr_sd=0.0, preg=0.0, nrand=8, nstart=6, K=10)
reg('C_hyb', pw=1.0, fr_sd=0.0, preg=0.0, nstart=1); reg('C_ms10', K=10); reg('C_ms30', K=30)
reg('C_map1_K10', pw=1.0, fr_sd=0.0, preg=0.0, K=10); reg('C_map1_K30', pw=1.0, fr_sd=0.0, preg=0.0, K=30)
def _est(a):
    name, items = a; cfg = CFGS[name]; ctx = nominal_ctx(); out = []; t0 = time.time(); nf = []
    for it in items: out.append(fit4(ctx, *it[:4], cfg, init=(it[4] if len(it) > 4 else None))); nf.append(fit4.nfev)
    return np.array(out), time.time() - t0, nf
def estimate_set(pool, name, D, chunk=4, inits=None):
    its = [(D['q0'][i].astype(float), D['taus'][i].astype(float), D['Qm'][i].astype(float), D['Vm'][i].astype(float)) + ((inits[i],) if inits is not None else ()) for i in range(len(D['key']))]
    r = pool.map(_est, [(name, its[i:i + chunk]) for i in range(0, len(its), chunk)], chunksize=1)
    return np.concatenate([a for a, b, c in r]), sum(b for a, b, c in r), sum(sum(c) for a, b, c in r)
