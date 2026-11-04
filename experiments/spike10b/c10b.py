"""spike10b shared code (whip, realistic conditions). Builds on experiments/spike09b/c9b.py and experiments/spike10a/c10a.py (imported, not modified).
Adds: (1) condition scaling (dose) and training reps, (2) integration-step fix for numerical blow-ups of the TRUE plant + instability counter,
(3) generalised plant model theta = [ls, m_1..m_10 (scales of 0.015 kg), ds, fric (Nm)] with three parametrisations (full / grad / eff),
(4) T-adapt3 (model-fit estimator on that class), L-est estimators, planner starting from an estimated state (Tuned3S), controllers Probe3.
Run with cwd = .scratch/run (MuJoCo writes MUJOCO_LOG.TXT in cwd)."""
import sys, os, json, hashlib, dataclasses, numpy as np, mujoco
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike10a')); sys.path.insert(0, os.path.join(_H, '..', 'spike09b'))
import c9b, c10a
from c9b import *
from c10a import SIG_Q, SIG_QD, MASS_SPREAD, FRIC0, ACT_LO, ACT_HI, OracleFull
from arena.perturb.base import Perturbation
import arena.env as AE
OUT = os.path.join(ROOT, 'results/spike10b'); PL = os.path.join(ROOT, '.scratch/s10b_plans'); os.makedirs(PL, exist_ok=True); os.makedirs(OUT, exist_ok=True)
SUBF = int(os.environ.get('SUBF', 4))          # plant integration step divided by SUBF (control rate unchanged); see step0 / README
# ---------------------------------------------------------------- instability counter (true plant only: arena/env.py)
INST = [0]
class _Shim:
    def __getattr__(self, k): return getattr(mujoco, k)
    def mj_step(self, m, d, *a, **k):
        t0 = d.time; mujoco.mj_step(m, d, *a, **k)
        if d.time < t0 or not np.isfinite(d.qpos).all(): INST[0] += 1
AE.mujoco = _Shim()
# ---------------------------------------------------------------- conditions: name -> (a, b, c, scale)
CONDS = {'none': (0, 0, 0, 1.0), 'a': (1, 0, 0, 1.0), 'b': (0, 1, 0, 1.0), 'c': (0, 0, 1, 1.0), 'abc': (1, 1, 1, 1.0), 'x0.5': (1, 1, 1, 0.5), 'x2': (1, 1, 1, 2.0)}
def guard(seeds, allow=()):
    s = np.asarray(seeds); ok = (s >= 1000) & (s < 10000) | (s >= 500000)
    for lo, hi in allow: ok |= (s >= lo) & (s <= hi)
    assert ok.all(), 'seed range not allowed here'
def draw_real(seed, rep, s):
    r = np.random.default_rng([int(seed), 1010] if rep is None else [int(seed), int(rep), 1010]); u = r.uniform(-1, 1, 10); fr = r.uniform(0.5, 1.5); sg = r.uniform(0, 1)
    return dict(mass_f=np.exp(MASS_SPREAD * s * u), fric=float(FRIC0 * s * fr), act_sigma=float(s * (ACT_LO + (ACT_HI - ACT_LO) * sg)))
class Real2(Perturbation):
    def __init__(self, seed, rep, cond):
        self.seed, self.rep, self.cond = int(seed), rep, cond; self.a, self.b, self.c, self.s = CONDS[cond]; self.d = draw_real(seed, rep, self.s)
    def modify_task(self, task):
        if self.b: task.chain_mass = np.asarray(task.chain_mass, float) * self.d['mass_f']
        return task
    def reset(self, seed, env):
        base = [self.seed] if self.rep is None else [self.seed, int(self.rep)]
        self.rng = np.random.default_rng(base + [1011]); self.rng2 = np.random.default_rng(base + [1012]); n = env.n
        if self.b: env.model.dof_frictionloss[n:] = self.d['fric']
        if SUBF > 1 and not getattr(env, '_sub_patched', False):
            env.model.opt.timestep = env.model.opt.timestep / SUBF; env.robot = dataclasses.replace(env.robot, dt_sim=env.robot.dt_sim / SUBF, substeps=env.robot.substeps * SUBF); env._sub_patched = True
    def on_obs(self, obs, env):
        if not self.a: return obs
        return obs.replace(q=obs.q + SIG_Q * self.s * self.rng.standard_normal(len(obs.q)), qd=obs.qd + SIG_QD * self.s * self.rng.standard_normal(len(obs.qd)))
    def on_command(self, tau, k): return tau + self.d['act_sigma'] * self.rng2.standard_normal(len(tau)) if self.c else tau
    def label(self): return f'real2_{self.cond}'
def scen(seed, cond, rep=None):
    t = true_chain(seed, rep); return scenario_of(t).with_perturb(Real2(seed, rep, cond))
def true_theta(seed, cond, rep=None):
    t = true_chain(seed, rep); a, b, c, s = CONDS[cond]; d = draw_real(seed, rep, s)
    ms = (1 + t['mass_rel']) * (d['mass_f'] if b else np.ones(10))
    return np.concatenate([[1 + t['len_rel']], ms, [t['damp_fac']], [d['fric'] if b else 0.0]])
# ---------------------------------------------------------------- generalised plant model
NL = 10; IDX = np.arange(NL); NOMV = dict(len=0.1, mass=0.015, damp=0.0004)
def theta_arrays(ctx, th):
    info = ctx.task_info
    return dict(chain_len=np.asarray(info['chain_len'], float) * th[0], chain_mass=np.asarray(info['chain_mass'], float) * np.asarray(th[1:11]), chain_damp=np.asarray(info['chain_damp'], float) * th[11])
def model_theta(ctx, th):
    arr = theta_arrays(ctx, th); t = make_task('whip', ctx.robot, n_links=int(ctx.task_info['n_links']))
    t.chain_len, t.chain_mass, t.chain_damp = arr['chain_len'], arr['chain_mass'], arr['chain_damp']
    m = mujoco.MjModel.from_xml_string(build_mjcf(ctx.robot, t.tip_xml(ctx.robot)))
    if th[12] > 0: m.dof_frictionloss[ctx.robot.n:] = th[12]
    return m
def replay3(ctx, th, q0, taus):
    m = model_theta(ctx, th); d = mujoco.MjData(m); n = ctx.robot.n; d.qpos[:n] = q0; d.qpos[n] = np.pi / 2 - q0.sum(); mujoco.mj_forward(m, d)
    sub = int(round(ctx.dt / m.opt.timestep)); Q = np.empty((len(taus), n)); V = np.empty((len(taus), n))
    for j, tau in enumerate(taus):
        d.ctrl[:n] = tau
        for _ in range(sub): mujoco.mj_step(m, d)
        Q[j] = d.qpos[:n]; V[j] = d.qvel[:n]
    return Q, V, (d.qpos.copy(), d.qvel.copy())
LM = 1.5     # box on log mass scale per link (global exp(U+-1) x per-link exp(U+-0.3) x2-dose margin)
BOX = dict(ls=(1 - W, 1 + W), lm=(-LM, LM), ld=(-2.2, 2.2), fr=(0.0, 8.0), g=(-0.4, 0.4))     # fr in units of 1e-3 Nm
def pm_spec(pm):
    if pm == 'full': lo = [BOX['ls'][0]] + [BOX['lm'][0]] * NL + [BOX['ld'][0], BOX['fr'][0]]; hi = [BOX['ls'][1]] + [BOX['lm'][1]] * NL + [BOX['ld'][1], BOX['fr'][1]]; xs = [0.1] + [0.3] * NL + [1.0, 1.0]
    elif pm == 'grad': lo = [BOX['ls'][0], BOX['lm'][0], BOX['g'][0], BOX['ld'][0], BOX['fr'][0]]; hi = [BOX['ls'][1], BOX['lm'][1], BOX['g'][1], BOX['ld'][1], BOX['fr'][1]]; xs = [0.1, 0.5, 0.2, 1.0, 1.0]
    elif pm == 'eff': lo = [BOX['ls'][0], BOX['lm'][0], BOX['ld'][0], BOX['fr'][0]]; hi = [BOX['ls'][1], BOX['lm'][1], BOX['ld'][1], BOX['fr'][1]]; xs = [0.1, 0.5, 1.0, 1.0]
    return np.array(lo), np.array(hi), np.array(xs)
def unpack(pm, x):
    if pm == 'full': return np.concatenate([[x[0]], np.exp(x[1:11]), [np.exp(x[11]), x[12] * 1e-3]])
    if pm == 'grad': return np.concatenate([[x[0]], np.exp(x[1] + x[2] * (IDX - 4.5) / 4.5), [np.exp(x[3]), x[4] * 1e-3]])
    return np.concatenate([[x[0]], np.full(NL, np.exp(x[1])), [np.exp(x[2]), x[3] * 1e-3]])
def clip_theta(th): return np.concatenate([[np.clip(th[0], *BOX['ls'])], np.exp(np.clip(np.log(np.maximum(th[1:11], 1e-6)), *BOX['lm'])), [np.exp(np.clip(np.log(max(th[11], 1e-6)), *BOX['ld'])), np.clip(th[12], 0, BOX['fr'][1] * 1e-3)]])
FRI = dict(full=12, grad=4, eff=3)
def fit3(ctx, q0, taus, Qm, Vm, cfg):
    """T-adapt3 estimator: grid over (ls, uniform log m, log ds) at fric = 2e-3, then bounded robust nonlinear LSQ in the chosen parametrisation (progressive prefixes of the probe)."""
    from scipy.optimize import least_squares
    pm = cfg['pm']; lo, hi, xs = pm_spec(pm); sq, sv, loss = cfg['sq'], cfg['sv'], cfg['loss']; preg = cfg.get('preg', 0.0); n = len(taus)
    def res_T(x, T):
        try: Q, V, _ = replay3(ctx, unpack(pm, x), q0, taus[:T])
        except Exception: return np.full(T * 6 + (NL if pm == 'full' else 0) + (1 if (pm == 'grad' and cfg.get('g_sd')) else 0) + (1 if cfg.get('fr_sd') else 0), 1e3)
        r = [((Q - Qm[:T]) / sq).ravel(), ((V - Vm[:T]) / sv).ravel()]
        if pm == 'full': r.append(preg * (x[1:11] - x[1:11].mean()) / 0.3)
        if pm == 'grad' and cfg.get('g_sd'): r.append(np.array([x[2] / cfg['g_sd']]))
        if cfg.get('fr_sd'): r.append(np.array([(x[FRI[pm]] - 2.0) / cfg['fr_sd']]))      # weak Gaussian prior on friction (centre 2e-3 Nm = dist. mean: classical prior knowledge, disclosed)
        return np.nan_to_num(np.concatenate(r), nan=1e3, posinf=1e3, neginf=-1e3).clip(-1e3, 1e3)
    def start(ls, lm, ld, fr):
        if pm == 'full': return np.array([ls] + [lm] * NL + [ld, fr])
        if pm == 'grad': return np.array([ls, lm, 0.0, ld, fr])
        return np.array([ls, lm, ld, fr])
    Ts = [max(5, int(round(f * n))) for f in cfg['prefixes']]; ng = cfg['ngrid']
    g = [np.linspace(BOX['ls'][0], BOX['ls'][1], ng), np.linspace(-1.2, 1.2, ng), np.linspace(-2.0, 2.0, ng)]
    pts = [start(a, b, c, 2.0) for a in g[0] for b in g[1] for c in g[2]]
    cs = np.array([0.5 * (res_T(p, Ts[0]) ** 2).sum() for p in pts]); best = None; nfev = len(pts)
    for j in np.argsort(cs)[:cfg['nstart']]:
        x = pts[j]
        for T in Ts:
            r = least_squares(res_T, np.clip(x, lo + 1e-9, hi - 1e-9), args=(T,), bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=cfg.get('nfev', 30), loss=loss, f_scale=cfg.get('f_scale', 1.0)); x = r.x; nfev += r.nfev
        if best is None or r.cost < best.cost: best = r
    fit3.nfev = nfev; return unpack(pm, best.x)
# ---------------------------------------------------------------- planner from an estimated state + controllers
class Tuned3S(TunedS):
    """Planner on an estimated plant theta (per-link arrays + Coulomb friction), starting from the reconstructed chain state x0 after the probe."""
    def __init__(self, ctx, th, x0, duration, qp=DEF['qp'], qv=DEF['qv']):
        arr = theta_arrays(ctx, th); fr = float(th[12]); self.thk = hashlib.md5(np.round(np.asarray(th), 6).tobytes()).hexdigest()[:12]
        super().__init__(dataclasses.replace(ctx, task_info={**ctx.task_info, **arr}), dict(ls=1.0, ms=1.0, ds=1.0, qp=qp, qv=qv), x0=x0, duration=duration)
        base_mk = self.ctx.make_model; n = ctx.robot.n
        def mk():
            m = base_mk()
            if fr > 0: m.dof_frictionloss[n:] = fr
            return m
        self.ctx = dataclasses.replace(self.ctx, make_model=mk); self.sim = SwingSimS(self.ctx); self.sim.x0 = x0
    def plan(self, q0, target):
        k = hashlib.md5(repr((int(self.seed), self.thk, self.xh, round(self.dur, 4), self.P['qp'], self.P['qv'])).encode()).hexdigest(); f = os.path.join(PL, f'fin3_{k}.npy')
        if os.path.exists(f):
            try: return np.load(f)
            except Exception: pass
        r = self._plan_impl(q0, target); np.save(f + f'.{os.getpid()}.npy', r); os.replace(f + f'.{os.getpid()}.npy', f); return r
    def _plan_impl(self, q0, target):
        n = self.ctx.robot.n; k1 = hashlib.md5(repr((int(self.seed), self.thk, self.xh)).encode()).hexdigest(); f1 = os.path.join(PL, f's13_{k1}.npy'); picked = None
        if os.path.exists(f1):
            try: picked = np.load(f1)
            except Exception: picked = None
        if picked is None:
            picked = self._stage1(q0, target); np.save(f1 + f'.{os.getpid()}.npy', picked); os.replace(f1 + f'.{os.getpid()}.npy', f1)
        scores = [self._closed_loop(q0, x.reshape(4, n), target) for x in picked]; j = int(np.argmin(scores)); self.plan_info = dict(planned_min_dist=scores[j]); return picked[j].reshape(4, n)
class Probe3(c9b.Probe):
    """probe segment (explicit T, amp) -> est_fn -> theta13 -> reconstruct chain state by replay -> plan on the estimated plant -> execute."""
    def __init__(self, ctx, est_fn, tag, T, amp, qp=DEF['qp'], qv=DEF['qv']):
        super().__init__(ctx, est_fn, tag, qp, qv); self.T, self.amp = float(T), np.array(amp, float); self.n_pr = int(round(self.T / ctx.dt))
    def _swing_start(self, obs):
        taus, Qm, Vm = self.history(); self.pq0 = self.hq0; key = est_key(self.tag, self.pq0, taus, Qm)
        if key not in EST_CACHE:
            f = os.path.join(PL, f'est3_{key}.npy')
            if os.path.exists(f):
                try: EST_CACHE[key] = tuple(float(x) for x in np.load(f))
                except Exception: pass
            if key not in EST_CACHE:
                EST_CACHE[key] = tuple(float(x) for x in self.est_fn(self.ctx, self.pq0, taus, Qm, Vm)); np.save(f + f'.{os.getpid()}.npy', np.array(EST_CACHE[key])); os.replace(f + f'.{os.getpid()}.npy', f)
        th = np.array(EST_CACHE[key]); self.est = th; _, _, x0 = replay3(self.ctx, th, self.pq0, taus)
        inner = Tuned3S(self.ctx, th, x0, self.ctx.task_info['duration'] - self.T, self.qp, self.qv); inner.reset(self.seed)
        self.q0 = np.asarray(obs.q, float).copy(); self.knots = inner.plan(self.q0, np.asarray(obs.ref['target'], float)); self.plan_info = dict(inner.plan_info)
    def act(self, obs):
        k = obs.k
        if k == 0: self.hq0 = np.asarray(obs.q, float).copy()
        if k > 0 and k <= self.n_pr: self.Qm.append(np.asarray(obs.q, float).copy()); self.Vm.append(np.asarray(obs.qd, float).copy())
        if k < self.n_pr:
            qr, vr, ar = probe_ref(self.hq0, k, self.ctx.dt, T=self.T, amp=self.amp); self.qr, self.qdr = qr, vr; v = ar + self.Kp * (qr - obs.q) + self.Kd * (vr - obs.qd)
            tau = np.clip(self.arm.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit); self.taus.append(tau.copy()); return tau
        if self.knots is None: self._swing_start(obs)
        t = np.array([(k - self.n_pr) * self.ctx.dt]); qr, qdr, qddr = (a[0] for a in WT.spline_traj(self.q0, self.knots, t))
        self.qr, self.qdr = qr, qdr; v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd); return np.clip(self.arm.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
class OracleFull3(OracleFull):
    """T-oracle (full): planner gets the true per-link arrays + friction (no probe, no estimation)."""
    pass
def cfg_tag(cfg): return 'a3_' + hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
A3_BASE = dict(pm='grad', T=0.4, amp=0.6, loss='soft_l1', f_scale=1.0, sq=0.01, sv=0.1, preg=0.0, g_sd=0.0, fr_sd=0.0, prefixes=[0.5, 0.75, 1.0], ngrid=7, nstart=4, nfev=30)
def make_adapt3(cfg):
    est = lambda ctx, q0, taus, Qm, Vm: fit3(ctx, q0, taus, Qm, Vm, cfg)
    return lambda ctx: Probe3(ctx, est, cfg_tag(cfg), cfg['T'], np.array([0.5, 0.5, 0.4]) * cfg['amp'])
def make_adapt2_10b(cfg):
    """T-adapt2 as spike10a (c10a.make_adapt2), unchanged estimator, sets c9b probe globals; restored by caller"""
    return c10a.make_adapt2(cfg)
# ---------------------------------------------------------------- episode runner
def reset_probe_globals(): c9b.PROBE_T = 0.2; c9b.PROBE_AMP = np.array([0.5, 0.5, 0.4]) * 0.3
def run10(args):
    """args = (spec, seed, rep, cond[, allow]) -> metrics. spec: 'nom'|'robust'|'adapt'|'oracle'|('adapt2',cfg)|('adapt3',cfg)|('est',tag,estfn,T,amp)"""
    spec, seed, rep, cond = args[:4]; allow = args[4] if len(args) > 4 else (); guard([seed], allow); reset_probe_globals(); sc = scen(seed, cond, rep); INST[0] = 0
    if spec in ('nom', 'robust', 'adapt'): fac = make_factory(spec)
    elif spec == 'oracle':
        th = true_theta(seed, cond, rep); arr = theta_arrays(type('C', (), dict(task_info=dict(chain_len=np.full(NL, .1), chain_mass=np.full(NL, .015), chain_damp=np.full(NL, .0004))))(), th); fac = lambda ctx: OracleFull(ctx, arr, float(th[12]))
    elif spec[0] == 'adapt2': fac = make_adapt2_10b(spec[1])
    elif spec[0] == 'adapt3': fac = make_adapt3(spec[1])
    elif spec[0] == 'est': fac = lambda ctx: Probe3(ctx, spec[2], spec[1], spec[3], np.array([0.5, 0.5, 0.4]) * spec[4])
    else: raise ValueError(spec)
    tr = run_episode(sc, fac, int(seed)); m = episode_metrics(sc, tr); m['inst'] = int(INST[0]); reset_probe_globals(); m['theta'] = true_theta(seed, cond, rep).tolist(); return m
# ---------------------------------------------------------------- probe-only data (L-est training / identifiability)
def probe_run3(env, ctx, obs, T, amp):
    n_pr = int(round(T / ctx.dt)); arm = ArmModel(ctx.robot); Kp, Kd = lqr_double_integrator(DEF['qp'], DEF['qv']); q0 = obs.q.copy(); taus, Qm, Vm = [], [], []
    for k in range(n_pr):
        qr, vr, ar = probe_ref(q0, k, ctx.dt, T=T, amp=amp); v = ar + Kp * (qr - obs.q) + Kd * (vr - obs.qd); tau = np.clip(arm.inverse(obs.q, obs.qd, v), -ctx.tau_limit, ctx.tau_limit)
        obs, *_ = env.step(tau); taus.append(tau); Qm.append(obs.q.copy()); Vm.append(obs.qd.copy())
    return q0, np.array(taus), np.array(Qm), np.array(Vm)
def probe_features10(ctx, q0, taus, Qm, Vm):
    Q0, V0, _ = replay3(ctx, np.concatenate([[1.0], np.ones(NL), [1.0, 0.0]]), q0, taus); f = np.concatenate([(Qm - Q0)[LSTRIDE - 1::LSTRIDE].ravel(), (Vm - V0)[LSTRIDE - 1::LSTRIDE].ravel()])
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0).astype(np.float32)
def gen_probe10(args):
    """list of (seed, rep, cond), T, amp_scale -> (X, TH) with TH the true theta13"""
    items, T, amp = args; X, TH = [], []; AMP = np.array([0.5, 0.5, 0.4]) * amp
    for sd, rep, cond in items:
        guard([sd]); sc = scen(sd, cond, rep); env = sc.make_env(); ctx = env.ctx; obs = env.reset(sd); q0, taus, Qm, Vm = probe_run3(env, ctx, obs, T, AMP)
        X.append(probe_features10(ctx, q0, taus, Qm, Vm)); TH.append(true_theta(sd, cond, rep))
    return np.array(X), np.array(TH)
def theta_to_y(TH, kind):
    ls, ms, ds, fr = TH[:, 0], TH[:, 1:11], TH[:, 11], TH[:, 12]
    if kind == 'full': return np.column_stack([ls, np.log(ms), np.log(ds), fr * 1e3])
    return np.column_stack([ls, np.log(ms.mean(1)), np.log(ds), fr * 1e3])      # eff: total mass = 10 x mean link mass (uniform chain)
def y_to_theta(y, kind):
    if kind == 'full': th = np.concatenate([[y[0]], np.exp(y[1:11]), [np.exp(y[11]), y[12] * 1e-3]])
    else: th = np.concatenate([[y[0]], np.full(NL, np.exp(y[1])), [np.exp(y[2]), y[3] * 1e-3]])
    return clip_theta(th)
class MLPEst10:
    """numpy forward of a trained tanh MLP: probe features -> theta13 (kind 'full' or 'eff')"""
    def __init__(self, st): self.st = st
    def __call__(self, ctx, q0, taus, Qm, Vm):
        st = self.st; x = (probe_features10(ctx, q0, taus, Qm, Vm) - st['mu']) / st['sd']
        for i, (Wt, b) in enumerate(st['layers']):
            x = Wt @ x + b
            if i < len(st['layers']) - 1: x = np.tanh(x)
        return y_to_theta(x * st['ysd'] + st['ymu'], st['kind'])
def boot_ci(x, **k): return A.boot_ci(x, **k)
