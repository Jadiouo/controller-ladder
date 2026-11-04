"""spike09b shared code (whip). Run from repo root with .venv/bin/python.
Controllers: Tuned (spike09a, planner on an assumed chain), Probe (probe segment -> estimate chain (ls,ms,ds) -> same planner).
Estimators: 'fit' (T-adapt: nonlinear least squares of a nominal-model replay), 'mlp' (L-est), 'oracle' (true chain).
Scenarios: chain error drawn per episode from spike09a dist (w=0.2); either from the seed (eval seeds) or from an explicit (seed, rep) for training."""
import sys, os, json, hashlib, importlib.util, dataclasses, numpy as np, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, ROOT)
_sp = importlib.util.spec_from_file_location('c9a', os.path.join(ROOT, 'experiments/spike09a/common.py')); A = importlib.util.module_from_spec(_sp); sys.modules['c9a'] = A; _sp.loader.exec_module(A)
from arena import Scenario
from arena.tasks import make_task
from arena.robots import build_mjcf
from arena.perturb import ChainMass, ChainDamping, ChainLength
from arena.eval.rollout import run_episode
from arena.controllers import whip_teacher as WT
from arena.controllers._arm import ArmModel, lqr_double_integrator
OUT = os.path.join(ROOT, 'results/spike09b'); PLANS = os.path.join(ROOT, '.scratch/s09b_plans'); os.makedirs(PLANS, exist_ok=True)
W = 0.2
DEF = A.DEF; ROBUST = json.load(open(os.path.join(ROOT, 'results/spike09a/robust.json')))['params']
def guard(seeds, allow_test=False):
    if not allow_test: assert np.min(seeds) >= 1000, 'TEST seeds forbidden here'
def true_chain(seed, rep=None, w=W):
    """true chain error. rep=None: derived from the seed (spike09a draw). rep=int: independent draw for (seed, rep) (training scenarios)."""
    if rep is None: return A.draw_true(seed, w)
    r = np.random.default_rng([int(seed), int(rep), 9110]); a, b, c = r.uniform(-1, 1, 3)
    return dict(len_rel=float(w * a), mass_rel=float(np.exp(5 * w * b) - 1), damp_fac=float(np.exp(10 * w * c)))
def scenario_of(t):
    return Scenario('whip').with_perturb(ChainLength(t['len_rel']), ChainMass(t['mass_rel']), ChainDamping(t['damp_fac'] - 1))
def tvec(t): return np.array([1 + t['len_rel'], 1 + t['mass_rel'], t['damp_fac']])     # (ls, ms, ds) of the true chain
# ---------------------------------------------------------------- probe
PROBE_T = 0.2; PROBE_AMP = np.array([0.5, 0.5, 0.4]) * 0.3       # tuned on CAL (see README); arm returns to q0 at rest at PROBE_T
def probe_ref(q0, k, dt, T=None, amp=None):
    T = PROBE_T if T is None else T; amp = PROBE_AMP if amp is None else amp
    t = min(k * dt, T); ph = 2 * np.pi * t / T; s = amp * (1 - np.cos(ph)) / 2; v = amp * np.sin(ph) * np.pi / T; a = amp * np.cos(ph) * 2 * np.pi ** 2 / T ** 2
    return q0 + s, v, a
def chain_model(ctx, ls, ms, ds):
    info = dict(ctx.task_info)
    for k, s in (('chain_len', ls), ('chain_mass', ms), ('chain_damp', ds)): info[k] = np.asarray(info[k], float) * s
    t = make_task('whip', ctx.robot, n_links=int(info['n_links'])); t.chain_len, t.chain_mass, t.chain_damp = (np.array(info[k], float) for k in ('chain_len', 'chain_mass', 'chain_damp'))
    return mujoco.MjModel.from_xml_string(build_mjcf(ctx.robot, t.tip_xml(ctx.robot)))
def replay(ctx, ls, ms, ds, q0, taus):
    """open-loop replay of the applied probe torques (zero-order hold per control step) on a chain model -> (q, qd) after each control step, (T,n) each"""
    m = chain_model(ctx, ls, ms, ds); d = mujoco.MjData(m); n = ctx.robot.n; d.qpos[:n] = q0; d.qpos[n] = np.pi / 2 - q0.sum(); mujoco.mj_forward(m, d)
    sub = int(round(ctx.dt / m.opt.timestep)); Q = np.empty((len(taus), n)); V = np.empty((len(taus), n))
    for j, tau in enumerate(taus):
        d.ctrl[:n] = tau
        for _ in range(sub): mujoco.mj_step(m, d)
        Q[j] = d.qpos[:n]; V[j] = d.qvel[:n]
    replay.last = (d.qpos.copy(), d.qvel.copy())
    return Q, V
def run_probe(env, ctx, obs, n_pr=None, Kp=None, Kd=None, arm=None):
    """execute the probe segment in env (CTC on probe_ref, DEF gains) -> (obs after, q0, taus, Qm, Vm)"""
    n_pr = int(round(PROBE_T / ctx.dt)) if n_pr is None else n_pr
    arm = arm or ArmModel(ctx.robot); Kp, Kd = lqr_double_integrator(DEF['qp'], DEF['qv']); q0 = obs.q.copy(); taus, Qm, Vm = [], [], []
    for k in range(n_pr):
        qr, vr, ar = probe_ref(q0, k, ctx.dt); v = ar + Kp * (qr - obs.q) + Kd * (vr - obs.qd); tau = np.clip(arm.inverse(obs.q, obs.qd, v), -ctx.tau_limit, ctx.tau_limit)
        obs, *_ = env.step(tau); taus.append(tau); Qm.append(obs.q.copy()); Vm.append(obs.qd.copy())
    return obs, q0, np.array(taus), np.array(Qm), np.array(Vm)
PRIOR_W = 0.0
LO = np.array([1 - W, -5 * W, -10 * W]); HI = np.array([1 + W, 5 * W, 10 * W])      # support of the error distribution (ls, log ms, log ds): given to T-adapt and L-est
def fit_chain(ctx, q0, taus, Qm, Vm, prior_w=None, ngrid=7, nstart=4, prefixes=(0.5, 0.75, 1.0)):
    """T-adapt estimator: coarse grid over the box (ls, log ms, log ds) + nonlinear least squares from the best starts, progressively over growing prefixes of the probe
    (the low-damping replay is chaotic: short prefixes first). Residual = (q,qd) of the nominal-model replay under the recorded probe torques minus measured."""
    from scipy.optimize import least_squares
    pw = PRIOR_W if prior_w is None else prior_w; sq, sv = 0.01, 0.1; nfev = 0
    def res_T(x, T):
        try: Q, V = replay(ctx, x[0], np.exp(x[1]), np.exp(x[2]), q0, taus[:T])
        except Exception: return np.full(T * 6 + 3, 1e3)
        r = np.concatenate([((Q - Qm[:T]) / sq).ravel(), ((V - Vm[:T]) / sv).ravel(), pw * np.array([(x[0] - 1) / 0.2, x[1] / 1.0, x[2] / 2.0])])
        return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3).clip(-1e3, 1e3)
    n = len(taus); Ts = [max(5, int(round(f * n))) for f in prefixes]
    g = [np.linspace(LO[i], HI[i], ngrid) for i in range(3)]; pts = np.array([(a, b, c) for a in g[0] for b in g[1] for c in g[2]])
    cs = np.array([0.5 * (res_T(p, Ts[0]) ** 2).sum() for p in pts]); nfev += len(pts); best = None
    for j in np.argsort(cs)[:nstart]:
        x = pts[j]
        for T in Ts:
            r = least_squares(res_T, x, args=(T,), bounds=(LO, HI), x_scale=[0.1, 0.5, 1.0], diff_step=1e-3, max_nfev=30); x = r.x; nfev += r.nfev
        if best is None or r.cost < best.cost: best = r
    return np.array([best.x[0], np.exp(best.x[1]), np.exp(best.x[2])]), float(best.cost), int(nfev)
# ---------------------------------------------------------------- controllers
class Tuned(A.Tuned):
    """spike09a Tuned planner, with the closed-loop re-scoring horizon settable (probe episodes have less time left)."""
    def __init__(self, ctx, P=None, duration=None, **kw):
        if duration is not None: ctx = dataclasses.replace(ctx, task_info={**ctx.task_info, 'duration': float(duration)})
        super().__init__(ctx, P, **kw); self.dur = float(ctx.task_info['duration'])
    def _fkey(self, extra=''):
        P = self.P; return hashlib.md5(repr((int(self.seed), round(P['ls'], 6), round(P['ms'], 6), round(P['ds'], 6), round(P['qp'], 1), round(P['qv'], 2), round(self.dur, 4), extra)).encode()).hexdigest()
    def plan(self, q0, target):
        f = os.path.join(PLANS, f'fin_{self._fkey(getattr(self, "xh", ""))}.npy')
        if os.path.exists(f):
            try: return np.load(f)
            except Exception: pass
        k = self._plan_impl(q0, target); np.save(f + f'.{os.getpid()}.npy', k); os.replace(f + f'.{os.getpid()}.npy', f); return k
    def _plan_impl(self, q0, target): return super().plan(q0, target)
    n_pr = 0
    def act(self, obs):
        tau = super().act(obs); qr, qdr, _ = (a[0] for a in WT.spline_traj(self.q0, self.knots, np.array([obs.k * self.ctx.dt]))); self.qr, self.qdr = qr, qdr; return tau
EST_CACHE = {}
class SwingSimS(WT.SwingSim):
    """SwingSim whose initial state is an explicit (qpos, qvel) of arm + chain (the model-estimated state at the end of the probe)."""
    x0 = None
    def initial_state(self, q0):
        mujoco.mj_resetData(self.model, self.data); self.data.qpos[:] = self.x0[0]; self.data.qvel[:] = self.x0[1]; mujoco.mj_forward(self.model, self.data)
        s = np.zeros(self.nstate); mujoco.mj_getState(self.model, self.data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS); return s
class TunedS(Tuned):
    """Tuned planner that starts from an explicit full state x0=(qpos,qvel) (chain swinging after the probe)."""
    def __init__(self, ctx, P=None, x0=None, **kw):
        super().__init__(ctx, P, **kw); self.x0 = x0; self.sim = SwingSimS(self.ctx); self.sim.x0 = x0; self.xh = hashlib.md5(np.round(np.concatenate(x0), 7).tobytes()).hexdigest()[:10]
    def _closed_loop(self, q0, Qk, target):
        ctx, info = self.ctx, self.ctx.task_info; m = ctx.make_model(); d = mujoco.MjData(m); n = ctx.robot.n
        d.qpos[:] = self.x0[0]; d.qvel[:] = self.x0[1]; mujoco.mj_forward(m, d); sid = m.site('whip_tip').id; arm = self.model; dmin = np.inf
        steps = int(round(info['duration'] / ctx.dt)); t = np.arange(steps) * ctx.dt; qr, qdr, qddr = WT.spline_traj(q0, Qk, t); sub = int(round(ctx.dt / m.opt.timestep))
        for k in range(steps):
            q, qd = d.qpos[:n].copy(), d.qvel[:n].copy(); v = qddr[k] + self.Kp * (qr[k] - q) + self.Kd * (qdr[k] - qd)
            d.ctrl[:n] = np.clip(arm.inverse(q, qd, v), -ctx.tau_limit, ctx.tau_limit); tb = d.time
            for _ in range(sub): mujoco.mj_step(m, d)
            if d.time < tb: return float('inf')
            dmin = min(dmin, float(np.linalg.norm(d.site_xpos[sid][[0, 2]] - target)))
            if dmin < info['r_hit'] * 0.5: break
        return dmin
    def _plan_impl(self, q0, target):
        P = self.P; xh = hashlib.md5(np.round(np.concatenate(self.x0), 7).tobytes()).hexdigest()[:10]
        k1 = hashlib.md5(repr((int(self.seed), round(P['ls'], 5), round(P['ms'], 5), round(P['ds'], 5), xh)).encode()).hexdigest(); f1 = os.path.join(PLANS, f's1S_{k1}.npy'); n = self.ctx.robot.n
        picked = None
        if os.path.exists(f1):
            try: picked = np.load(f1)
            except Exception: picked = None
        if picked is None:
            picked = self._stage1(q0, target); np.save(f1 + f'.{os.getpid()}.npy', picked); os.replace(f1 + f'.{os.getpid()}.npy', f1)
        scores = [self._closed_loop(q0, x.reshape(4, n), target) for x in picked]; j = int(np.argmin(scores)); self.plan_info = dict(planned_min_dist=scores[j]); return picked[j].reshape(4, n)
def est_key(tag, q0, taus, Qm): return hashlib.md5(tag.encode() + np.round(np.concatenate([q0, taus.ravel(), Qm.ravel()]), 9).tobytes()).hexdigest()
class Probe(WT.WhipTeacher.__mro__[1]):          # plain Controller
    """probe segment (CTC on a fixed small excitation, DEF gains) -> estimate (ls, ms, ds) with est_fn -> plan on the estimated chain -> execute the swing, time-shifted by the probe.
    est_fn(ctx, q0, taus, Qm, Vm) -> (ls, ms, ds). `res_fn` (optional) is unused here (residual wrappers are separate)."""
    def __init__(self, ctx, est_fn, tag='fit', qp=DEF['qp'], qv=DEF['qv']):
        super().__init__(ctx)
        self.est_fn, self.tag = est_fn, tag; self.arm = ArmModel(ctx.robot); self.Kp, self.Kd = lqr_double_integrator(qp, qv); self.qp, self.qv = qp, qv
        self.n_pr = int(round(PROBE_T / ctx.dt)); self.knots = None; self.taus, self.Qm, self.Vm = [], [], []; self.est = None; self.q0 = None; self.plan_info = {}
    def reset(self, seed=None):
        super().reset(seed); self.knots = None; self.taus, self.Qm, self.Vm = [], [], []; self.est = None
    def history(self): return np.array(self.taus), np.array(self.Qm), np.array(self.Vm)
    def _swing_start(self, obs):
        taus, Qm, Vm = self.history(); self.pq0 = self.hq0
        key = est_key(self.tag, self.pq0, taus, Qm)
        if key not in EST_CACHE:
            f = os.path.join(PLANS, f'est_{key}.npy')
            if os.path.exists(f):
                try: EST_CACHE[key] = tuple(float(x) for x in np.load(f))
                except Exception: pass
            if key not in EST_CACHE:
                EST_CACHE[key] = tuple(float(x) for x in self.est_fn(self.ctx, self.pq0, taus, Qm, Vm)); np.save(f + f'.{os.getpid()}.npy', np.array(EST_CACHE[key])); os.replace(f + f'.{os.getpid()}.npy', f)
        self.est = EST_CACHE[key]; ls, ms, ds = self.est
        replay(self.ctx, ls, ms, ds, self.pq0, taus); x0 = replay.last          # chain state at the end of the probe, reconstructed on the estimated chain
        inner = TunedS(self.ctx, dict(ls=ls, ms=ms, ds=ds, qp=self.qp, qv=self.qv), x0=x0, duration=self.ctx.task_info['duration'] - PROBE_T); inner.reset(self.seed)
        self.q0 = np.asarray(obs.q, float).copy(); self.knots = inner.plan(self.q0, np.asarray(obs.ref['target'], float)); self.plan_info = dict(inner.plan_info)
    def act(self, obs):
        k = obs.k
        if k == 0: self.hq0 = np.asarray(obs.q, float).copy()
        if k > 0 and k <= self.n_pr: self.Qm.append(np.asarray(obs.q, float).copy()); self.Vm.append(np.asarray(obs.qd, float).copy())
        if k < self.n_pr:
            qr, vr, ar = probe_ref(self.hq0, k, self.ctx.dt); self.qr, self.qdr = qr, vr; v = ar + self.Kp * (qr - obs.q) + self.Kd * (vr - obs.qd)
            tau = np.clip(self.arm.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit); self.taus.append(tau.copy()); return tau
        if self.knots is None: self._swing_start(obs)
        t = np.array([(k - self.n_pr) * self.ctx.dt]); qr, qdr, qddr = (a[0] for a in WT.spline_traj(self.q0, self.knots, t))
        self.qr, self.qdr = qr, qdr; v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd); return np.clip(self.arm.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
def fit_est(ctx, q0, taus, Qm, Vm): return fit_chain(ctx, q0, taus, Qm, Vm)[0]
METHODS = {}
def make_factory(spec):
    """spec: 'nom' | 'robust' | 'oracle:<ls>,<ms>,<ds>' | 'adapt' | ('est', tag, est_fn)"""
    if spec == 'nom': return lambda ctx: Tuned(ctx, DEF)
    if spec == 'robust': return lambda ctx: Tuned(ctx, ROBUST)
    if spec == 'adapt': return lambda ctx: Probe(ctx, fit_est, 'fit')
    if isinstance(spec, tuple) and spec[0] == 'est': return lambda ctx: Probe(ctx, spec[2], spec[1])
    if isinstance(spec, tuple) and spec[0] == 'oracle': return lambda ctx: Tuned(ctx, dict(ls=spec[1][0], ms=spec[1][1], ds=spec[1][2], qp=DEF['qp'], qv=DEF['qv']))
    raise ValueError(spec)
def episode_metrics(sc, tr):
    ex = sc.make_env().task.extra_metrics(tr); T = max(len(tr.tau_applied), 1)
    return dict(success=float(ex['success']), min_dist=float(ex['min_dist']), hit_speed=float(ex['hit_speed']), energy=float((tr.tau_applied ** 2).sum() / T), diverged=float(tr.diverged))
def run_spec(args):
    """args = (spec, seed, rep, allow_test): one episode of a classical/estimator method on the chain drawn for (seed, rep)."""
    spec, seed, rep, allow = args; guard([seed], allow); t = true_chain(seed, rep); sc = scenario_of(t)
    tr = run_episode(sc, make_factory(spec), int(seed)); return {**episode_metrics(sc, tr), 'chain': tvec(t).tolist()}
# ---------------------------------------------------------------- L-est (learned estimator)
LSTRIDE = 2
def probe_features(ctx, q0, taus, Qm, Vm):
    """probe-segment history -> feature vector: (q, qd) measured minus the open-loop replay of the same torques on the NOMINAL chain, every LSTRIDE steps (scaled later by train std)."""
    Q0, V0 = replay(ctx, 1.0, 1.0, 1.0, q0, taus); f = np.concatenate([(Qm - Q0)[LSTRIDE - 1::LSTRIDE].ravel(), (Vm - V0)[LSTRIDE - 1::LSTRIDE].ravel()])
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0).astype(np.float32)      # unstable (diverged) simulations produce garbage: clipped
def y_of(tv): return np.array([(tv[0] - 1) / W, np.log(tv[1]) / (5 * W), np.log(tv[2]) / (10 * W)], np.float32)      # in [-1,1]
def y_inv(y): y = np.clip(y, -1, 1); return np.array([1 + W * y[0], np.exp(5 * W * y[1]), np.exp(10 * W * y[2])])
def gen_probe_data(seeds_reps):
    """probe-only episodes (no swing): features and true chain. Uses the same probe controller as T-adapt."""
    X, Y = [], []
    for sd, rep in seeds_reps:
        assert sd >= 500000; t = true_chain(sd, rep); env = scenario_of(t).make_env(); ctx = env.ctx; obs = env.reset(sd)
        _, q0, taus, Qm, Vm = run_probe(env, ctx, obs); X.append(probe_features(ctx, q0, taus, Qm, Vm)); Y.append(y_of(tvec(t)))
    return np.array(X), np.array(Y)
class MLPEst:
    """numpy forward of a trained tanh MLP estimator with input standardisation: est_fn for Probe"""
    def __init__(self, st): self.st = st
    def __call__(self, ctx, q0, taus, Qm, Vm):
        x = (probe_features(ctx, q0, taus, Qm, Vm) - self.st['mu']) / self.st['sd']
        for i, (Wt, b) in enumerate(self.st['layers']):
            x = Wt @ x + b
            if i < len(self.st['layers']) - 1: x = np.tanh(x)
        return y_inv(x)
# ---------------------------------------------------------------- residual RL pieces
BOUND = 12.0                  # residual cap (Nm) per joint, as spike07
LAGS = list(range(5, 41, 5))  # history: (q - q_now, qd) at these lags (steps)
def base_spec_factory(base):
    """base: 'nom' | 'robust' | 'adapt' | ('est', tag, fn)"""
    return make_factory(base)
class Feat:
    def __init__(self, hist, oracle): self.hist, self.oracle = hist, oracle; self.dim = 19 + (6 * len(LAGS) if hist else 0) + (3 if oracle else 0)
    def reset(self): self.buf = []
    def push(self, obs): self.buf.append(np.concatenate([obs.q, obs.qd]))
    def __call__(self, obs, tau_base, base, yt=None):
        qr, qdr = getattr(base, 'qr', obs.q), getattr(base, 'qdr', obs.qd); ts = max(0, obs.k - getattr(base, 'n_pr', 0)) * 0.005
        f = [obs.q, obs.qd * 0.2, np.asarray(obs.ref['target']) / 1.7, [obs.k * 0.005 / 2.5, ts / 2.0], tau_base / 30, (obs.q - qr) * 20, (obs.qd - qdr) * 3]
        if self.hist:
            cur = self.buf[-1]; h = []
            for L in LAGS:
                p = self.buf[max(len(self.buf) - 1 - L, 0)]; h.append(np.concatenate([(p[:3] - cur[:3]) * 3, p[3:] * 0.2]))
            f.append(np.concatenate(h))
        if self.oracle: f.append(yt)
        return np.concatenate([np.ravel(x) for x in f]).astype(np.float32)
def np_mlp(state, x):
    for i, (Wt, b) in enumerate(state):
        x = Wt @ x + b
        if i < len(state) - 1: x = np.tanh(x)
    return x
class ResEnv:
    """one training/eval episode of residual control: env + base controller + featurizer. step(a) with a in [-1,1]^3."""
    def __init__(self, base, fz, seed, rep):
        t = true_chain(seed, rep); self.sc = scenario_of(t); self.env = self.sc.make_env(); self.ctx = self.env.ctx; self.base = make_factory(base)(self.ctx); self.fz = fz; self.yt = y_of(tvec(t)); self.seed = seed
        self.obs = self.env.reset(seed); self.base.reset(seed); self.fz.reset(); self.fz.push(self.obs); self.task = self.env.task
        self.dmin = float(np.linalg.norm(self.task.chain_tip(self.env) - self.task.target)); self.rn = []; self.tips = [self.task.chain_tip(self.env).copy()]; self.en = []
        self.tb = self.base.act(self.obs); self.x = self.fz(self.obs, self.tb, self.base, self.yt)
    def step(self, a):
        a = np.clip(a, -1, 1); res = BOUND * a * (self.obs.k >= getattr(self.base, 'n_pr', 0)); tau = np.clip(self.tb + res, -self.ctx.tau_limit, self.ctx.tau_limit)
        if self.obs.k >= getattr(self.base, 'n_pr', 0): self.rn.append(float(np.linalg.norm(res)))
        obs, _, term, trunc, info = self.env.step(tau); self.en.append(float((info['applied'] ** 2).sum())); self.tips.append(self.task.chain_tip(self.env).copy()); d = float(np.linalg.norm(self.task.chain_tip(self.env) - self.task.target)); r = 10.0 * max(self.dmin - d, 0.0); self.dmin = min(self.dmin, d)
        succ = info['outcome'] == 'success'; r += 5.0 * succ - 5.0 * (info['outcome'] == 'failure') - 2e-6 * float((res ** 2).sum()); done = term or trunc
        if not done: self.obs = obs; self.fz.push(obs); self.tb = self.base.act(obs); self.x = self.fz(obs, self.tb, self.base, self.yt)
        return self.x, r, done, dict(success=succ, min_dist=self.dmin)
def run_residual_episode(args):
    """evaluation of a residual policy (numpy MLP layer list, or None = zero residual) on one episode -> metrics (same definitions as arena Task.extra_metrics / energy)"""
    base, hist, oracle, state, seed, rep, allow = args; guard([seed], allow); fz = Feat(hist, oracle); e = ResEnv(base, fz, seed, rep); done = False; x = e.x
    while not done:
        a = np.zeros(3) if state is None else np.clip(np_mlp(state, x.astype(np.float64)), -1, 1); x, r, done, info = e.step(a)
    tip = np.array(e.tips); dist = np.linalg.norm(tip - e.task.target, axis=1); hit = np.where(dist < e.task.r_hit)[0]
    hs = float(np.linalg.norm(tip[max(int(hit[0]), 1)] - tip[max(int(hit[0]), 1) - 1]) / e.ctx.dt) if len(hit) and len(tip) > 1 else float('nan')
    return dict(success=float(info['success']), min_dist=float(dist.min()), hit_speed=hs, energy=float(np.mean(e.en)), diverged=0.0, res_mean=float(np.mean(e.rn)) if e.rn else 0.0, res_max=float(np.max(e.rn)) if e.rn else 0.0,
                steps=len(e.en), chain=tvec(true_chain(seed, rep)).tolist())
from arena.controllers.base import Controller
class ResCtl(Controller):
    """residual controller for run_episode / video (same features as ResEnv). yt = oracle label (only for the oracle variant)."""
    def __init__(self, ctx, base, hist, oracle, state, yt=None):
        super().__init__(ctx); self.b = make_factory(base)(ctx); self.fz = Feat(hist, oracle); self.state, self.yt = state, yt; self.rn = []
    def reset(self, seed=None): super().reset(seed); self.b.reset(seed); self.fz.reset(); self.rn = []
    def act(self, obs):
        self.fz.push(obs); tb = self.b.act(obs); x = self.fz(obs, tb, self.b, self.yt); a = np.clip(np_mlp(self.state, x.astype(np.float64)), -1, 1) if self.state is not None else np.zeros(3)
        on = obs.k >= getattr(self.b, 'n_pr', 0)
        if on: self.rn.append(float(np.linalg.norm(BOUND * a)))
        return np.clip(tb + BOUND * a * on, -self.ctx.tau_limit, self.ctx.tau_limit)
