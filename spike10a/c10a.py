"""spike10a: realistic-condition layer on top of spike09a dist (w=0.2) + evaluation of T-nom/T-robust/T-adapt/T-adapt2/T-oracle (in-class, full).
CAL 1000-1049 and holdout 1100-1599 only. TEST never touched (guard). Run from repo root."""
import sys, os, json, hashlib, dataclasses, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spike09b'))
import c9b
from c9b import *
from arena.perturb.base import Perturbation
OUT = os.path.join(ROOT, 'results/spike10a'); PLANS10 = os.path.join(ROOT, '.scratch/s10a_plans'); os.makedirs(PLANS10, exist_ok=True); os.makedirs(OUT, exist_ok=True)
# ---- realistic-condition constants (fixed a priori, before any evaluation)
SIG_Q, SIG_QD = 1e-3, 1e-2            # (a) encoder noise: position 1e-3 rad, velocity 1e-2 rad/s, iid gaussian per control step, on the 3 arm joints (the chain is never observed by anyone)
MASS_SPREAD = 0.3                     # (b) per-link mass factor exp(U(-0.3,0.3)) iid per link (on top of the global spike09a mass factor)
FRIC0 = 2e-3                          # (b) Coulomb (dry) friction on every chain hinge, Nm: FRIC0 * U(0.5,1.5) per episode (one value for the whole chain)
ACT_LO, ACT_HI = 0.01, 0.02           # (c) actuator torque noise sigma ~ U(0.01,0.02) Nm per episode, iid per step per joint, added to the applied torque (controller does not know it)
COND = {'none': (0, 0, 0), 'a': (1, 0, 0), 'b': (0, 1, 0), 'c': (0, 0, 1), 'abc': (1, 1, 1)}
def guard(seeds): assert np.min(seeds) >= 1000 and np.max(seeds) < 10000, 'TEST/training seeds forbidden'
def draw_real(seed):
    r = np.random.default_rng([int(seed), 1010]); f = np.exp(MASS_SPREAD * r.uniform(-1, 1, 10)); fr = FRIC0 * r.uniform(0.5, 1.5); sg = r.uniform(ACT_LO, ACT_HI)
    return dict(mass_f=f, fric=float(fr), act_sigma=float(sg))
class Real(Perturbation):
    """a: sensor noise on obs.q/qd; b: per-link mass factors (modify_task) + chain Coulomb friction (model.dof_frictionloss); c: actuator noise"""
    def __init__(self, seed, a, b, c):
        self.seed, self.a, self.b, self.c = int(seed), a, b, c; self.d = draw_real(seed)
    def modify_task(self, task):
        if self.b: task.chain_mass = np.asarray(task.chain_mass, float) * self.d['mass_f']
        return task
    def reset(self, seed, env):
        self.rng = np.random.default_rng([self.seed, 1011]); self.rng2 = np.random.default_rng([self.seed, 1012]); n = env.n
        if self.b: env.model.dof_frictionloss[n:] = self.d['fric']
    def on_obs(self, obs, env):
        if not self.a: return obs
        return obs.replace(q=obs.q + SIG_Q * self.rng.standard_normal(len(obs.q)), qd=obs.qd + SIG_QD * self.rng.standard_normal(len(obs.qd)))
    def on_command(self, tau, k):
        return tau + self.d['act_sigma'] * self.rng2.standard_normal(len(tau)) if self.c else tau
    def label(self): return f'real{int(self.a)}{int(self.b)}{int(self.c)}'
def scenario10(seed, cond):
    a, b, c = COND[cond]; t = true_chain(seed); sc = scenario_of(t)
    return sc.with_perturb(Real(seed, a, b, c)) if (a or b or c) else sc
def true_arrays(seed, cond):
    t = true_chain(seed); a, b, c = COND[cond]; d = draw_real(seed); nl = 10
    ml = np.full(nl, 0.015) * (1 + t['mass_rel']) * (d['mass_f'] if b else 1.0)
    return dict(chain_len=np.full(nl, 0.1) * (1 + t['len_rel']), chain_mass=ml, chain_damp=np.full(nl, 0.0004) * t['damp_fac']), (d['fric'] if b else 0.0)
def in_class_params(seed, cond):
    t = tvec(true_chain(seed)); b = COND[cond][1]; d = draw_real(seed)
    return [float(t[0]), float(t[1] * (d['mass_f'].mean() if b else 1.0)), float(t[2])]
# ---- full-plant oracle (planner sees true per-link arrays + Coulomb friction); no disk/in-process plan cache (keys would collide with T-nom)
class OracleFull(Tuned):
    def __init__(self, ctx, arrays, fric):
        info = {**ctx.task_info, **{k: np.asarray(v, float).copy() for k, v in arrays.items()}}
        super().__init__(dataclasses.replace(ctx, task_info=info), dict(DEF))
        base_mk = self.ctx.make_model; n = ctx.robot.n
        def mk():
            m = base_mk(); m.dof_frictionloss[n:] = fric; return m
        self.ctx = dataclasses.replace(self.ctx, make_model=mk)
    def plan(self, q0, target):
        n = self.ctx.robot.n; picked = self._stage1(q0, target); scores = [self._closed_loop(q0, x.reshape(4, n), target) for x in picked]; j = int(np.argmin(scores))
        self.plan_info = dict(planned_min_dist=scores[j]); return picked[j].reshape(4, n)
    def act(self, obs):
        if self.knots is None:
            self.q0 = np.asarray(obs.q, float).copy(); self.knots = self.plan(self.q0, np.asarray(obs.ref['target'], float)); self.plan_info = dict(self.plan_info)
        t = np.array([obs.k * self.ctx.dt]); qr, qdr, qddr = (a[0] for a in WT.spline_traj(self.q0, self.knots, t)); self.qr, self.qdr = qr, qdr
        v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd)
        return np.clip(self.model.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
# ---- T-adapt2 configurable estimator
A2 = {}      # active T-adapt2 config in this process: probe_T, amp_scale, loss, sq, sv, prefixes, ngrid
def fit_chain2(ctx, q0, taus, Qm, Vm, cfg):
    from scipy.optimize import least_squares
    sq, sv, loss = cfg['sq'], cfg['sv'], cfg['loss']; pw = 0.0; nfev = 0
    def res_T(x, T):
        try: Q, V = replay(ctx, x[0], np.exp(x[1]), np.exp(x[2]), q0, taus[:T])
        except Exception: return np.full(T * 6, 1e3)
        r = np.concatenate([((Q - Qm[:T]) / sq).ravel(), ((V - Vm[:T]) / sv).ravel()])
        return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3).clip(-1e3, 1e3)
    n = len(taus); Ts = [max(5, int(round(f * n))) for f in cfg['prefixes']]; ng = cfg['ngrid']
    g = [np.linspace(LO[i], HI[i], ng) for i in range(3)]; pts = np.array([(a, b, c) for a in g[0] for b in g[1] for c in g[2]])
    cs = np.array([0.5 * (res_T(p, Ts[0]) ** 2).sum() for p in pts]); best = None
    for j in np.argsort(cs)[:cfg['nstart']]:
        x = pts[j]
        for T in Ts:
            r = least_squares(res_T, x, args=(T,), bounds=(LO, HI), x_scale=[0.1, 0.5, 1.0], diff_step=1e-3, max_nfev=30, loss=loss, f_scale=cfg.get('f_scale', 1.0)); x = r.x
        if best is None or r.cost < best.cost: best = r
    return np.array([best.x[0], np.exp(best.x[1]), np.exp(best.x[2])])
def make_adapt2(cfg):
    tag = 'a2_' + hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    def est(ctx, q0, taus, Qm, Vm): return fit_chain2(ctx, q0, taus, Qm, Vm, cfg)
    def fac(ctx):
        c9b.PROBE_T = cfg['T']; c9b.PROBE_AMP = np.array([0.5, 0.5, 0.4]) * cfg['amp']; return Probe(ctx, est, tag)
    return fac
def run10(args):
    """args = (spec, seed, cond): spec 'nom'|'robust'|'adapt'|'oracle_in'|'oracle_full'|('adapt2', cfgdict) -> metrics"""
    spec, seed, cond = args; guard([seed]); sc = scenario10(seed, cond)
    c9b.PROBE_T = 0.2; c9b.PROBE_AMP = np.array([0.5, 0.5, 0.4]) * 0.3     # restore T-adapt (spike09b) probe in this worker
    if spec in ('nom', 'robust', 'adapt'): fac = make_factory(spec)
    elif spec == 'oracle_in': fac = make_factory(('oracle', in_class_params(seed, cond)))
    elif spec == 'oracle_full':
        arr, fr = true_arrays(seed, cond); fac = lambda ctx: OracleFull(ctx, arr, fr)
    else: fac = make_adapt2(spec[1])
    tr = run_episode(sc, fac, int(seed)); m = episode_metrics(sc, tr); c9b.PROBE_T = 0.2; c9b.PROBE_AMP = np.array([0.5, 0.5, 0.4]) * 0.3
    m['chain'] = tvec(true_chain(seed)).tolist(); return m
