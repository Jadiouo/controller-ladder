"""spike13b shared code: headroom ladder R0..R5 + fragility sweep. Imports spike10b / spike11a unchanged. Run with cwd = .scratch/run13."""
import sys, os, json, time, dataclasses, numpy as np
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike11a')); sys.path.insert(0, os.path.join(_H, '..', 'spike10b'))
import c10b, c11, s11
from r10b import *
OUT13 = os.path.join(ROOT, 'results/spike13b'); os.makedirs(OUT13, exist_ok=True)
def guard13(seed):
    s = int(seed); ok = (1000 <= s < 1050) or (1100 <= s <= 1599) or (700600 <= s <= 700999)
    assert ok, f'seed {s} not allowed in spike13b (never TEST 0-49 / TEST-B 20000-20499)'
c11.guard11 = lambda seeds, cal=False: [guard13(x) for x in np.atleast_1d(seeds)]; s11.guard11 = c11.guard11
ALLOW = ((700000, 800000),)
# ---------------------------------------------------------------- per-factor noise scale (sensor 'a', actuator 'c'); other factors stay at x1
SC = {}
for m in (0.5, 2, 4, 8):
    for f in 'ac':
        nm = f'abc_{f}{m}'; c10b.CONDS[nm] = (1, 1, 1, 1.0); SC[nm] = (m if f == 'a' else 1.0, m if f == 'c' else 1.0)
class Real3(c10b.Real2):
    def __init__(self, seed, rep, cond): super().__init__(seed, rep, cond); self.sa, self.sc = SC.get(cond, (1.0, 1.0))
    def on_obs(self, obs, env):
        if not self.a: return obs
        return obs.replace(q=obs.q + c10b.SIG_Q * self.s * self.sa * self.rng.standard_normal(len(obs.q)), qd=obs.qd + c10b.SIG_QD * self.s * self.sa * self.rng.standard_normal(len(obs.qd)))
    def on_command(self, tau, k): return tau + self.d['act_sigma'] * self.sc * self.rng2.standard_normal(len(tau)) if self.c else tau
c10b.Real2 = Real3
# ---------------------------------------------------------------- rungs
class Cheat:
    def __init__(self, th): self.th = np.asarray(th, float)
    def __call__(self, ctx, q0, taus, Qm, Vm): return self.th
class Probe3True(Probe3):
    """R2: as Probe3 with the true theta, but the swing planner starts from the TRUE simulator state at the end of the probe."""
    env = None
    def _swing_start(self, obs):
        th = np.asarray(self.est_fn(None, None, None, None, None), float); self.est = th; m = model_theta(self.ctx, th); d = self.env.data
        x0 = (np.array(d.qpos[:m.nq], float), np.array(d.qvel[:m.nv], float))
        inner = Tuned3S(self.ctx, th, x0, self.ctx.task_info['duration'] - self.T, self.qp, self.qv); inner.reset(self.seed)
        self.q0 = np.asarray(obs.q, float).copy(); self.knots = inner.plan(self.q0, np.asarray(obs.ref['target'], float)); self.plan_info = dict(inner.plan_info)
class OracleDelay(OracleFull):
    """R3: true plant + true rest state; arm holds q0 for nd steps (same clock as the probe), then a swing planned for the remaining horizon."""
    def __init__(self, ctx, arrays, fric, nd):
        self.nd = nd; super().__init__(dataclasses.replace(ctx, task_info={**ctx.task_info, 'duration': ctx.task_info['duration'] - nd * ctx.dt}), arrays, fric); self.hq0 = None
    def act(self, obs):
        k = obs.k
        if k == 0: self.hq0 = np.asarray(obs.q, float).copy()
        if k < self.nd:
            v = self.Kp * (self.hq0 - obs.q) + self.Kd * (0.0 - obs.qd); return np.clip(self.model.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
        if self.knots is None:
            self.q0 = np.asarray(obs.q, float).copy(); self.knots = self.plan(self.q0, np.asarray(obs.ref['target'], float)); self.plan_info = dict(self.plan_info)
        t = np.array([(k - self.nd) * self.ctx.dt]); qr, qdr, qddr = (a[0] for a in WT.spline_traj(self.q0, self.knots, t)); self.qr, self.qdr = qr, qdr
        v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd); return np.clip(self.model.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
class Hybrid:
    """supplementary decomposition of R0->R1: take the cached T-adapt3-full fit of this episode and replace a GROUP of parameters by the true values ('in'), or take the true theta and replace a group by the fit ('out')."""
    G = dict(ls=[0], m=list(range(1, 11)), ds=[11], fr=[12])
    def __init__(self, th, mode, grp): self.th, self.mode, self.grp = np.asarray(th, float), mode, grp; self.cfg = spec_of('adapt3_full')[1]
    def __call__(self, ctx, q0, taus, Qm, Vm):
        key = est_key(cfg_tag(self.cfg), q0, taus, Qm); f = os.path.join(PL, f'est3_{key}.npy')
        fit = np.load(f) if os.path.exists(f) else np.asarray(fit3(ctx, q0, taus, Qm, Vm, self.cfg), float); idx = self.G[self.grp]
        if self.mode == 'in': out = fit.copy(); out[idx] = self.th[idx]
        else: out = self.th.copy(); out[idx] = fit[idx]
        return out
def _arrays(seed, cond):
    th = true_theta(seed, cond, None); return th, theta_arrays(type('C', (), dict(task_info=dict(chain_len=np.full(NL, .1), chain_mass=np.full(NL, .015), chain_damp=np.full(NL, .0004))))(), th)
def _metrics(sc, tr): m = episode_metrics(sc, tr); m['inst'] = int(INST[0]); return m
def run13(args):
    """args=(method, seed, cond) -> metrics. methods: nom robust oracle(R4) adapt3_full(R0) lest_full#s | R1 R2 R3 R5 | lestcyc (lest_full#(seed%3))"""
    meth, seed, cond = args; guard13(seed); pr = probe_of_lest()
    if meth == 'lestcyc': meth = f'lest_full#{int(seed) % 3}'
    if meth in ('nom', 'robust', 'oracle', 'adapt3_full') or meth.startswith('lest_'):
        m = run10((spec_of(meth), seed, None, cond, ALLOW)); return {k: m[k] for k in ('success', 'min_dist', 'inst', 'diverged')}
    reset_probe_globals(); INST[0] = 0; sc = scen(seed, cond, None); th, arr = _arrays(seed, cond)
    if meth == 'R1':
        tr = run_episode(sc, lambda ctx: Probe3(ctx, Cheat(th), f'cheat{seed}', pr['T'], np.array([0.5, 0.5, 0.4]) * pr['amp']), int(seed)); return _metrics(sc, tr)
    if meth.startswith('H_'):
        _, mode, grp = meth.split('_'); tr = run_episode(sc, lambda ctx: Probe3(ctx, Hybrid(th, mode, grp), f'hyb{meth}{seed}', pr['T'], np.array([0.5, 0.5, 0.4]) * pr['amp']), int(seed)); return _metrics(sc, tr)
    if meth == 'R2':
        env = sc.make_env()
        def fac(ctx): p = Probe3True(ctx, Cheat(th), f'cheat{seed}', pr['T'], np.array([0.5, 0.5, 0.4]) * pr['amp']); p.env = env; return p
        tr = run_episode(sc, fac, int(seed), env=env); return _metrics(sc, tr)
    if meth == 'R3':
        tr = run_episode(sc, lambda ctx: OracleDelay(ctx, arr, float(th[12]), int(round(pr['T'] / ctx.dt))), int(seed)); return _metrics(sc, tr)
    if meth == 'R5':
        m = s11.run_ep(dict(seed=int(seed), rep=None, cond=cond, student=None, H=6, stride=5, label='fixed', canon=True)); return {k: m[k] for k in ('success', 'min_dist', 'inst', 'diverged')}
    raise ValueError(meth)
def boot(x, n=20000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); b = x[r.integers(0, len(x), (n, len(x)))].mean(1); return float(np.quantile(b, .025)), float(np.quantile(b, .975))
