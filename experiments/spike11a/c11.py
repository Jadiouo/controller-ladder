"""spike11a shared code: privileged-teacher distillation feasibility (whip, realistic conditions abc x1, SUBF=4 plant).
Imports spike10b (c10b/r10b) unchanged. Run with cwd = .scratch/run (MuJoCo writes MUJOCO_LOG.TXT in cwd).
Teacher = T-oracle planner (true per-link plant + friction, true chain state), generalised so that it can REPLAN from an arbitrary true state
(arm at any (q, qd), chain moving): clamped cubic spline with initial velocity v0, 4 free knots over a span S, zero end velocity.
At k=0 with v0=0, S=1.6 it is exactly the spike10b T-oracle plan structure."""
import sys, os, json, hashlib, time, dataclasses, numpy as np, mujoco
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'spike10b'))
from r10b import *
from scipy.interpolate import CubicSpline
from mujoco import rollout as mj_rollout
OUT11 = os.path.join(ROOT, 'results/spike11a'); os.makedirs(OUT11, exist_ok=True)
DT = 0.005; NSTEP = 500
def guard11(seeds, cal=False):
    s = np.asarray(seeds); ok = (s >= 500000) | ((s >= 1000) & (s < 1100)) if cal else (s >= 500000)
    assert ok.all(), 'seed range not allowed in spike11a'
# ---------------------------------------------------------------- generalised spline
def spline_g(q0, v0, Qk, S, t):
    """q, qd, qdd at times t (relative to the plan start) of the spline through (q0, Qk) at S*[0,.25,.5,.75,1], start velocity v0, end velocity 0; held after S"""
    X = np.vstack([q0[None], Qk]); kt = S * np.array([0, .25, .5, .75, 1.0])
    cs = CubicSpline(kt, X, bc_type=((1, np.asarray(v0, float)), (1, np.zeros(X.shape[1])))); tc = np.minimum(t, S); late = (t > S)[:, None]
    return cs(tc), np.where(late, 0.0, cs(tc, 1)), np.where(late, 0.0, cs(tc, 2))
class SwingSimG(WT.SwingSim):
    """batched PD-servo proxy starting from an explicit full state (qpos, qvel) with arm velocity; trajectories from spline_g"""
    def run_from(self, x0, q0, v0, S, Q, T_end, nthread=4):
        nstep = int(round(T_end / self.dt)); t = (np.arange(nstep) + 1) * self.dt
        mujoco.mj_resetData(self.model, self.data); self.data.qpos[:] = x0[0]; self.data.qvel[:] = x0[1]; mujoco.mj_forward(self.model, self.data)
        s0 = np.zeros(self.nstate); mujoco.mj_getState(self.model, self.data, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        ctrl = np.empty((len(Q), nstep, self.n))
        for b in range(len(Q)):
            qr, qdr, qddr = spline_g(q0, v0, Q[b], S, t); ctrl[b] = qr + (self.kd / self.kp) * qdr + qddr / self.omega ** 2
        out, _ = mj_rollout.rollout(self.model, [mujoco.MjData(self.model) for _ in range(max(1, min(nthread, len(Q))))], np.tile(s0, (len(Q), 1)), ctrl)
        bad = (np.diff(out[..., 0], axis=1) < 0).any(1); return self.tip_xz(out[..., 1:1 + self.model.nq]), bad
def oracle_ctx(ctx, seed, cond, rep):
    th = true_theta(seed, cond, rep); arr = theta_arrays(type('C', (), dict(task_info=dict(chain_len=np.full(NL, .1), chain_mass=np.full(NL, .015), chain_damp=np.full(NL, .0004))))(), th); fr = float(th[12])
    info = {**ctx.task_info, **{k: np.asarray(v, float).copy() for k, v in arr.items()}}; base_mk = ctx.make_model; n = ctx.robot.n; c2 = dataclasses.replace(ctx, task_info=info)
    def mk():
        t = make_task('whip', c2.robot, n_links=int(info['n_links'])); t.chain_len, t.chain_mass, t.chain_damp = (np.array(info[k], float) for k in ('chain_len', 'chain_mass', 'chain_damp'))
        m = mujoco.MjModel.from_xml_string(build_mjcf(c2.robot, t.tip_xml(c2.robot)))
        if fr > 0: m.dof_frictionloss[n:] = fr
        return m
    return dataclasses.replace(c2, make_model=mk)
class Replanner:
    """T-oracle planner from an arbitrary true state. cfg: generations, lam, restarts, sigma0, n_rescore, warm (start CMA from previous plan)"""
    CFG = dict(generations=40, lam=24, restarts=2, sigma0=0.6, n_rescore=8, reg=0.002)
    def __init__(self, ctx2, cfg=None):
        self.ctx = ctx2; self.cfg = {**self.CFG, **(cfg or {})}; self.n = ctx2.robot.n; self.sim = SwingSimG(ctx2); self.arm = ArmModel(ctx2.robot); self.Kp, self.Kd = lqr_double_integrator(DEF['qp'], DEF['qv'])
        self.model = ctx2.make_model(); self.data = mujoco.MjData(self.model); self.sid = self.model.site('whip_tip').id; self.ncall = 0; self.tcall = 0.0
    def _cost(self, out, Q, target):
        tips, bad = out; d = np.linalg.norm(tips - target, axis=-1); d = np.where(np.isfinite(d), d, 10.0).min(1); return d + 10.0 * bad + self.cfg['reg'] * np.abs(Q).mean((1, 2))
    def closed_loop(self, x0, q0, v0, S, Qk, target, Hrem):
        ctx, info = self.ctx, self.ctx.task_info; m, d, n = self.model, self.data, self.n; d.qpos[:] = x0[0]; d.qvel[:] = x0[1]; mujoco.mj_forward(m, d); d.time = 0.0
        steps = int(round(Hrem / ctx.dt)); t = np.arange(steps) * ctx.dt; qr, qdr, qddr = spline_g(q0, v0, Qk, S, t); sub = int(round(ctx.dt / m.opt.timestep)); dmin = np.inf
        for k in range(steps):
            q, qd = d.qpos[:n].copy(), d.qvel[:n].copy(); v = qddr[k] + self.Kp * (qr[k] - q) + self.Kd * (qdr[k] - qd); d.ctrl[:n] = np.clip(self.arm.inverse(q, qd, v), -ctx.tau_limit, ctx.tau_limit); tb = d.time
            for _ in range(sub): mujoco.mj_step(m, d)
            if d.time < tb: return float('inf')
            dmin = min(dmin, float(np.linalg.norm(d.site_xpos[self.sid][[0, 2]] - target)))
            if dmin < info['r_hit'] * 0.5: break
        return dmin
    def plan(self, x0, tcur, target, seed, prev=None, x0k=None):
        """x0 = (qpos, qvel) of arm+chain (true); returns dict(q0, v0, S, knots, score)"""
        t0 = time.time(); cfg, n = self.cfg, self.n; q0 = np.array(x0[0][:n]); v0 = np.array(x0[1][:n]); Hrem = self.ctx.task_info['duration'] - tcur; S = float(min(1.6, max(0.4, Hrem - 0.6)))
        rng = np.random.default_rng([int(seed), int(round(tcur / DT)), 0x57A1]); archive = []
        for r in range(cfg['restarts']):
            if x0k is not None and r == 0: x0c = np.asarray(x0k, float).ravel()
            elif prev is not None and r == 0:
                tk = S * np.array([.25, .5, .75, 1.0]); qk, _, _ = spline_g(prev['q0'], prev['v0'], prev['knots'], prev['S'], np.maximum(tcur - prev['t'] + tk, 0)); x0c = qk.ravel()
            else: x0c = np.tile(q0, 4) + (rng.normal(0, 0.5, 4 * n) if r else 0.0)
            es = WT.CMAES(x0c, cfg['sigma0'], rng, cfg['lam'])
            for g in range(cfg['generations']):
                X = np.clip(es.ask(), WT.Q_LO, WT.Q_HI); Q = X.reshape(-1, 4, n); f = self._cost(self.sim.run_from(x0, q0, v0, S, Q, S + 0.2), Q, target); es.tell(f); archive += [(float(fi), x.copy()) for fi, x in zip(f, X)]
                if f.min() < 0.01: break
        archive.sort(key=lambda a: a[0]); picked = []
        for fv, x in archive:
            if all(np.linalg.norm(x - p) > 0.3 for p in picked): picked.append(x)
            if len(picked) >= cfg['n_rescore']: break
        sc = [self.closed_loop(x0, q0, v0, S, x.reshape(4, n), target, Hrem) for x in picked]; j = int(np.argmin(sc)); self.ncall += 1; self.tcall += time.time() - t0
        return dict(q0=q0, v0=v0, S=S, knots=picked[j].reshape(4, n), score=float(sc[j]), t=float(tcur), cost=time.time() - t0)
# ---------------------------------------------------------------- canonical (continuation) teacher: plans are warm-started from a smooth grid of nominal-plant plans over the target
GRID_D = np.linspace(1.15, 1.70, 6); GRID_A = np.linspace(-0.45, 0.45, 7); GRID_F = os.path.join(OUT11, 'canon_grid.npz')
def canon_x0(target):
    g = np.load(GRID_F)['K']; d = np.hypot(*target); a = np.arctan2(target[1], target[0]); fd = np.interp(d, GRID_D, np.arange(len(GRID_D))); fa = np.interp(a, GRID_A, np.arange(len(GRID_A)))
    i, j = min(int(fd), len(GRID_D) - 2), min(int(fa), len(GRID_A) - 2); u, v = fd - i, fa - j
    return (1 - u) * (1 - v) * g[i, j] + u * (1 - v) * g[i + 1, j] + (1 - u) * v * g[i, j + 1] + u * v * g[i + 1, j + 1]
CANON_CFG = dict(sigma0=0.2, restarts=1)
# ---------------------------------------------------------------- teacher runtime
def ctc(arm, Kp, Kd, lim, q, qd, qr, qdr, qddr): return np.clip(arm.inverse(q, qd, qddr + Kp * (qr - q) + Kd * (qdr - qd)), -lim, lim)
class Teacher:
    """holds the current plan; replans every `every` steps while t <= tmax (every=None: plan once at k=0 = T-oracle). Provides reference triple at step k."""
    def __init__(self, env, seed, cond, rep, every=None, tmax=1.0, cfg=None, warm=False, canon=False):
        self.env, self.seed, self.every, self.tmax, self.warm, self.canon = env, seed, every, tmax, warm, canon; self.c2 = oracle_ctx(env.ctx, seed, cond, rep); self.rp = Replanner(self.c2, {**(CANON_CFG if canon else {}), **(cfg or {})}); self.plan_ = None; self.costs = []
    def state(self): d = self.env.data; nq = self.rp.model.nq; return (d.qpos[:nq].copy(), d.qvel[:self.rp.model.nv].copy())
    def maybe_plan(self, k, target):
        if self.plan_ is None or (self.every is not None and k % self.every == 0 and k * DT <= self.tmax + 1e-9 and k > 0):
            x0k = canon_x0(target) if (self.canon and self.plan_ is None) else None
            p = self.rp.plan(self.state(), k * DT, target, self.seed, prev=self.plan_ if (self.warm or self.canon) else None, x0k=x0k); p['k0'] = k; self.plan_ = p; self.costs.append(p['cost'])
    def ref(self, k):
        p = self.plan_; qr, qdr, qddr = spline_g(p['q0'], p['v0'], p['knots'], p['S'], np.array([(k - p['k0']) * DT])); return qr[0], qdr[0], qddr[0]
def new_env(seed, cond, rep):
    sc = scen(seed, cond, rep); env = sc.make_env(); obs = env.reset(seed); return sc, env, obs
def run_teacher(args):
    """args=(seed, rep, cond, every, warm, cfg, cal) -> metrics + plan costs. Teacher controls (CTC on the noisy obs, like T-oracle)"""
    seed, rep, cond, every, warm, cfg, cal = args; guard11([seed], cal); INST[0] = 0; sc, env, obs = new_env(seed, cond, rep); ctx = env.ctx; arm = ArmModel(ctx.robot); Kp, Kd = lqr_double_integrator(DEF['qp'], DEF['qv'])
    T = Teacher(env, seed, cond, rep, every, cfg=cfg, warm=warm); target = np.asarray(obs.ref['target'], float); tips = [env.task.chain_tip(env).copy()]; done = False; info = {}
    while not done:
        T.maybe_plan(obs.k, target); qr, qdr, qddr = T.ref(obs.k); tau = ctc(arm, Kp, Kd, ctx.tau_limit, obs.q, obs.qd, qr, qdr, qddr)
        obs, _, term, trunc, info = env.step(tau); tips.append(env.task.chain_tip(env).copy()); done = term or trunc
    dist = np.linalg.norm(np.array(tips) - env.task.target, axis=1)
    return dict(success=float(info['outcome'] == 'success'), min_dist=float(dist.min()), diverged=float(info.get('diverged', False)), inst=int(INST[0]), nplan=len(T.costs), plan_cost=float(np.sum(T.costs)), costs=T.costs, steps=env.k)
