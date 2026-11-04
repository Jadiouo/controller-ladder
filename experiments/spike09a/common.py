"""spike09a shared helpers: whip, CAL only (seeds >= 1000). Run from repo root.
Tunable teacher = WhipTeacher whose PLANNER assumes a chain (ls, ms, ds = scales of nominal length/mass/damping) and whose tracking gains are (qp, qv).
True chain per episode drawn from the seed (dist). Plans are cached on disk (keyed by seed + params) so evaluations are resumable."""
import sys, os, json, hashlib, dataclasses, numpy as np, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments', 'spike05'))
from arena import Scenario
from arena.tasks import make_task
from arena.robots import build_mjcf
from arena.perturb import ChainMass, ChainDamping, ChainLength
from arena.eval.rollout import run_episode
from arena.controllers import whip_teacher as WT
OUT = os.path.join(ROOT, 'results/spike09a'); PLANS = os.path.join(ROOT, '.scratch/s09a_plans')
DEF = dict(ls=1.0, ms=1.0, ds=1.0, qp=1.6e5, qv=800.0)
def guard(seeds): assert np.min(seeds) >= 1000, 'TEST seeds forbidden'
def draw_true(seed, w):
    """true chain error for an episode: length rel ~ U(-w,w); mass factor = exp(U(-5w,5w)); damping factor = exp(U(-10w,10w)) (w=0.1: len +-10%, mass x0.6..x1.65, damping x0.37..x2.7)."""
    r = np.random.default_rng([int(seed), 9109]); a, b, c = r.uniform(-1, 1, 3)
    return dict(len_rel=float(w * a), mass_rel=float(np.exp(5 * w * b) - 1), damp_fac=float(np.exp(10 * w * c)))
def scenario(seed, w):
    if w == 0: return Scenario('whip')
    t = draw_true(seed, w); return Scenario('whip').with_perturb(ChainLength(t['len_rel']), ChainMass(t['mass_rel']), ChainDamping(t['damp_fac'] - 1))
class Tuned(WT.WhipTeacher):
    """WhipTeacher planning on an ASSUMED chain (scaled nominal) with tracking gains qp, qv."""
    def __init__(self, ctx, P=None, **kw):
        P = {**DEF, **(P or {})}; self.P = P; info = dict(ctx.task_info)
        for k, s in (('chain_len', P['ls']), ('chain_mass', P['ms']), ('chain_damp', P['ds'])): info[k] = np.asarray(info[k], float) * s
        nom = ctx.make_model
        def mk(robot=ctx.robot, info=info):
            t = make_task('whip', robot, n_links=int(info['n_links']))
            t.chain_len, t.chain_mass, t.chain_damp = (np.array(info[k], float) for k in ('chain_len', 'chain_mass', 'chain_damp'))
            return mujoco.MjModel.from_xml_string(build_mjcf(robot, t.tip_xml(robot)))
        super().__init__(dataclasses.replace(ctx, task_info=info, make_model=mk), qp=P['qp'], qv=P['qv'], **kw)
    def _stage1(self, q0, target):
        """CMA-ES in the proxy on the ASSUMED chain -> list of distinct candidates (depends on ls, ms, ds only)."""
        if self.sim is None: self.sim = WT.SwingSim(self.ctx)
        sim, cfg, n = self.sim, self.cfg, self.ctx.robot.n; rng = np.random.default_rng([int(self.seed), 0x57A1]); archive = []
        for r in range(cfg['restarts']):
            x0 = np.tile(q0, 4) + (rng.normal(0, 0.5, 4 * n) if r else 0.0); es = WT.CMAES(x0, cfg['sigma0'], rng, cfg['lam'])
            for g in range(cfg['generations']):
                X = np.clip(es.ask(), WT.Q_LO, WT.Q_HI); Q = X.reshape(-1, 4, n); f = self._cost(sim.run(q0, Q, 1.8), Q, target); es.tell(f)
                archive += [(float(fi), x.copy()) for fi, x in zip(f, X)]
                if f.min() < 0.01: break
        archive.sort(key=lambda a: a[0]); picked = []
        for fv, x in archive:
            if all(np.linalg.norm(x - p) > 0.3 for p in picked): picked.append(x)
            if len(picked) >= cfg['n_rescore']: break
        return np.array(picked)
    def plan(self, q0, target):
        P = self.P; k1 = hashlib.md5(repr((int(self.seed), round(P['ls'], 5), round(P['ms'], 5), round(P['ds'], 5))).encode()).hexdigest()
        f1 = os.path.join(PLANS, f's1_{k1}.npy'); n = self.ctx.robot.n
        if os.path.exists(f1):
            try: picked = np.load(f1)
            except Exception: picked = None
        else: picked = None
        if picked is None:
            picked = self._stage1(q0, target); np.save(f1 + f'.{os.getpid()}.npy', picked); os.replace(f1 + f'.{os.getpid()}.npy', f1)
        scores = [self._closed_loop(q0, x.reshape(4, n), target) for x in picked]; j = int(np.argmin(scores))
        self.plan_info = dict(planned_min_dist=scores[j]); return picked[j].reshape(4, n)
    def act(self, obs):
        if self.knots is None:
            target = np.asarray(obs.ref['target'], float); q0 = np.asarray(obs.q, float); P = self.P
            key = (int(self.seed), tuple(round(P[k], 5) for k in ('ls', 'ms', 'ds')), round(P['qp'], 1), round(P['qv'], 2))
            if key not in WT._CACHE: WT._CACHE[key] = (self.plan(q0, target), dict(self.plan_info), q0)
            self.knots, self.plan_info, self.q0 = WT._CACHE[key]
        return super().act(obs)
def run_one(args):
    """one episode -> (success, min_dist, tip_speed). noise_sigma: iid gaussian torque noise (Nm) added to the commanded torque."""
    seed, w, P, noise = args; guard([seed]); sc = scenario(seed, w); sigma = noise or 0.0
    def fac(ctx):
        c = Tuned(ctx, {k: v for k, v in (P or {}).items() if k in DEF})
        if not sigma: return c
        rng = np.random.default_rng([int(seed), int((P or {}).get('_rep', 0)), 8081]); lim = ctx.tau_limit; act0 = c.act
        c.act = lambda obs: np.clip(act0(obs) + sigma * rng.standard_normal(len(obs.q)), -lim, lim); return c
    tr = run_episode(sc, fac, int(seed)); ex = sc.make_env().task.extra_metrics(tr)
    return float(ex['success']), float(ex['min_dist']), float(ex['hit_speed'])
def boot_ci(x, n=10000, seed=0, a=0.05):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = r.choice(len(x), (n, len(x))); b = x[m].mean(1); return float(np.quantile(b, a / 2)), float(np.quantile(b, 1 - a / 2))
def peval(pool, seeds, w, P, noise=None):
    """parallel episodes; returns arrays (succ, min_dist, hit_speed) in seed order. chunksize 1 for load balance (planning time varies)."""
    out = pool.map(run_one, [(int(s), w, P, noise) for s in seeds], chunksize=1); return np.array(out).T
def fit_of(succ, md): return float(succ.mean() - 0.2 * np.minimum(np.nan_to_num(md, nan=0.5), 0.5).mean())
