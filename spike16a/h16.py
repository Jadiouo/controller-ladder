"""spike16a common code: seeds + guard, real-condition draws, extended classical teachers, the episode wrapper (residual + bias),
features, ES tuner, evaluation pool, bootstrap statistics. Written to results/spike16a/README.md section 0 BEFORE any run."""
import os, sys, json, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
import numpy as np
import multiprocessing as mp
from arena.scenario import Scenario
from arena.robots import fk
from arena.controllers.catch_teacher import CatchTeacher
from arena.controllers.toss_teacher import TossTeacher, solve_launch
from arena.controllers.pusht_teacher import PushTTeacher
from arena.controllers._pusht_plan import PushPlanner
from arena.controllers._arm import ArmModel
from arena.tasks import _ballistics as bal
from arena.robots import jacobian
from arena.perturb.ball_tasks import AirDrag, ObjectMass, ReleaseDelay, BallObsNoise, BallObsDelay
from arena.perturb.effects import Friction, SensorNoise, Delay
from arena.perturb.whip_pusht import TFriction, TMass, TComOffset, TPoseNoise

RES = os.path.join(ROOT, 'results', 'spike16a')
CAL_A = list(range(1000, 1050)); CAL_S = CAL_A + list(range(2000, 2050)); CAL = CAL_A + list(range(2000, 2150))
HOLD = list(range(1100, 1600)); NEW = list(range(705000, 705200)); GATE = list(range(706000, 706300))
LOCKED = [(0, 49), (20000, 20499), (30000, 30499), (40000, 40499)]
ALLOWED = [(1000, 1049), (2000, 2149), (1100, 1599), (705000, 705199), (706000, 706299), (510000, 10 ** 9)]
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
BOUND = 12.0
REP = dict(catch=4, toss=4, push_t=10)
GAMMA = dict(catch=0.99, toss=0.99, push_t=0.995)
TEACHER_NAME = dict(catch='catch_teacher', toss='toss_teacher', push_t='push_t_teacher')


def guard(seeds):
    for s in np.atleast_1d(seeds):
        s = int(s)
        if any(a <= s <= b for a, b in LOCKED):
            raise RuntimeError(f"seed {s} is LOCKED")
        if not any(a <= s <= b for a, b in ALLOWED):
            raise RuntimeError(f"seed {s} is outside CAL/HOLD/NEW/GATE/TRAIN")


# ---------------------------------------------------------------- real conditions
def draw(task, seed, s):
    r = np.random.default_rng([int(seed), 16016]); U = r.uniform; c = lambda v: max(float(v), -0.8)
    if task == 'catch':
        return dict(drag=c(s * U(-0.5, 2.0)), mass=c(s * U(-0.3, 1.5)), noise=float(s * U(0, 0.008)), delay=int(np.floor(s * U(0, 3.5))))
    if task == 'toss':
        return dict(drag=c(s * U(-0.5, 2.0)), mass=c(s * U(-0.3, 1.5)), rel_ms=float(s * U(0, 60)), fc=float(s * U(0, 4)), fv=float(s * U(0, 0.8)),
                    nq=float(s * U(0, 0.002)), nqd=float(s * U(0, 0.05)))
    if task == 'push_t':
        return dict(mu=c(s * U(-0.5, 1.0)), mass=c(s * U(-0.3, 1.0)), dx=float(s * U(-0.015, 0.015)), dz=float(s * U(-0.015, 0.015)),
                    npos=float(s * U(0, 0.004)), nang=float(s * U(0, 0.03)), delay=int(np.floor(s * U(0, 8))))
    raise ValueError(task)


def perts(task, d):
    if task == 'catch':
        return (AirDrag(d['drag']), ObjectMass(d['mass']), BallObsNoise(d['noise']), BallObsDelay(d['delay']))
    if task == 'toss':
        return (AirDrag(d['drag']), ObjectMass(d['mass']), ReleaseDelay(d['rel_ms']), Friction(d['fc'], d['fv']), SensorNoise(q=d['nq'], qd=d['nqd']))
    return (TFriction(d['mu']), TMass(d['mass']), TComOffset(d['dx'], d['dz']), TPoseNoise(d['npos'], d['nang']), Delay(steps=d['delay'], where='obs'))


# ---------------------------------------------------------------- extended classical teachers (arena code untouched)
class CatchT(CatchTeacher):
    def __init__(self, ctx, delay_hat=0.0, k_scale=1.0, **kw):
        super().__init__(ctx, **kw)
        self.k_drag = self.k_drag * k_scale
        self.delay_hat = delay_hat

    def act(self, obs):
        t, c = obs.t, self.cfg
        self.ts.append(t - self.delay_hat)
        self.ps.append(np.asarray(obs.extra["ball_pos"], float))
        if (t >= c["t_first"] and len(self.ts) >= 8 and not self.done_planning and t - self.last_plan_t >= c["replan_dt"] - 1e-9):
            if self.t_c is not None and self.t_pre - t < 0.06:
                self.done_planning = True
            else:
                self._replan(t)
        if self.plan is None:
            qr, qdr, qddr = self.q_home, np.zeros(3), np.zeros(3)
        else:
            i = int(np.clip(round((t - self.plan["t0"]) / self.dt), 0, len(self.plan["t"]) - 1))
            qr, qdr, qddr = self.plan["Q"][i], self.plan["Qd"][i], self.plan["Qdd"][i]
            if t > self.plan["t"][-1] + 1e-9:
                qdr, qddr = np.zeros(3), np.zeros(3)
        v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd)
        return np.clip(self.model.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)


class TossT(TossTeacher):
    def __init__(self, ctx, k_scale=1.0, t_shift=0.0, v_scale=1.0, **kw):
        super().__init__(ctx, **kw)
        self.k_drag = self.k_drag * k_scale
        self.t_r = self.t_r + t_shift
        self.v_scale = v_scale

    def _plan(self, q0, basket):
        xb, zb = basket
        z_rim = zb + self.h
        t_r, lim = self.t_r, self.ctx.tau_limit
        best = None
        for R in (0.5, 0.65, 0.8, 0.9):
            for alpha in np.deg2rad([-10, 10, 30, 50, 70]):
                p_r = R * np.array([np.cos(alpha), np.sin(alpha)])
                for theta in np.deg2rad([30, 40, 50, 60, 70]):
                    s = solve_launch(p_r, theta, xb, z_rim, self.k_drag, self.g)
                    if s is None:
                        continue
                    v_r = self.v_scale * s * np.array([np.cos(theta), np.sin(theta)])
                    for phi in (-0.9, -0.45, 0.0, 0.45, 0.9):
                        q_r = bal.level_ik(self.spec, p_r, phi)
                        if q_r is None:
                            continue
                        qd_r = np.linalg.pinv(jacobian(self.spec, q_r)) @ v_r
                        seg = bal.quintic(0.0, (q0, np.zeros(3), np.zeros(3)), t_r, (q_r, qd_r, np.zeros(3)))
                        Qs, Qds, Qdds = seg(np.linspace(0.0, t_r, 14))
                        tau_pk = max(np.abs(self.hold_model.inverse(q, qd, qdd)).max() for q, qd, qdd in zip(Qs, Qds, Qdds))
                        cost = tau_pk / lim + 0.01 * np.abs(Qds).max()
                        if best is None or cost < best[0]:
                            best = (cost, seg, q_r, qd_r, dict(p_r=p_r, theta=theta, speed=s, phi=phi, tau_ratio=tau_pk / lim))
        return best


class PushTT(PushTTeacher):
    def __init__(self, ctx, mu_scale=1.0, **kw):
        super().__init__(ctx, **kw)
        info = ctx.task_info
        self.planner = PushPlanner(info["mu"] * mu_scale, info["rho2"], self.r_p, self.goal, depth=4, beam=10)


TEACHERS = dict(catch=CatchT, toss=TossT, push_t=PushTT)
# (name, lo, hi, default, 'lin'|'log')
SPACE = dict(
    catch=[('fit_window', .10, .45, .25, 'lin'), ('t_first', .06, .25, .10, 'lin'), ('replan_dt', .01, .08, .03, 'lin'), ('tmin', .12, .35, .20, 'lin'),
           ('tau_s', .04, .15, .08, 'lin'), ('d_pre', 0., .06, .03, 'lin'), ('brake_T', .15, .5, .3, 'lin'), ('torque_frac', .5, 1., .85, 'lin'),
           ('qp', 3e4, 1e6, 1.6e5, 'log'), ('qv', 200., 3000., 800., 'log'), ('delay_hat', 0., .025, 0., 'lin'), ('k_scale', .5, 3., 1., 'lin')],
    toss=[('k_scale', .5, 3., 1., 'lin'), ('t_shift', -.02, .08, 0., 'lin'), ('v_scale', .9, 1.25, 1., 'lin'), ('brake_T', .3, 1., .6, 'lin'),
          ('qp', 3e4, 1e6, 1.6e5, 'log'), ('qv', 200., 3000., 800., 'log')],
    push_t=[('mu_scale', .4, 2., 1., 'lin'), ('v_push', .03, .15, .08, 'lin'), ('v_approach', .2, .6, .4, 'lin'), ('kp', 300., 2500., 900., 'log'),
            ('kd', 30., 120., 60., 'lin'), ('hold_s', .1, .6, .2, 'lin'), ('travel_w', .1, .8, .3, 'lin')])


def u2p(task, u):
    out = {}
    for ui, (n, lo, hi, d, sc) in zip(np.clip(u, 0, 1), SPACE[task]):
        out[n] = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo)))) if sc == 'log' else float(lo + ui * (hi - lo))
    return out


def p2u(task, p):
    return np.array([(np.log(p[n]) - np.log(lo)) / (np.log(hi) - np.log(lo)) if sc == 'log' else (p[n] - lo) / (hi - lo) for n, lo, hi, d, sc in SPACE[task]])


def default_params(task):
    return {n: float(d) for n, lo, hi, d, sc in SPACE[task]}


# ---------------------------------------------------------------- features
def rotm(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def feats(task, env, obs, tb, st):
    q, qd, lim, t = obs.q, obs.qd, env.tau_limit, obs.t
    base = [q, qd / 10.0, tb / lim]
    if task == 'catch':
        tip = fk(env.nominal, q)
        rel = np.asarray(obs.extra['ball_pos'], float) - tip
        st.setdefault('h', []).append(rel)
        h = st['h']; last = [h[max(0, len(h) - 1 - j)] for j in (3, 2, 1, 0)]
        v = (h[-1] - h[max(0, len(h) - 3)]) / (0.01) / 5.0 if len(h) > 2 else np.zeros(2)
        return np.concatenate(base + last + [v, [t / 2.0]]).astype(np.float32)
    if task == 'toss':
        b = np.asarray(obs.ref['basket'], float); ti = env.nominal_task.release_time
        return np.concatenate(base + [[t / 2.6, np.clip((ti - t) / 0.8, -1, 1)], b, np.asarray(obs.extra['ball_pos'], float) - b]).astype(np.float32)
    pose = np.asarray(obs.extra['t_pose'], float); goal = np.asarray(obs.ref['goal'], float)
    st.setdefault('h', []).append(pose); h = st['h']
    tip = fk(env.nominal, q)
    d10 = (h[-1] - h[max(0, len(h) - 11)]) * 10.0; d40 = (h[-1] - h[max(0, len(h) - 41)]) * 2.5
    loc = rotm(pose[2]).T @ (tip - pose[:2])
    dth = pose[2] - goal[2]
    return np.concatenate(base + [tip, pose[:2] - goal[:2], [np.cos(dth), np.sin(dth)], loc, d10, d40, [obs.k / 6000.0]]).astype(np.float32)


def feat_dim(task):
    return dict(catch=20, toss=15, push_t=21)[task]


# ---------------------------------------------------------------- episode wrapper
class Ep:
    """One episode of (task, real condition) with the classical teacher + optional constant bias + residual action (zero = classical)."""

    def __init__(self, task, seed, strength, params, bias=None, want_feats=True):
        guard([seed])
        self.task, self.seed = task, int(seed)
        self.cond = draw(task, seed, strength)
        sc = Scenario(task, perturbations=perts(task, self.cond))
        self.env = sc.make_env()
        self.teacher = TEACHERS[task](self.env.ctx, **params)
        self.bias = np.zeros(self.env.n) if bias is None else np.asarray(bias, float)
        self.rep, self.want = REP[task], want_feats
        self.st = {}
        obs = self.env.reset(self.seed); self.teacher.reset(self.seed)
        self.tb = self.teacher.act(obs); self.obs = obs
        self.done, self.outcome, self.rn = False, None, []
        self.mind, self.prev_z, self.land, self.maxcov = 9.0, None, None, 0.0
        t = self.env.task
        if task == 'toss':
            self.z_rim = float(t.basket[1] + t.basket_h)
        self.x = feats(task, self.env, obs, self.tb, self.st) if want_feats else None

    def _track(self):
        env, t = self.env, self.env.task
        if self.task == 'catch':
            tip = env.data.site_xpos[env.model.site("tip").id][[0, 2]]
            self.mind = min(self.mind, float(np.linalg.norm(t._ball(env)[0] - tip)))
        elif self.task == 'toss':
            p = t._obj(env)[0]
            if t._released and self.prev_z is not None and self.land is None and self.prev_z >= self.z_rim > p[1]:
                self.land = abs(float(p[0] - t.basket[0]))
            self.prev_z = float(p[1]); self.last_x = float(p[0])
        else:
            self.maxcov = max(self.maxcov, float(t.coverage_now(env)))

    def step(self, a):
        env, task = self.env, self.task
        res = BOUND * np.clip(np.asarray(a, float), -1, 1)
        r = 0.0; self.rn.append(float(np.linalg.norm(res)))
        info = {}
        for j in range(self.rep):
            tau = np.clip(self.tb + res + self.bias, -env.tau_limit, env.tau_limit)
            pc = self.maxcov
            obs, rew, term, trunc, info = env.step(tau)
            self._track()
            if task == 'push_t':
                r += 10.0 * max(0.0, self.maxcov - pc)
            else:
                r += rew * (0.1 if task == 'toss' else 1.0)
            if task == 'toss' and self.land is not None and not getattr(self, '_paid', False):
                self._paid = True
                if info['outcome'] != 'success':
                    r += -3.0 * min(self.land, 1.0)
            if term or trunc:
                self.done, self.outcome = True, info['outcome']
                if info['outcome'] == 'success':
                    r += 5.0 if task == 'push_t' else 0.0
                elif info['outcome'] == 'failure':
                    r -= 1.0
                break
            self.obs = obs
            self.tb = self.teacher.act(obs)
        if not self.done and self.want:
            self.x = feats(task, env, self.obs, self.tb, self.st)
        return self.x, r, self.done, dict(success=float(self.outcome == 'success'))

    def smooth(self):
        if self.task == 'catch':
            return -min(self.mind, 1.0)
        if self.task == 'toss':
            e = self.land if self.land is not None else abs(self.last_x - float(self.env.task.basket[0]))
            return -min(e, 1.0)
        return self.maxcov


def np_mlp(layers, x):
    for i, (W, b) in enumerate(layers):
        x = W @ x + b
        if i < len(layers) - 1:
            x = np.tanh(x)
    return x


def run_ep(args):
    """args: (task, seed, strength, params, layers|None, bias|None) -> dict. Errors are counted, never silent."""
    task, seed, s, params, layers, bias = args
    try:
        e = Ep(task, seed, s, params, bias, want_feats=layers is not None)
        a0 = np.zeros(e.env.n)
        while not e.done:
            a = a0 if layers is None else np.clip(np_mlp(layers, e.x.astype(np.float64)), -1, 1)
            e.step(a)
        return dict(success=float(e.outcome == 'success'), smooth=e.smooth(), res=float(np.mean(e.rn)), err=0)
    except Exception as ex:                                    # pragma: no cover
        import traceback; traceback.print_exc()
        return dict(success=0.0, smooth=0.0, res=0.0, err=1)


_POOL = None
def pool(n=6):
    global _POOL
    if _POOL is None:
        _POOL = mp.Pool(n)
    return _POOL


def evaluate(task, seeds, s, params, layers=None, bias=None, n=6):
    guard(seeds)
    r = pool(n).map(run_ep, [(task, int(sd), s, params, layers, None if bias is None else list(bias)) for sd in seeds], chunksize=1)
    return dict(succ=np.array([x['success'] for x in r]), smooth=np.array([x['smooth'] for x in r]), res=float(np.mean([x['res'] for x in r])), err=int(sum(x['err'] for x in r)))


def fitness(task, e):
    return float(e['succ'].mean() + 0.1 * e['smooth'].mean())


def bias_vec(task, b):
    return (np.array([1.0, -1.0, 1.0]) if task != 'push_t' else np.array([1.0, -1.0])) * float(b)


# ---------------------------------------------------------------- statistics
def boot_diff(d, B=10000, seed=0):
    r = np.random.default_rng(seed); n = len(d); m = d[r.integers(0, n, (B, n))].mean(1)
    return dict(diff=float(d.mean()), lo95=float(np.percentile(m, 2.5)), hi95=float(np.percentile(m, 97.5)), lo90=float(np.percentile(m, 5)),
                hi90=float(np.percentile(m, 95)), p_gt0=float(((m <= 0).sum() + 1) / (B + 1)))


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
