"""Classical teacher for `whip`: offline trajectory optimisation (CMA-ES) on the nominal model + computed-torque tracking.

Why this design (rather than MPPI / iLQR): the target is only reachable by a fling whose success is decided ~1 s after the arm has moved
(chain dynamics, 10 passive hinges, strongly nonlinear, under-actuated), so a receding-horizon quadratic-cost method with a short horizon
has no gradient to follow; the cost "closest approach of the chain tip over the whole swing" is non-smooth in the arm torques but smooth and
low-dimensional in a few joint-space keyframes, which is a good fit for a derivative-free search. So, per target:
  1. Parametrisation: joint trajectory = clamped cubic spline through the start posture and 4 free keyframes per joint (t = 0.4, 0.8, 1.2, 1.6 s),
     zero velocity at both ends (12 parameters).
  2. Search: (mu/mu_w, lambda)-CMA-ES (own implementation, seeded) on the NOMINAL MuJoCo model (arm + chain from ctx.make_model()); the arm is
     simulated as a stiff PD servo on the trajectory with gravity compensation and the real torque limit (a fast, batched proxy for computed
     torque: `mujoco.rollout`); cost = closest approach of the chain tip to the target over the swing (+ small keyframe-size regulariser).
  3. Re-scoring: the best distinct candidates are re-simulated in the exact closed loop that will run (computed torque, 200 Hz, the
     nominal model) and the best one is kept. The chain is never measured: execution is open loop for the chain.
  4. Execution: computed torque  tau = ID(q, qd, qdd_ref + Kp e + Kd ed)  with the arm-only nominal model (the chain and its reaction are an
     unmodelled disturbance that feedback has to absorb).
The plan depends only on (target, start posture, nominal parameters, seed) and is cached, so perturbation sweeps over the same seeds do not
re-plan (the perturbations never reach the planner). Success is therefore a property of how well an open-loop-planned fling survives a
different TRUE chain.
"""
from __future__ import annotations

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout
from scipy.interpolate import CubicSpline

from ..types import Obs
from ._arm import ArmModel, lqr_double_integrator
from .base import Controller, register_controller

KNOT_T = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
Q_LO, Q_HI = -2.8, 3.0
_CACHE: dict = {}


class CMAES:
    """Minimal (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial defaults), minimising f."""

    def __init__(self, x0, sigma, rng, lam=None):
        n = len(x0)
        self.n, self.rng, self.mean, self.sigma = n, rng, np.array(x0, float), float(sigma)
        self.lam = lam or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.gen = 0

    def ask(self):
        D2, B = np.linalg.eigh(self.C)
        self.B, self.D = B, np.sqrt(np.maximum(D2, 1e-20))
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.y = (self.z * self.D) @ B.T
        return self.mean + self.sigma * self.y

    def tell(self, f):
        idx = np.argsort(f)[: self.mu]
        yw = self.w @ self.y[idx]
        self.mean = self.mean + self.sigma * yw
        zw = self.w @ self.z[idx]
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ zw)
        self.gen += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        Y = self.y[idx]
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(min(self.sigma, 2.0))


class SwingSim:
    """Nominal arm + chain, batched, with the arm as a PD servo on a joint trajectory (the planning proxy)."""

    def __init__(self, ctx, dt_plan=0.0025, omega=60.0, zeta=1.0):
        info = ctx.task_info
        self.model = ctx.make_model()
        self.n = ctx.robot.n
        self.nc = int(info["n_links"])
        self.lim, self.omega = float(ctx.tau_limit), float(omega)
        inertia = ArmModel(ctx.robot).mass_diag(np.zeros(self.n))          # per-joint gains ~ inertia of the stretched arm (stable explicit PD)
        self.kp, self.kd = self.omega ** 2 * inertia, 2 * zeta * self.omega * inertia
        m = self.model
        m.opt.timestep = dt_plan
        self.dt = dt_plan
        for name in [f"l{i + 1}" for i in range(self.n)] + ["tip"]:
            m.body_gravcomp[m.body(name).id] = 1.0
        for i in range(self.n):
            m.actuator_ctrllimited[i] = 0
            m.actuator_gainprm[i, 0] = self.kp[i]
            m.actuator_biastype[i] = int(mujoco.mjtBias.mjBIAS_AFFINE)
            m.actuator_biasprm[i, :3] = [0.0, -self.kp[i], -self.kd[i]]
            m.actuator_forcelimited[i] = 1
            m.actuator_forcerange[i] = [-self.lim, self.lim]
        self.data = mujoco.MjData(m)
        self.lens = np.concatenate([np.asarray(ctx.robot.link_lengths, float), np.asarray(info["chain_len"], float)])
        self.nstate = int(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))

    def initial_state(self, q0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n] = q0
        self.data.qpos[self.n] = np.pi / 2 - np.sum(q0)
        mujoco.mj_forward(self.model, self.data)
        s = np.zeros(self.nstate)
        mujoco.mj_getState(self.model, self.data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return s

    def tip_xz(self, qpos):
        """Chain-tip position (..., 2) from the planar chain angles (arm + chain are one open chain of hinges about +y)."""
        phi = np.cumsum(qpos[..., :self.n + self.nc], axis=-1)
        return np.stack([(self.lens * np.cos(phi)).sum(-1), -(self.lens * np.sin(phi)).sum(-1)], -1)

    def run(self, q0, Q, T_end, nthread=4):
        """Q: (B, 4, n) free keyframes -> closest approach inputs: returns the chain-tip trajectories (B, T, 2) and joint trajectories."""
        nstep = int(round(T_end / self.dt))
        t = (np.arange(nstep) + 1) * self.dt
        s0 = self.initial_state(q0)
        ctrl = np.empty((len(Q), nstep, self.n))
        for b in range(len(Q)):
            qr, qdr, qddr = spline_traj(q0, Q[b], t)
            ctrl[b] = qr + (self.kd / self.kp) * qdr + qddr / self.omega ** 2     # PD + velocity and acceleration feed-forward
        out, _ = mj_rollout.rollout(self.model, [mujoco.MjData(self.model) for _ in range(max(1, min(nthread, len(Q))))],
                                    np.tile(s0, (len(Q), 1)), ctrl)
        bad = (np.diff(out[..., 0], axis=1) < 0).any(1)                  # MuJoCo auto-resets an unstable rollout: the clock jumps back
        return self.tip_xz(out[..., 1:1 + self.model.nq]), bad


def spline_traj(q0, Qk, t):
    """q, qd, qdd of the clamped cubic spline through (q0, Qk) at times t (held after the last keyframe)."""
    X = np.vstack([q0[None], Qk])
    cs = CubicSpline(KNOT_T, X, bc_type=((1, np.zeros(X.shape[1])), (1, np.zeros(X.shape[1]))))
    tc = np.minimum(t, KNOT_T[-1])
    late = (t > KNOT_T[-1])[:, None]
    return cs(tc), np.where(late, 0.0, cs(tc, 1)), np.where(late, 0.0, cs(tc, 2))


@register_controller("whip_teacher")
class WhipTeacher(Controller):
    def __init__(self, ctx, generations=40, lam=24, restarts=2, sigma0=0.6, n_rescore=8, qp=1.6e5, qv=800.0, reg=0.002):
        super().__init__(ctx)
        if ctx.task != "whip":
            raise NotImplementedError(f"whip_teacher does not support task {ctx.task}")
        self.model = ArmModel(ctx.robot)                                    # arm only; the chain is unmodelled for tracking
        self.Kp, self.Kd = lqr_double_integrator(qp, qv)
        self.cfg = dict(generations=generations, lam=lam, restarts=restarts, sigma0=sigma0, n_rescore=n_rescore, reg=reg)
        self.sim = None
        self.knots = None
        self.plan_info: dict = {}

    def reset(self, seed=None):
        super().reset(seed)
        self.knots = None

    # ---- planning -----------------------------------------------------------------------------------------------
    def _cost(self, sim_out, Q, target):
        tips, bad = sim_out
        d = np.linalg.norm(tips - target, axis=-1)
        d = np.where(np.isfinite(d), d, 10.0).min(1)
        return d + 10.0 * bad + self.cfg["reg"] * np.abs(Q).mean((1, 2))

    def _closed_loop(self, q0, Qk, target):
        """Exact closed loop that will be executed (nominal model, 200 Hz): returns the closest approach of the chain tip."""
        ctx, info = self.ctx, self.ctx.task_info
        m = ctx.make_model()
        d = mujoco.MjData(m)
        n = ctx.robot.n
        d.qpos[:n] = q0
        d.qpos[n] = np.pi / 2 - np.sum(q0)
        mujoco.mj_forward(m, d)
        sid = m.site("whip_tip").id
        arm = self.model
        dmin = np.inf
        steps = int(round(info["duration"] / ctx.dt))
        t = np.arange(steps) * ctx.dt
        qr, qdr, qddr = spline_traj(q0, Qk, t)
        sub = int(round(ctx.dt / m.opt.timestep))
        for k in range(steps):
            q, qd = d.qpos[:n].copy(), d.qvel[:n].copy()
            v = qddr[k] + self.Kp * (qr[k] - q) + self.Kd * (qdr[k] - qd)
            tau = np.clip(arm.inverse(q, qd, v), -ctx.tau_limit, ctx.tau_limit)
            d.ctrl[:n] = tau
            t_before = d.time
            for _ in range(sub):
                mujoco.mj_step(m, d)
            if d.time < t_before:                                          # unstable candidate: MuJoCo reset the state
                return float("inf")
            dmin = min(dmin, float(np.linalg.norm(d.site_xpos[sid][[0, 2]] - target)))
            if dmin < info["r_hit"] * 0.5:
                break
        return dmin

    def plan(self, q0, target):
        if self.sim is None:
            self.sim = SwingSim(self.ctx)
        sim, cfg, n = self.sim, self.cfg, self.ctx.robot.n
        rng = np.random.default_rng([0 if self.seed is None else int(self.seed), 0x57A1])
        archive = []
        for r in range(cfg["restarts"]):
            x0 = np.tile(q0, 4) + (rng.normal(0, 0.5, 4 * n) if r else 0.0)
            es = CMAES(x0, cfg["sigma0"], rng, cfg["lam"])
            for g in range(cfg["generations"]):
                X = np.clip(es.ask(), Q_LO, Q_HI)
                Q = X.reshape(-1, 4, n)
                f = self._cost(sim.run(q0, Q, 1.8), Q, target)
                es.tell(f)
                archive += [(float(fi), x.copy()) for fi, x in zip(f, X)]
                if f.min() < 0.01:
                    break
        archive.sort(key=lambda a: a[0])
        picked, best = [], None
        for fv, x in archive:                                              # distinct candidates for the exact re-scoring
            if all(np.linalg.norm(x - p) > 0.3 for p in picked):
                picked.append(x)
            if len(picked) >= cfg["n_rescore"]:
                break
        scores = [self._closed_loop(q0, x.reshape(4, n), target) for x in picked]
        j = int(np.argmin(scores))
        self.plan_info = dict(proxy_best=archive[0][0], rescored=scores, chosen=j, planned_min_dist=scores[j])
        return picked[j].reshape(4, n)

    def act(self, obs: Obs):
        if self.knots is None:
            target = np.asarray(obs.ref["target"], float)
            q0 = np.asarray(obs.q, float)
            key = (type(self).__name__, None if self.seed is None else int(self.seed), tuple(np.round(target, 6)), tuple(np.round(q0, 6)),
                   tuple(self.cfg.items()), tuple(np.round(self.ctx.task_info["chain_len"], 6)),
                   tuple(np.round(self.ctx.task_info["chain_mass"], 6)), tuple(np.round(self.ctx.task_info["chain_damp"], 8)))
            if key not in _CACHE:
                _CACHE[key] = (self.plan(q0, target), dict(self.plan_info), q0)
            self.knots, self.plan_info, self.q0 = _CACHE[key]
        t = np.array([obs.k * self.ctx.dt])
        qr, qdr, qddr = (a[0] for a in spline_traj(self.q0, self.knots, t))
        v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd)
        return np.clip(self.model.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)


@register_controller("whip_random")
class WhipRandom(WhipTeacher):
    """Baseline: same tracking controller, but the keyframes are one random draw around the start posture (no search)."""

    def plan(self, q0, target):
        rng = np.random.default_rng([0 if self.seed is None else int(self.seed), 0x57A2])
        self.plan_info = dict(planned_min_dist=float("nan"))
        return np.clip(np.tile(q0, (4, 1)) + rng.normal(0, 0.8, (4, len(q0))), Q_LO, Q_HI)
