"""Classical teacher for `catch`: ballistic prediction + intercept planning + IK + computed torque + soft (velocity-matched) contact.

Model knowledge (NOMINAL only, from ctx.task_info): gravity, drag rate k = c / m, ball radius, home posture, arm model.
  1. Estimate: least-squares fit of the nominal drag-ballistic model to the last `fit_window` s of measured ball positions
     (timestamps = control time; a delayed measurement is therefore misread as a ball that is behind).
  2. Intercept: the earliest time t_c (>= tmin ahead) at which the predicted ball is inside the basket's reachable set
     (level basket: sum(q) = 0, analytic IK) AND the whole motion is feasible under the nominal torque limit (feed-forward
     inverse dynamics of the nominal arm). Fixed once chosen; the target at t_c is re-predicted every `replan_dt`.
  3. Path (tip space, C2): quintic reach to a pre-contact state `tau_s` before t_c with the ball's velocity and acceleration,
     a ride-along phase with the basket floor under the ball whose gap shrinks to 0 at t_c (velocity matched = soft contact),
     then a braking quintic that stops the basket (the ball is pressed into it). IK + spline give q, qd, qdd references.
  4. Control: computed torque with the LQR gains of the tracking teacher; the plan always continues from the previous
     plan's reference state, not from the (noisy) measurement.
"""
from __future__ import annotations

import numpy as np
from ..robots import fk
from ..tasks import _ballistics as bal
from ..types import Obs
from ._arm import ArmModel, lqr_double_integrator
from .base import Controller, register_controller

R_MIN, R_MAX = 0.15, 0.70          # allowed wrist radius at the planned contact (arm max = L1 + L2 = 0.75)


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


@register_controller("catch_teacher")
class CatchTeacher(Controller):
    def __init__(self, ctx, qp=1.6e5, qv=800.0, fit_window=0.25, t_first=0.10, replan_dt=0.03, tmin=0.20, tau_s=0.08,
                 d_pre=0.03, brake_T=0.3, torque_frac=0.85, qd_max=25.0):
        super().__init__(ctx)
        if ctx.task != "catch":
            raise NotImplementedError(f"catch_teacher does not support task {ctx.task}")
        ti = ctx.task_info
        self.k_drag, self.g, self.r_ball = ti["drag_rate"], ti["gravity"], ti["ball_radius"]
        self.q_home = np.asarray(ti["q_home"], float)
        self.spec, self.dt = ctx.robot, ctx.dt
        self.model = ArmModel(ctx.robot)
        self.Kp, self.Kd = lqr_double_integrator(qp, qv)
        self.cfg = dict(fit_window=fit_window, t_first=t_first, replan_dt=replan_dt, tmin=tmin, tau_s=tau_s, d_pre=d_pre,
                        brake_T=brake_T, torque_frac=torque_frac, qd_max=qd_max)
        self.reset(None)

    def reset(self, seed=None):
        super().reset(seed)
        self.ts, self.ps = [], []
        self.t_c = None
        self.t_pre = None
        self.plan = None            # dict(t0, Q, Qd, Qdd, P, V, A) on the control grid
        self.last_plan_t = -1.0
        self.done_planning = False
        self.info = {}

    # ---- planning ------------------------------------------------------------------------
    def _ball_fn(self):
        w = self.cfg["fit_window"]
        ts, ps = np.array(self.ts), np.array(self.ps)
        sel = ts >= ts[-1] - w
        p0, v0 = bal.fit(ts[sel], ps[sel], self.k_drag, self.g)
        k, g = self.k_drag, self.g
        return (lambda t: bal.position(t, p0, v0, k, g)), (lambda t: bal.velocity(t, v0, k, g)), \
               (lambda t: -k * bal.velocity(t, v0, k, g) + np.array([0.0, -g]))

    def _ik_proj(self, p):
        """level IK of tip points p (n, 2); targets whose wrist lies outside the reachable annulus are projected onto it.
        Returns (Q (n, 3), projection distance (n,))."""
        L1, L2, L3 = self.spec.link_lengths
        wx, wz = p[:, 0] - L3, p[:, 1]
        r = np.hypot(wx, wz)
        rc = np.clip(r, 0.08, 0.745)
        wx, wz = wx * rc / np.maximum(r, 1e-9), wz * rc / np.maximum(r, 1e-9)
        D = np.clip((rc ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
        q2 = np.arccos(D)
        th1 = np.arctan2(wz, wx) - np.arctan2(-L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = -th1
        return np.stack([q1, q2, -q1 - q2], 1), np.abs(r - rc)

    def _build(self, t0, ps, vs, as_, tc, ball):
        """Tip path (P, V, A analytic) on the control grid from t0 (start state ps, vs, as_) through the intercept at tc,
        and the joint references: level IK for q, J3 qd = (V, 0), J3 qdd = (A - Jdot qd, 0). dt = control period."""
        c, dt, rb = self.cfg, self.dt, self.r_ball
        pos, vel, acc = ball
        t_pre = tc - c["tau_s"]
        n = int(np.ceil((tc + c["brake_T"] - t0) / dt)) + 1
        t = t0 + np.arange(n) * dt
        up = np.array([0.0, 1.0])
        pre = pos(t_pre) - up * (rb + c["d_pre"])
        reach = bal.quintic(t0, (ps, vs, as_), t_pre, (pre, vel(t_pre), acc(t_pre)))
        p_c = pos(tc) - up * rb
        stop = p_c
        for f in (0.4, 0.3, 0.2, 0.12, 0.06, 0.0):           # braking distance: as long as the stop point stays reachable
            stop = p_c + f * vel(tc) * c["brake_T"]
            if self._wrist_r(stop) <= 0.70:
                break
        brake = bal.quintic(tc, (p_c, vel(tc), acc(tc)), tc + c["brake_T"], (stop, np.zeros(2), np.zeros(2)))
        P, V, A = (np.zeros((n, 2)) for _ in range(3))
        m1_ = t < t_pre
        m2_ = (t >= t_pre) & (t <= tc)
        m3_ = t > tc
        if m1_.any():
            P[m1_], V[m1_], A[m1_] = reach(t[m1_])
        if m2_.any():
            s_ = (t[m2_] - t_pre) / c["tau_s"]
            m, m1, m2 = _minjerk(s_), 30 * s_ ** 2 - 60 * s_ ** 3 + 30 * s_ ** 4, 60 * s_ - 180 * s_ ** 2 + 120 * s_ ** 3
            d, d1, d2 = c["d_pre"] * (1 - m), -c["d_pre"] * m1 / c["tau_s"], -c["d_pre"] * m2 / c["tau_s"] ** 2
            P[m2_] = pos(t[m2_]) - up * (rb + d)[:, None]
            V[m2_] = vel(t[m2_]) - up * d1[:, None]
            A[m2_] = acc(t[m2_]) - up * d2[:, None]
        if m3_.any():
            P[m3_], V[m3_], A[m3_] = brake(t[m3_])
        Q, dproj = self._ik_proj(P)
        phi = np.cumsum(Q, 1)
        Lk = np.asarray(self.spec.link_lengths)
        dx, dz = -Lk * np.sin(phi), -Lk * np.cos(phi)
        J = np.stack([np.cumsum(dx[:, ::-1], 1)[:, ::-1], np.cumsum(dz[:, ::-1], 1)[:, ::-1]], 1)     # (n, 2, 3)
        J3 = np.concatenate([J, np.ones((n, 1, 3))], 1)
        Jinv = np.swapaxes(J3, 1, 2) @ np.linalg.inv(J3 @ np.swapaxes(J3, 1, 2) + 1e-6 * np.eye(3))
        Qd = np.einsum("nij,nj->ni", Jinv, np.concatenate([V, np.zeros((n, 1))], 1))
        w = np.cumsum(Qd, 1)
        jd = np.stack([-(Lk * np.cos(phi) * w ** 2).sum(1), (Lk * np.sin(phi) * w ** 2).sum(1), np.zeros(n)], 1)
        Qdd = np.einsum("nij,nj->ni", Jinv, np.concatenate([A, np.zeros((n, 1))], 1) - jd)
        return dict(t0=t0, t=t, Q=Q, Qd=Qd, Qdd=Qdd, P=P, V=V, A=A, proj=float(dproj.max()), tc=tc)

    def _cost(self, plan, stride=2):
        lim = self.ctx.tau_limit
        taus = [np.abs(self.model.inverse(plan["Q"][i], plan["Qd"][i], plan["Qdd"][i])).max() / lim
                for i in range(0, len(plan["t"]), stride)]
        plan["parts"] = (round(float(max(taus)), 2), round(float(np.abs(plan["Qd"]).max() / self.cfg["qd_max"]), 2),
                         round(float(plan["proj"]), 3))
        return max(max(taus), np.abs(plan["Qd"]).max() / self.cfg["qd_max"]) + 5.0 * plan["proj"]

    def _wrist_r(self, p):
        return float(np.hypot(p[0] - self.spec.link_lengths[2], p[1]))

    def _contact_ok(self, pos, tc):
        up = np.array([0.0, self.r_ball])
        return all(R_MIN <= self._wrist_r(pos(t) - up) <= R_MAX for t in (tc, tc - self.cfg["tau_s"]))

    def _start_state(self, t):
        if self.plan is None:
            return fk(self.spec, self.q_home), np.zeros(2), np.zeros(2)
        i = int(np.clip(round((t - self.plan["t0"]) / self.dt), 0, len(self.plan["t"]) - 1))
        return self.plan["P"][i], self.plan["V"][i], self.plan["A"][i]

    def _replan(self, t):
        c = self.cfg
        ball = self._ball_fn()
        ps, vs, as_ = self._start_state(t)
        pos = ball[0]
        if self.t_c is None:
            best = None
            for tc in np.arange(t + c["tmin"], t + 1.1, 0.01):
                if not self._contact_ok(pos, tc):
                    continue
                plan = self._build(t, ps, vs, as_, tc, ball)
                cost = self._cost(plan)
                if best is None or cost < best[0]:
                    best = (cost, plan)
                if cost <= c["torque_frac"]:
                    best = (cost, plan)
                    break
            if best is None:                  # nothing reachable in the predicted trajectory: keep the home pose
                return
            self.plan, self.t_c = best[1], best[1]["tc"]
            self.info["cost"] = best[0]
            self.info["parts"] = best[1]["parts"]
        else:
            self.plan = self._build(t, ps, vs, as_, self.t_c, ball)
        self.t_pre = self.t_c - c["tau_s"]
        self.last_plan_t = t

    # ---- control -------------------------------------------------------------------------------
    def act(self, obs: Obs):
        t, c = obs.t, self.cfg
        self.ts.append(t)
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
