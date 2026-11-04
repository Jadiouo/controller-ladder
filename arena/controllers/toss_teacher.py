"""Classical teacher for `toss`: ballistic inversion -> release state -> joint trajectory -> computed torque.

Model knowledge (NOMINAL only): gravity, drag rate k = c / m, object mass, the scheduled release time t_r, the basket position
(Obs.ref['basket'], read at the first step), the arm model.
  1. Inversion: for a grid of release points p_r (arc around the base) and launch angles theta, the launch speed that makes the
     nominal drag-ballistic flight cross the basket's rim plane at the basket centre is found by root finding.
  2. Release posture: level IK is not required here; the tip velocity is mapped to joints with the minimum-norm pseudo-inverse of the
     tip Jacobian, the release posture sum(q) = phi is searched over a small grid.
  3. Trajectory: quintic in joint space from the start posture (rest) to (q_r, qd_r) at t_r; the candidate with the smallest
     peak feed-forward torque (nominal inverse dynamics incl. the carried object, relative to the torque limit) is used.
     After t_r a braking quintic stops the arm.
  4. Control: computed torque (LQR gains of the tracking teacher), nominal arm + object mass before t_r, bare arm after.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..robots import jacobian
from ..tasks import _ballistics as bal
from ..types import Obs
from ._arm import ArmModel, lqr_double_integrator
from .base import Controller, register_controller

T_GRID = np.arange(0.0, 3.0, 0.002)


def _cross_x(p0, v0, k, g, z_rim):
    """x where the flight from p0 with v0 crosses z = z_rim while falling (nan if it never does)."""
    P = bal.position(T_GRID, p0, v0, k, g)
    z = P[:, 1] - z_rim
    idx = np.where((z[:-1] >= 0) & (z[1:] < 0))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]
    f = z[i] / (z[i] - z[i + 1])
    return P[i, 0] + f * (P[i + 1, 0] - P[i, 0])


def solve_launch(p0, theta, target_x, z_rim, k, g, s_lo=0.5, s_hi=14.0):
    """Launch speed s (m/s) at angle theta (rad above +x) so that the drag-ballistic flight from p0 crosses the rim plane at
    target_x; None if there is none in [s_lo, s_hi]."""
    d = np.array([np.cos(theta), np.sin(theta)])

    def f(s):
        x = _cross_x(p0, s * d, k, g, z_rim)
        return (x - target_x) if np.isfinite(x) else -10.0
    try:
        if f(s_lo) * f(s_hi) > 0:
            return None
        return brentq(f, s_lo, s_hi, xtol=1e-4)
    except ValueError:
        return None


@register_controller("toss_teacher")
class TossTeacher(Controller):
    def __init__(self, ctx, qp=1.6e5, qv=800.0, brake_T=0.6, torque_frac=0.8):
        super().__init__(ctx)
        if ctx.task != "toss":
            raise NotImplementedError(f"toss_teacher does not support task {ctx.task}")
        ti = ctx.task_info
        self.k_drag, self.g, self.t_r = ti["drag_rate"], ti["gravity"], ti["release_time"]
        self.q_start, self.h = np.asarray(ti["q_start"], float), ti["basket_h"]
        self.spec = ctx.robot
        self.hold_model = ArmModel(ctx.robot.with_(payload_mass=ctx.robot.payload_mass + ti["ball_mass"]))
        self.free_model = ArmModel(ctx.robot)
        self.Kp, self.Kd = lqr_double_integrator(qp, qv)
        self.brake_T, self.torque_frac = brake_T, torque_frac
        self.reset(None)

    def reset(self, seed=None):
        super().reset(seed)
        self.seg1 = self.seg2 = None
        self.info = {}

    # ---- planning ----------------------------------------------------------------------------
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
                    v_r = s * np.array([np.cos(theta), np.sin(theta)])
                    for phi in (-0.9, -0.45, 0.0, 0.45, 0.9):
                        q_r = bal.level_ik(self.spec, p_r, phi)
                        if q_r is None:
                            continue
                        qd_r = np.linalg.pinv(jacobian(self.spec, q_r)) @ v_r
                        seg = bal.quintic(0.0, (q0, np.zeros(3), np.zeros(3)), t_r, (q_r, qd_r, np.zeros(3)))
                        Qs, Qds, Qdds = seg(np.linspace(0.0, t_r, 14))
                        tau_pk = max(np.abs(self.hold_model.inverse(q, qd, qdd)).max() for q, qd, qdd in zip(Qs, Qds, Qdds))
                        qd_pk = np.abs(Qds).max()
                        cost = tau_pk / lim + 0.01 * qd_pk
                        if best is None or cost < best[0]:
                            best = (cost, seg, q_r, qd_r, dict(p_r=p_r, theta=theta, speed=s, phi=phi, tau_ratio=tau_pk / lim))
        return best

    def _start(self, obs: Obs):
        best = self._plan(obs.q, obs.ref["basket"])
        if best is None:                      # no feasible inversion: hold the start posture
            q_hold = obs.q.copy()
            self.seg1 = lambda t: (q_hold, np.zeros(3), np.zeros(3))
            self.seg2 = None
            return
        _, self.seg1, q_r, qd_r, self.info = best
        stop = q_r + 0.4 * qd_r * self.brake_T
        self.seg2 = bal.quintic(self.t_r, (q_r, qd_r, np.zeros(3)), self.t_r + self.brake_T, (stop, np.zeros(3), np.zeros(3)))

    # ---- control -----------------------------------------------------------------------------
    def act(self, obs: Obs):
        if self.seg1 is None:
            self._start(obs)
        t = obs.t
        if t <= self.t_r + 1e-9 or self.seg2 is None:
            qr, qdr, qddr = self.seg1(min(t, self.t_r))
        else:
            qr, qdr, qddr = self.seg2(t)
            if t > self.t_r + self.brake_T:
                qdr, qddr = np.zeros(3), np.zeros(3)
        model = self.hold_model if t < self.t_r else self.free_model
        v = qddr + self.Kp * (qr - obs.q) + self.Kd * (qdr - obs.qd)
        return np.clip(model.inverse(obs.q, obs.qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
