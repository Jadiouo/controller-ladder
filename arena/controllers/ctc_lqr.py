"""L1+L2: computed torque (nominal inverse dynamics) with LQR gains.

tracking     : v = qdd_ref + Kp e + Kd ed, with [Kp, Kd] the LQR gain of the per-joint double integrator;
               tau = inverse_dynamics(q, qd, v). This is exactly the spike01 teacher.
pole_balance : outer LQR on [x - x_hold, xdot, theta, omega] for the linearised pole (input = tip x acceleration),
               z held by PD; inner operational-space computed torque
               qdd = J^+ (a - Jdot qd) (+ null-space posture PD for n>2); tau = inverse_dynamics(q, qd, qdd).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from ..robots import fk, jacobian, jdot_qd
from ..types import Obs
from ._arm import ArmModel, lqr_double_integrator
from .base import Controller, register_controller


class PoleLQR:
    """Tip-acceleration command (a_x, a_z) that balances the pole (shared by any model-based controller)."""

    def __init__(self, ctx, q_diag=(20.0, 2.0, 200.0, 20.0), r=0.01, kz=(400.0, 40.0)):
        L, g = ctx.task_info["pole_length"], ctx.task_info.get("gravity", 9.81)
        self.hold = np.asarray(ctx.task_info["hold"], float)
        A = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, g / L, 0.0]])
        B = np.array([[0], [1.0], [0], [-1.0 / L]])
        P = solve_continuous_are(A, B, np.diag(q_diag), np.array([[r]]))
        self.K = (B.T @ P / r).ravel()
        self.kz = kz
        self.spec = ctx.robot

    def tip_accel(self, obs: Obs):
        J = jacobian(self.spec, obs.q)
        x, v = fk(self.spec, obs.q), J @ obs.qd
        s = np.array([x[0] - self.hold[0], v[0], float(obs.extra["pole_theta"]), float(obs.extra["pole_omega"])])
        ax = -self.K @ s
        az = -self.kz[0] * (x[1] - self.hold[1]) - self.kz[1] * v[1]
        return np.array([ax, az]), J


@register_controller("ctc_lqr")
class CtcLqr(Controller):
    def __init__(self, ctx, qp=1.6e5, qv=800.0, pole_q=(20.0, 2.0, 200.0, 20.0), pole_r=0.01, null_kp=100.0, null_kd=20.0):
        super().__init__(ctx)
        if ctx.task not in ("tracking", "pole_balance"):
            raise NotImplementedError(f"ctc_lqr does not support task {ctx.task}")
        self.model = ArmModel(ctx.robot)
        self.Kp, self.Kd = lqr_double_integrator(qp, qv)
        self.null_kp, self.null_kd = null_kp, null_kd
        self.pole = PoleLQR(ctx, pole_q, pole_r) if ctx.task == "pole_balance" else None

    def act(self, obs: Obs):
        q, qd = obs.q, obs.qd
        if self.pole is None:
            v = obs.ref["qdd"] + self.Kp * (obs.ref["q"] - q) + self.Kd * (obs.ref["qd"] - qd)
        else:
            a, J = self.pole.tip_accel(obs)
            Jp = np.linalg.pinv(J)
            v = Jp @ (a - jdot_qd(self.ctx.robot, q, qd))
            if self.ctx.robot.n > 2:
                v = v + (np.eye(len(q)) - Jp @ J) @ (self.null_kp * (obs.ref["q"] - q) - self.null_kd * qd)
        return np.clip(self.model.inverse(q, qd, v), -self.ctx.tau_limit, self.ctx.tau_limit)
