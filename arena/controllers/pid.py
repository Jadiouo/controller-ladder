"""L0: joint-space PID with gravity compensation (nominal model).

tracking     : tau = g(q) + Kp (q_ref - q) + Kd (qd_ref - qd) + Ki * integral(q_ref - q)
pole_balance : joint PD velocity servo (optional weak position anchor at the hold posture, default off: the tip drifts) driven by a PID on the pole angle
               that outputs a tip velocity command, mapped to joints with the hold-point Jacobian:
               v_cmd = c_i * int(theta) + c_p * theta + c_d * omega ;  qd_target = J^+ [v_cmd, 0].
               (A pure position-target scheme cannot balance a pole: the closed loop has a negative constant
               Routh coefficient. The integral term is what makes a = K*theta at the tip.) No inverse dynamics.
"""
from __future__ import annotations

import numpy as np

from ..robots import jacobian
from ..types import Obs
from ._arm import ArmModel
from .base import Controller, register_controller


@register_controller("pid")
class PID(Controller):
    def __init__(self, ctx, kp=None, kd=None, wn=None, ki=0.0, i_clip=0.5, gravity=True, c_i=None, c_p=None, c_d=None, anchor=0.0):
        super().__init__(ctx)
        if ctx.task not in ("tracking", "pole_balance"):
            raise NotImplementedError(f"pid does not support task {ctx.task}")
        n = ctx.robot.n
        self.kp, self.kd = kp, kd           # None -> per-joint critically damped gains from nominal inertia (wn rad/s)
        self.wn = wn if wn is not None else (30.0 if ctx.task == "pole_balance" else 20.0)
        self.ki = np.broadcast_to(np.asarray(ki, float), (n,)).copy()
        self.i_clip, self.gravity = i_clip, gravity
        self.model = ArmModel(ctx.robot)
        if ctx.task == "pole_balance":     # defaults tuned on the calibration seeds (see tests/ + README)
            self.c_i, self.c_p, self.c_d, self.anchor = (70.0 if c_i is None else c_i), (5.0 if c_p is None else c_p), \
                (0.2 if c_d is None else c_d), anchor
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        self.z = np.zeros(self.ctx.robot.n)
        self.zth = 0.0
        self._kp = self._kd = None

    def _set_gains(self, q_nominal):
        n = self.ctx.robot.n
        m = self.model.mass_diag(q_nominal)
        self._kp = self.wn ** 2 * m if self.kp is None else np.broadcast_to(np.asarray(self.kp, float), (n,)).copy()
        self._kd = 2 * self.wn * m if self.kd is None else np.broadcast_to(np.asarray(self.kd, float), (n,)).copy()

    def act(self, obs: Obs):
        q, qd = obs.q, obs.qd
        if self._kp is None:
            self._set_gains(obs.ref["q"])
        qr, qdr = obs.ref["q"], obs.ref["qd"]
        e = qr - q
        if self.ctx.task == "pole_balance":
            th, om = float(obs.extra["pole_theta"]), float(obs.extra["pole_omega"])
            self.zth = float(np.clip(self.zth + th * self.ctx.dt, -0.2, 0.2))
            v = np.array([self.c_i * self.zth + self.c_p * th + self.c_d * om, 0.0])
            qdr = qdr + np.linalg.pinv(jacobian(self.ctx.robot, qr)) @ v
            e = self.anchor * e
        self.z = np.clip(self.z + e * self.ctx.dt, -self.i_clip, self.i_clip)
        tau = self._kp * e + self._kd * (qdr - qd) + self.ki * self.z
        if self.gravity:
            tau = tau + self.model.gravity(q)
        return np.clip(tau, -self.ctx.tau_limit, self.ctx.tau_limit)
