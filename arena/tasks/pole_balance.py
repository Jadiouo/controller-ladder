"""Pole balance: a passive-hinge inverted pendulum stands on the tool tip; keep it up and the tip near a hold point.

Pole = massless rod of length L with a point bob on a free hinge at the tip (so the plant is exactly the textbook
inverted pendulum driven by the tip acceleration: theta'' = (g sin(theta) - a_x cos(theta)) / L, theta > 0 leans to +x).
theta is the WORLD angle from vertical = sum(q) + hinge angle.
Obs.extra: pole_theta, pole_omega.  Obs.ref: q (hold posture), qd, qdd (zeros), ee (hold point).
Error = [theta] (rad) ONLY: one unit, so rmse / settle / recovery are well defined. The tip position error
[tip_x - x_hold, tip_z - z_hold] (m) is a separate log channel 'tip_err' with its own metrics tip_rmse / tip_settle_time.
The episode terminates (outcome 'failure') when |theta| > fall_angle; reaching the horizon is the 'timeout' outcome.
Primary metric: survival_time (s, higher is better). Extra metrics: fell (0/1), tip_rmse, tip_settle_time.
After a fall every remaining step is charged a FIXED penalty (protocol FAILURE_ERR rad on theta, FAIL_TIP_ERR m on the tip).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..robots import RobotSpec, fk, ik
from ..types import readonly
from .base import Task, register_task


FAIL_TIP_ERR = 0.5       # m per component, charged for every step after a fall (fixed penalty)
TIP_SETTLE_THR = 0.02    # m
FALL_REWARD = -10.0      # RL reward on the step that ends the episode with a fall


@register_task("pole_balance")
class PoleBalance(Task):
    name = "pole_balance"
    higher_is_better = frozenset({"survival_time"})
    binary_metrics = frozenset({"fell"})
    primary_metric = "survival_time"
    settle_thr = 0.02        # rad (theta is the only error component)

    def __init__(self, robot: RobotSpec | None = None, duration: float = 5.0, pole_length: float = 0.35,
                 bob_mass: float = 0.05, hold=(0.55, 0.10), fall_angle: float = 0.6,
                 theta0: float = 0.08, omega0: float = 0.2, **params):
        super().__init__(robot, **params)
        self.duration, self.L, self.bob_mass = duration, pole_length, bob_mass
        self.hold, self.fall_angle, self.theta0, self.omega0 = np.array(hold, float), fall_angle, theta0, omega0
        spec = self.robot
        q = np.array([0.0, 1.2] + [0.4] * (spec.n - 2))
        self.q_hold = ik(spec, self.hold, q, posture=q)
        self._ref = dict(q=self.q_hold, qd=np.zeros(spec.n), qdd=np.zeros(spec.n), ee=self.hold)

    @classmethod
    def default_robot(cls):
        return RobotSpec.two_link(payload_mass=0.5)

    def tip_xml(self, robot):
        return (f'<body name="pole" pos="0 0 0"><joint name="pole" type="hinge" axis="0 1 0" damping="0.001"/>'
                f'<geom type="capsule" fromto="0 0 0 0 0 {self.L!r}" size="0.004" mass="0.001" contype="0" conaffinity="0"/>'
                f'<geom type="sphere" pos="0 0 {self.L!r}" size="0.02" mass="{self.bob_mass!r}" contype="0" conaffinity="0"/>'
                f'</body>')

    def info(self):
        return dict(pole_length=self.L, bob_mass=self.bob_mass, hold=self.hold.copy(), gravity=9.81,
                    fall_angle=self.fall_angle, q_hold=self.q_hold.copy(), duration=self.duration,
                    hinge_damping=0.001, theta0=self.theta0, omega0=self.omega0)

    def reset(self, seed, env):
        n = env.n
        r = np.random.default_rng(int(seed))
        dq = r.uniform(-0.02, 0.02, n)
        th0, om0 = r.uniform(-self.theta0, self.theta0), r.uniform(-self.omega0, self.omega0)
        q0 = self.q_hold + dq
        env.data.qpos[:n] = q0
        env.data.qpos[n] = th0 - q0.sum()       # hinge angle relative to the last link
        env.data.qvel[n] = om0

    def reference(self, k):
        return readonly(self._ref)

    def ee_ref(self, k):
        return self.hold

    def _theta(self, env):
        n = env.n
        return float(env.data.qpos[:n + 1].sum()), float(env.data.qvel[:n + 1].sum())

    def measure(self, env):
        th, om = self._theta(env)
        return dict(pole_theta=np.array(th), pole_omega=np.array(om))

    def _tip(self, env):
        return fk(env.robot, env.data.qpos[:env.n])

    def error(self, env):
        return np.array([self._theta(env)[0]])

    def terminated(self, env):
        return abs(self._theta(env)[0]) > self.fall_angle

    def reward(self, env):
        th = self._theta(env)[0]
        if abs(th) > self.fall_angle:
            return FALL_REWARD
        return 1.0 - (th / self.fall_angle) ** 2 - 0.5 * float(np.sum((self._tip(env) - self.hold) ** 2))

    def log(self, env):
        th, om = self._theta(env)
        tip = self._tip(env)
        return dict(pole_theta=th, ee=tip, tip_err=tip - self.hold)

    def extra_metrics(self, trace):
        from ..eval.metrics import settle_time          # lazy: eval imports the tasks
        fell = trace.outcome == "failure"
        tip = trace.series_channel("tip_err", FAIL_TIP_ERR)
        out = dict(survival_time=float(trace.t_end) if fell else float(trace.n_steps * trace.dt), fell=float(fell))
        if tip is not None:
            out["tip_rmse"] = float(np.sqrt((tip ** 2).sum(1).mean()))
            out["tip_settle_time"] = settle_time(np.abs(tip).max(1), trace.dt, TIP_SETTLE_THR)
        return out
