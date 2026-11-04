"""Catch: a ball is thrown into the workspace from outside it; a 3-DoF vertical-plane arm with a shallow basket (tool 'net')
must catch it and keep it for >= 0.5 s.

World (x forward, z up, arm base at the origin): the ball is a point-like sphere with two slide joints (x, z; no spin) and a
linear air drag F = -c v implemented as joint damping, so c / m is the drag rate k (1/s). Launch (seed): from x0 ~ U(1.25, 1.55) m,
z0 ~ U(0.4, 0.8) m, with the velocity that would take the ball through a random aim point (x ~ U(0.5, 0.8), z ~ U(-0.15, 0.35))
after a flight time U(0.6, 0.9) s under the NOMINAL drag and mass (so a drag / mass error moves the real ball away from it).
The ball is visible from t = 0 (Obs.extra['ball_pos'], xz in m, no velocity: estimate it).
Obs.ref is empty (a catching task has no reference); tool: basket whose opening points along the tool's local +z (up when sum(q) = 0).

Primary metric `success` (0/1): the ball centre stays inside the basket volume for >= `dwell` = 0.5 s CONTINUOUSLY (the episode ends
then, outcome 'success'). The ball falling to the floor (z < -0.85 m) or leaving to x < -1 m ends the episode ('failure').
Secondary: `impact_speed` (m/s, |v_ball - v_tip| just before the first ball-basket contact; NaN if there was no contact: a miss),
`min_dist` (m, closest approach ball - tip), generic `energy` (mean sum tau^2 per executed step), `sat`, `alive_frac`.
True task parameters a perturbation may change (`modify_task`): drag (N s/m), ball_mass (kg), restitution (BallRestitution).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..robots import RobotSpec, jacobian
from .base import Task, register_task
from . import _ballistics as bal

FLOOR_Z = -0.85        # ball centre below this = on the floor = failure
HOME_TIP = (0.72, 0.45)


def restitution_to_dampratio(e: float) -> float:
    """MuJoCo contact damping ratio zeta that gives (approximately) the coefficient of restitution e of a spring-damper."""
    e = float(np.clip(e, 0.02, 0.98))
    ln = np.log(e)
    return float(-ln / np.sqrt(np.pi ** 2 + ln ** 2))


@register_task("catch")
class Catch(Task):
    name = "catch"
    duration = 2.0
    metrics = ("energy", "sat", "alive_frac")
    binary_metrics = frozenset({"success"})
    higher_is_better = frozenset({"success"})
    primary_metric = "success"

    def __init__(self, robot: RobotSpec | None = None, duration: float = 2.0, drag: float = 0.03, ball_mass: float = 0.10,
                 ball_radius: float = 0.03, restitution: float = 0.25, dwell: float = 0.5, **params):
        super().__init__(robot, **params)
        self.duration, self.drag, self.ball_mass, self.ball_radius = duration, drag, ball_mass, ball_radius
        self.restitution, self.dwell, self.g = restitution, dwell, bal.G
        self._nom = dict(drag=drag, ball_mass=ball_mass)     # launches are generated from the NOMINAL physics
        self.q_home = bal.level_ik(self.robot, HOME_TIP)
        self.launch = None
        self._dwell_n, self._last_k, self._ball_gid, self._net_gids = 0, -1, None, None

    @classmethod
    def default_robot(cls):
        return RobotSpec.three_link(tool="net", payload_mass=0.0, tool_mass=0.3)

    # ---- model -----------------------------------------------------------------------
    def tip_xml(self, robot):
        zeta = restitution_to_dampratio(self.restitution)
        return ("<!--@world-->"
                '<geom name="floor_vis" type="box" size="4 0.3 0.01" pos="1 0 -0.86" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>'
                '<geom name="pedestal" type="box" size="0.06 0.06 0.43" pos="0 0 -0.43" rgba="0.40 0.42 0.46 1" contype="0" conaffinity="0"/>'
                '<body name="ball" pos="0 0 0">'
                f'<joint name="bx" type="slide" axis="1 0 0" damping="{self.drag!r}"/>'
                f'<joint name="bz" type="slide" axis="0 0 1" damping="{self.drag!r}"/>'
                f'<geom name="ball" type="sphere" size="{self.ball_radius!r}" mass="{self.ball_mass!r}" rgba="1 0.45 0.1 1" '
                f'contype="1" conaffinity="1" priority="1" solref="0.01 {zeta!r}" friction="0.6 0.005 0.0001"/></body>')

    def info(self):
        return dict(gravity=self.g, drag=self._nom["drag"], ball_mass=self._nom["ball_mass"], ball_radius=self.ball_radius,
                    drag_rate=self._nom["drag"] / self._nom["ball_mass"], q_home=self.q_home.copy(), duration=self.duration,
                    dwell=self.dwell, basket_half_width=0.10, basket_wall_height=0.07, restitution=self.restitution)

    # ---- episode ---------------------------------------------------------------------------
    def reset(self, seed, env):
        n = env.n
        r = np.random.default_rng([int(seed), 5150])
        x0, z0 = r.uniform(1.25, 1.55), r.uniform(0.4, 0.8)
        aim = np.array([r.uniform(0.45, 0.75), r.uniform(0.10, 0.40)])
        T = r.uniform(0.55, 0.75)
        k_nom = self._nom["drag"] / self._nom["ball_mass"]
        v0 = bal.launch_velocity([x0, z0], aim, T, k_nom, self.g)
        self.launch = dict(p0=np.array([x0, z0]), v0=v0, aim=aim, T=T)
        r2 = np.random.default_rng([int(seed), 5151])
        env.data.qpos[:n] = self.q_home + r2.uniform(-0.02, 0.02, n)
        env.data.qpos[n:n + 2] = [x0, z0]
        env.data.qvel[n:n + 2] = v0
        self._dwell_n, self._last_k = 0, -1
        self._ball_gid = env.model.geom("ball").id
        self._net_gids = {env.model.geom(nm).id for nm in ("net_floor", "net_wall0", "net_wall1")}

    def reference(self, k):
        return {}

    def measure(self, env):
        return dict(ball_pos=env.data.qpos[env.n:env.n + 2].copy())

    # ---- events ------------------------------------------------------------------------------
    def _ball(self, env):
        n = env.n
        return env.data.qpos[n:n + 2].copy(), env.data.qvel[n:n + 2].copy()

    def _in_basket(self, env) -> bool:
        sid = env.model.site("tip").id
        R = env.data.site_xmat[sid].reshape(3, 3)
        p, _ = self._ball(env)
        loc = R.T @ (np.array([p[0], 0.0, p[1]]) - env.data.site_xpos[sid])
        return bool(abs(loc[0]) < 0.10 and 0.0 < loc[2] < 0.075)

    def _update(self, env):
        """Dwell counter; called once per env step (Task.terminated is the env's per-step hook), idempotent per step."""
        if env.k == self._last_k:
            return
        self._last_k = env.k
        self._dwell_n = self._dwell_n + 1 if self._in_basket(env) else 0

    def terminated(self, env):
        self._update(env)
        p, _ = self._ball(env)
        return bool(p[1] < FLOOR_Z or p[0] < -1.0) and self._dwell_n * env.dt < self.dwell

    def succeeded(self, env):
        self._update(env)
        return self._dwell_n * env.dt >= self.dwell

    def reward(self, env):
        p, _ = self._ball(env)
        tip = env.data.site_xpos[env.model.site("tip").id][[0, 2]]
        return -0.01 * min(float(np.linalg.norm(p - tip)), 2.0) + (5.0 if self._dwell_n * env.dt >= self.dwell else 0.0)

    # ---- logging / metrics ---------------------------------------------------------------------
    def _contact(self, env) -> float:
        d = env.data
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if (g1 == self._ball_gid and g2 in self._net_gids) or (g2 == self._ball_gid and g1 in self._net_gids):
                return 1.0
        return 0.0

    def log(self, env):
        p, v = self._ball(env)
        tip = env.data.site_xpos[env.model.site("tip").id][[0, 2]].copy()
        tv = jacobian(env.robot, env.data.qpos[:env.n]) @ env.data.qvel[:env.n]
        return dict(ball=p, ball_vel=v, tip=tip, tip_vel=tv, contact=self._contact(env))

    def extra_metrics(self, trace):
        out = dict(success=float(trace.outcome == "success"), impact_speed=float("nan"), min_dist=float("nan"))
        ch = trace.channels
        if "contact" not in ch or len(ch["contact"]) == 0:
            return out
        d = np.linalg.norm(ch["ball"] - ch["tip"], axis=1)
        out["min_dist"] = float(d.min())
        hit = np.where(ch["contact"] > 0.5)[0]
        if len(hit):
            k = max(int(hit[0]) - 1, 0)                 # state BEFORE the step in which the contact first appeared
            out["impact_speed"] = float(np.linalg.norm(ch["ball_vel"][k] - ch["tip_vel"][k]))
        return out
