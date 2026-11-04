"""Toss: the arm holds an object at the tool tip and must throw it into a basket whose position is random (some baskets are
beyond the arm's reach: the object has to be thrown).

Hold: the object (sphere, two slide joints, no spin) is attached to the tip body by a `connect` equality constraint (tool 'claw'
is only the visual holder). The release is SCHEDULED by the task: the constraint is switched off at t_release (+ `release_delay`,
a true task parameter the controller does not know) so the arm cannot decide to release by itself; the controller has to
arrange position and velocity of the tip at t_release. Linear drag F = -c v (joint damping of the object) acts all the time.
World: basket = floor + two thin walls (inner half width `basket_hw`, wall height `basket_h`, floor top at z = zb), centre x = xb.
Seed -> basket xb ~ U(0.7, 1.8) m, zb ~ U(-0.5, -0.1) m (arm reach 1.0 m); initial posture = fixed wind-up posture + U(-0.02, 0.02) rad.
Obs.ref['basket'] = (xb, zb) (the goal, floor-top centre); Obs.extra['ball_pos'] = object xz (it equals the tip until released).
Success (0/1, primary): the object centre gets inside the basket volume (|x - xb| < hw, zb < z < zb + h) = it came in through
the opening; the episode ends then. Failure: after the release, the object centre falls below z = -0.9 m (floor) or leaves to x > 3 m.
Secondary: `landing_err` (m): |x - xb| where the object crosses the rim plane z = zb + h while falling after the release
(fallback if it never does: |x - xb| at the end), `energy` (mean sum tau^2 per executed step), `sat`.
True task parameters a perturbation may change: drag (N s/m), ball_mass (kg), release_delay (s).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..robots import RobotSpec
from .base import Task, register_task
from . import _ballistics as bal

FLOOR_Z = -0.9
Q_START = np.array([0.9, 0.7, 0.4])        # wind-up posture (tip about (0.28, -0.8)... see fk)


@register_task("toss")
class Toss(Task):
    name = "toss"
    duration = 2.6
    metrics = ("energy", "sat", "alive_frac")
    binary_metrics = frozenset({"success"})
    higher_is_better = frozenset({"success"})
    primary_metric = "success"

    def __init__(self, robot: RobotSpec | None = None, duration: float = 2.6, drag: float = 0.10, ball_mass: float = 0.15,
                 ball_radius: float = 0.04, release_time: float = 0.8, release_delay: float = 0.0, basket_hw: float = 0.11,
                 basket_h: float = 0.20, **params):
        super().__init__(robot, **params)
        self.duration, self.drag, self.ball_mass, self.ball_radius = duration, drag, ball_mass, ball_radius
        self.release_time, self.release_delay, self.basket_hw, self.basket_h = release_time, release_delay, basket_hw, basket_h
        self.g = bal.G
        self._nom = dict(drag=drag, ball_mass=ball_mass)
        self.L = float(sum(self.robot.link_lengths))
        self.q_start = Q_START.copy()
        self.basket = np.array([1.0, -0.3])
        self._released, self._last_k, self._inside = False, -1, False

    @classmethod
    def default_robot(cls):
        return RobotSpec.three_link(tool="claw", payload_mass=0.0, tool_mass=0.3)

    # ---- model ---------------------------------------------------------------------------
    def tip_xml(self, robot):
        xb, zb = float(self.basket[0]), float(self.basket[1])
        hw, h, w = self.basket_hw, self.basket_h, 0.016
        col = 'rgba="0.15 0.55 0.8 1"'
        return ("<!--@world-->"
                '<geom name="floor_vis" type="box" size="4 0.3 0.01" pos="1 0 -0.91" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>'
                '<geom name="pedestal" type="box" size="0.06 0.06 0.455" pos="0 0 -0.455" rgba="0.40 0.42 0.46 1" contype="0" conaffinity="0"/>'
                f'<geom name="basket_floor" type="box" size="{hw + w!r} 0.06 {w / 2!r}" pos="{xb!r} 0 {zb - w / 2!r}" {col} contype="1" conaffinity="1"/>'
                + "".join(f'<geom name="basket_wall{i}" type="box" size="{w / 2!r} 0.06 {h / 2!r}" pos="{xb + s * (hw + w / 2)!r} 0 {zb + h / 2!r}" '
                          f'{col} contype="1" conaffinity="1"/>' for i, s in enumerate((-1, 1)))
                + f'<body name="obj" pos="{self.L!r} 0 0">'
                f'<joint name="ox" type="slide" axis="1 0 0" damping="{self.drag!r}"/>'
                f'<joint name="oz" type="slide" axis="0 0 1" damping="{self.drag!r}"/>'
                f'<geom name="obj" type="sphere" size="{self.ball_radius!r}" mass="{self.ball_mass!r}" rgba="1 0.45 0.1 1" '
                f'contype="1" conaffinity="1" priority="1" solref="0.01 0.5" friction="0.6 0.005 0.0001"/></body>'
                '<!--@equality-->'
                '<connect name="hold" body1="tip" body2="obj" anchor="0 0 0" solref="0.005 1"/>')

    def info(self):
        return dict(gravity=self.g, drag=self._nom["drag"], ball_mass=self._nom["ball_mass"], ball_radius=self.ball_radius,
                    drag_rate=self._nom["drag"] / self._nom["ball_mass"], q_start=self.q_start.copy(), duration=self.duration,
                    release_time=self.release_time, basket_hw=self.basket_hw, basket_h=self.basket_h, arm_reach=self.L)

    # ---- episode -----------------------------------------------------------------------------
    def reset(self, seed, env):
        n = env.n
        r = np.random.default_rng([int(seed), 6160])
        self.basket = np.array([r.uniform(0.7, 1.8), r.uniform(-0.5, -0.1)])
        r2 = np.random.default_rng([int(seed), 6161])
        env.data.qpos[:n] = self.q_start + r2.uniform(-0.02, 0.02, n)
        mujoco.mj_forward(env.model, env.data)
        tip = env.data.site_xpos[env.model.site("tip").id]
        env.data.qpos[n:n + 2] = [tip[0] - self.L, tip[2]]
        self._released, self._last_k = False, -1
        env.data.eq_active[0] = 1
        # the basket geoms were built at construction from the (seed-independent) default basket: move them to this seed's basket
        self._place_basket(env)

    def _place_basket(self, env):
        xb, zb = self.basket
        hw, h, w = self.basket_hw, self.basket_h, 0.016
        m = env.model
        m.geom_pos[m.geom("basket_floor").id] = [xb, 0, zb - w / 2]
        for i, s in enumerate((-1, 1)):
            m.geom_pos[m.geom(f"basket_wall{i}").id] = [xb + s * (hw + w / 2), 0, zb + h / 2]

    def reference(self, k):
        return {"basket": self.basket.copy()}

    def ee_ref(self, k):
        return None

    def _obj(self, env):
        n = env.n
        return np.array([self.L + env.data.qpos[n], env.data.qpos[n + 1]]), env.data.qvel[n:n + 2].copy()

    def measure(self, env):
        return dict(ball_pos=self._obj(env)[0])

    # ---- events ---------------------------------------------------------------------------------
    def _advance(self, env):
        """Per-step state machine (release schedule); called once per env step through `terminated`, idempotent per step."""
        if env.k == self._last_k:
            return
        self._last_k = env.k
        if not self._released and env.k * env.dt >= self.release_time + self.release_delay - 1e-9:
            env.data.eq_active[0] = 0
            self._released = True

    def _inside_basket(self, env) -> bool:
        p, _ = self._obj(env)
        xb, zb = self.basket
        return bool(abs(p[0] - xb) < self.basket_hw and zb < p[1] < zb + self.basket_h)

    def terminated(self, env):
        self._advance(env)
        p, _ = self._obj(env)
        return bool(self._released and (p[1] < FLOOR_Z or p[0] > 3.0) and not self._inside_basket(env))

    def succeeded(self, env):
        return self._inside_basket(env)

    def reward(self, env):
        p, _ = self._obj(env)
        d = float(np.linalg.norm(p - self.basket))
        return -0.01 * min(d, 3.0) + (10.0 if self._inside_basket(env) else 0.0)

    # ---- logging / metrics -------------------------------------------------------------------------
    def log(self, env):
        p, v = self._obj(env)
        return dict(obj=p, obj_vel=v, released=float(self._released), basket=self.basket.copy())

    def extra_metrics(self, trace):
        out = dict(success=float(trace.outcome == "success"), landing_err=float("nan"))
        ch = trace.channels
        if "obj" not in ch or len(ch["obj"]) == 0:
            return out
        obj, rel = np.asarray(ch["obj"], float), np.asarray(ch["released"], float)
        if trace.final is not None:
            obj = np.vstack([obj, np.asarray(trace.final["channels"]["obj"], float)[None]])
            rel = np.append(rel, np.asarray(trace.final["channels"]["released"], float))
        xb, zb = np.asarray(ch["basket"][0], float)
        out["landing_err"] = float(abs(obj[-1, 0] - xb))                      # fallback: where it ended
        z_rim = zb + self.basket_h
        for i in range(len(obj) - 1):
            if rel[i] > 0.5 and obj[i, 1] >= z_rim > obj[i + 1, 1]:           # first downward crossing after the release
                f = (obj[i, 1] - z_rim) / (obj[i, 1] - obj[i + 1, 1])
                out["landing_err"] = float(abs(obj[i, 0] + f * (obj[i + 1, 0] - obj[i, 0]) - xb))
                break
        return out
