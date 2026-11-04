"""Push-T: a 2-link arm moves a cylindrical pusher on a table and has to push a T-shaped block onto a fixed goal pose.

Plane. The arm lives in the x-z plane like every arena task. For this task that plane is the TABLE seen from above: gravity is switched
off in `reset` (the true model) and `task_info['gravity'] = 0` tells the controllers to do the same in their nominal models. The T block
slides on three planar joints (x, z, hinge about +y). The table friction is a MIX of viscous and dry friction: with F = mu m g (the
Coulomb force), a fraction `coulomb_frac` (0.5) is dry friction (joint frictionloss F on both slides, F * rho on the hinge) and the rest is
viscous, b = (1 - coulomb_frac) mu m g / v_ref (= the rest of F at the reference speed v_ref = 0.1 m/s) on both slides and b * rho^2 on the
hinge, rho^2 = I / m. The viscous part alone is an exact ellipsoidal limit surface (the model of quasi-static pushing theory, and what the
teacher assumes); the dry part is what the model does not capture (the block stops dead when the pusher stops; the table force depends on
mass and mu).
The pusher-block contact is MuJoCo's own friction cone (coefficient mu). Block geometry/proportions: `_pusht_geom` (Diffusion Policy T,
bar 0.20 x 0.05 m, stem 0.05 x 0.15 m, 1 px = 1.667 mm).

COM offset. The block's body frame sits at its centre of mass, the geometry is shifted by -com; so friction acts about the true COM. The
controller observes the pose of the SHAPE frame (the centroid of the outline, as a camera would see it) and does not know the offset.

Seed -> initial T pose = goal position + U(-0.10, 0.10)^2 and orientation = goal + U(-max_rot, max_rot) (rejected if coverage > 0.3 or the
outline leaves the work area), initial pusher at a random free point of the work area (>= 0.10 m from the outline).
Obs.ref['goal'] = goal pose (x, z, th); Obs.extra['t_pose'] = shape pose (x, z, th). The pusher tip comes from forward kinematics.
Success (0/1, primary) = coverage >= 0.9 (episode ends then). Failure = the T leaves the work area. Otherwise 'timeout' at `duration`.
Coverage = overlap area / goal area (Diffusion Policy definition), computed exactly by polygon clipping.
Secondary: `coverage` (at the end of the episode), `max_cov` (best during the episode), `time_to_success` (s, NaN without success),
`path_len` (m travelled by the pusher), `energy`, `sat`.
True task parameters a perturbation may change: mu (friction coefficient: pusher-block cone and dry table friction), table_b (viscous table
friction), t_mass (kg; also scales the dry table force), com (m, shape frame).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..robots import RobotSpec, fk, ik
from . import _pusht_geom as G
from .base import Task, register_task

GOAL = (0.55, 0.0, np.pi / 4)
X_RANGE, Z_RANGE = (0.20, 1.00), (-0.50, 0.50)      # work area: the T leaving it (centroid) is a failure
G_ACC = 9.81


def make_arm(robot=None):
    return robot or RobotSpec.two_link(payload_mass=0.3)


@register_task("push_t")
class PushT(Task):
    name = "push_t"
    duration = 30.0
    metrics = ("energy", "sat", "alive_frac")
    binary_metrics = frozenset({"success"})
    higher_is_better = frozenset({"success", "coverage", "max_cov"})
    primary_metric = "success"

    def __init__(self, robot: RobotSpec | None = None, duration: float = 30.0, mu: float = 0.5, t_mass: float = 0.4,
                 pusher_radius: float = 0.025, v_ref: float = 0.1, cov_thr: float = 0.9, max_rot: float = 3.14159, coulomb_frac: float = 0.5,
                 goal=GOAL, **params):
        super().__init__(robot, **params)
        self.duration, self.mu, self.t_mass, self.pusher_radius = duration, mu, t_mass, pusher_radius
        self.v_ref, self.cov_thr, self.max_rot, self.coulomb_frac = v_ref, cov_thr, max_rot, coulomb_frac
        self.goal = np.asarray(goal, float)
        self.table_b = (1.0 - coulomb_frac) * mu * t_mass * G_ACC / v_ref     # NOMINAL viscous table friction (kept when only the mass changes)
        self.com = np.zeros(2)
        self._nom = dict(mu=mu, t_mass=t_mass, table_b=self.table_b)
        self._cov_k, self._cov = -1, 0.0

    @classmethod
    def default_robot(cls):
        return make_arm()

    # ---- model ---------------------------------------------------------------------------------
    def tip_xml(self, robot):
        com, m, b = np.asarray(self.com, float), float(self.t_mass), float(self.table_b)
        gx, gz, gth = (float(v) for v in self.goal)
        deg = np.degrees(gth)
        pusher = (f'<geom name="pusher" type="cylinder" size="{self.pusher_radius!r} 0.03" euler="90 0 0" mass="0.05" '
                  f'rgba="0.9 0.35 0.2 1" contype="1" conaffinity="1" friction="0.001 0.005 0.0001" solref="0.005 1"/>')
        fl = float(self.coulomb_frac * self.mu * m * G_ACC)                   # dry friction force of the table
        parts = [f'<body name="tbody" gravcomp="1"><joint name="tx" type="slide" axis="1 0 0" damping="{b!r}" frictionloss="{fl!r}"/>'
                 f'<joint name="tz" type="slide" axis="0 0 1" damping="{b!r}" frictionloss="{fl!r}"/>'
                 f'<joint name="tth" type="hinge" axis="0 1 0" damping="{float(b * G.RHO2)!r}" frictionloss="{float(fl * np.sqrt(G.RHO2))!r}"/>'
                 f'<inertial pos="0 0 0" mass="{m!r}" diaginertia="{float(m * G.RHO2)!r} {float(m * G.RHO2)!r} {float(m * G.RHO2)!r}"/>']
        for name, (cu, cv, hu, hv) in zip(("tbar", "tstem"), G.RECTS):
            parts.append(f'<geom name="{name}" type="box" size="{hu!r} 0.03 {hv!r}" pos="{float(cu - com[0])!r} 0 {float(cv - com[1])!r}" '
                         f'rgba="0.25 0.45 0.85 1" contype="1" conaffinity="1" friction="{self.mu!r} 0.005 0.0001" solref="0.005 1"/>')
        parts.append("</body>")
        goal_vis = "".join(
            f'<geom name="goal_{name}" type="box" size="{hu!r} 0.002 {hv!r}" rgba="0.3 0.8 0.35 0.55" contype="0" conaffinity="0" '
            f'pos="{float(gx + np.cos(gth) * cu + np.sin(gth) * cv)!r} 0.036 {float(gz - np.sin(gth) * cu + np.cos(gth) * cv)!r}" euler="0 {float(deg)!r} 0"/>'
            for name, (cu, cv, hu, hv) in zip(("bar", "stem"), G.RECTS))
        return (pusher + "<!--@world-->"
                '<geom name="table" type="box" size="0.55 0.004 0.40" pos="0.62 0.045 0" rgba="0.82 0.78 0.68 1" contype="0" conaffinity="0"/>'
                + goal_vis + "".join(parts))

    def info(self):
        n = self._nom
        return dict(gravity=0.0, goal=self.goal.copy(), pusher_radius=self.pusher_radius, mu=n["mu"], t_mass=n["t_mass"],
                    table_b=n["table_b"], rho2=G.RHO2, v_ref=self.v_ref, cov_thr=self.cov_thr, duration=self.duration,
                    x_range=X_RANGE, z_range=Z_RANGE)

    # ---- episode -----------------------------------------------------------------------------------
    def _sample(self, seed):
        r = np.random.default_rng([int(seed), 7701])
        for _ in range(500):
            pose = np.array([self.goal[0] + r.uniform(-0.10, 0.10), self.goal[1] + r.uniform(-0.10, 0.10),
                             self.goal[2] + r.uniform(-self.max_rot, self.max_rot)])
            poly = G.poly_world(pose)
            if G.coverage(pose, self.goal) <= 0.3 and (poly[:, 0] > 0.34).all() and (poly[:, 0] < 0.86).all() and (np.abs(poly[:, 1]) < 0.26).all():
                break
        for _ in range(500):
            p = np.array([r.uniform(0.40, 0.85), r.uniform(-0.25, 0.25)])
            _, d = G.closest_on_polygon(p[None], poly)
            if d[0] > 0.10 and not G.inside_polygon((p[None] - pose[:2]) @ G.rot(pose[2]), G.POLY)[0]:
                break
        return pose, p

    def reset(self, seed, env):
        n = env.n
        env.model.opt.gravity[:] = 0.0
        pose, pusher = self._sample(seed)
        self.t0_pose, self.p0 = pose, pusher
        env.data.qpos[:n] = ik(self.robot, pusher, np.array([0.4, 1.4]))
        env.data.qpos[n:n + 2] = pose[:2] + G.rot(pose[2]) @ np.asarray(self.com, float)    # body frame = COM frame
        env.data.qpos[n + 2] = pose[2]
        self._cov_k = -1
        mujoco.mj_forward(env.model, env.data)

    def reference(self, k):
        return {"goal": self.goal.copy()}

    def ee_ref(self, k):
        return None

    def shape_pose(self, env):
        n = env.n
        th = float(env.data.qpos[n + 2])
        xz = env.data.qpos[n:n + 2] - G.rot(th) @ np.asarray(self.com, float)
        return np.array([xz[0], xz[1], th])

    def measure(self, env):
        return dict(t_pose=self.shape_pose(env))

    def tip_pos(self, env):
        return fk(env.robot, env.data.qpos[:env.n])

    # ---- events ---------------------------------------------------------------------------------------
    def coverage_now(self, env) -> float:
        if env.k != self._cov_k:
            self._cov_k, self._cov = env.k, G.coverage(self.shape_pose(env), self.goal)
        return self._cov

    def terminated(self, env):
        p = self.shape_pose(env)
        out = not (X_RANGE[0] < p[0] < X_RANGE[1] and Z_RANGE[0] < p[1] < Z_RANGE[1])
        return bool(out and self.coverage_now(env) < self.cov_thr)

    def succeeded(self, env):
        return self.coverage_now(env) >= self.cov_thr

    def reward(self, env):
        return float(self.coverage_now(env)) - (1.0 if self.terminated(env) else 0.0) + (5.0 if self.succeeded(env) else 0.0)

    # ---- logging / metrics ---------------------------------------------------------------------------------
    def log(self, env):
        return dict(t_pose=self.shape_pose(env), tip=self.tip_pos(env), cov=self.coverage_now(env))

    def extra_metrics(self, trace):
        out = dict(success=float(trace.outcome == "success"), coverage=float("nan"), max_cov=float("nan"),
                   time_to_success=float("nan"), path_len=float("nan"))
        ch = trace.channels
        if "cov" not in ch or len(ch["cov"]) == 0:
            return out
        cov = np.asarray(ch["cov"], float)
        tip = np.asarray(ch["tip"], float)
        if trace.final is not None:
            cov = np.append(cov, float(np.asarray(trace.final["channels"]["cov"])))
            tip = np.vstack([tip, np.asarray(trace.final["channels"]["tip"], float)[None]])
        out["coverage"], out["max_cov"] = float(cov[-1]), float(cov.max())
        out["path_len"] = float(np.linalg.norm(np.diff(tip, axis=0), axis=1).sum())
        if trace.outcome == "success":
            out["time_to_success"] = float(trace.t_end)
        return out
