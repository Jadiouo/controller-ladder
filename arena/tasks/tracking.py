"""Tracking: the tool tip draws a figure in the workspace (Lissajous or a glyph); IK turns it into a joint reference.

ref kinds
  'lissajous'  : tip x/z sinusoids with random frequency ratio, amplitude, phase (default)
  'glyph'      : a pen stroke through the waypoints of S / 8 / M / Z / heart / circle, minimum-jerk timing
  'joint_sines': spike01-03 reference (two sines per joint, defined directly in joint space; exact legacy)
Observation ref keys: q, qd, qdd (joint space, n), ee (tip xz, 2).
Error = q - q_ref (rad); err2 = mean(error^2), i.e. the spike RMSE definition.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..robots import RobotSpec, fk, ik
from ..types import readonly
from .base import Task, register_task

FAIL_EE_ERR = 0.5      # m, tip distance charged for every step after a failure/divergence (fixed penalty)
_JR = np.array([[-1.2, 1.2], [0.3, 2.3]])      # spike joint-space reference range (rad)

GLYPHS = {   # unit-box waypoints (x right, z up), centred on 0
    "S": [(0.5, 0.8), (0, 1), (-0.5, 0.8), (-0.4, 0.3), (0.4, -0.3), (0.5, -0.8), (0, -1), (-0.5, -0.8)],
    "8": [(0, 0), (0.45, 0.5), (0, 1), (-0.45, 0.5), (0, 0), (0.5, -0.5), (0, -1), (-0.5, -0.5), (0, 0)],
    "M": [(-0.6, -1), (-0.6, 1), (0, 0.1), (0.6, 1), (0.6, -1)],
    "Z": [(-0.6, 1), (0.6, 1), (-0.6, -1), (0.6, -1)],
    "heart": [(0, -1), (-0.8, 0), (-0.5, 0.7), (0, 0.4), (0.5, 0.7), (0.8, 0), (0, -1)],
    "circle": [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 9)],
}


def _wrap(x):
    """wrap angles to (-pi, pi]"""
    return np.pi - np.mod(np.pi - np.asarray(x, float), 2 * np.pi)


def _track_ik(spec, target, prev_target, q, posture, max_jump=0.5, tol=1e-6):
    """IK step along the path. One long damped-least-squares solve near a singularity (straight arm) can leap to another
    configuration branch (seen: ref tip 0.9 m away from the figure). If the direct solve misses the target or moves a joint by more
    than `max_jump`, the target is approached in 8, then 64, then 512 equal sub-steps from the previous sample instead."""
    for n_sub in (1, 8, 64, 512):
        qq = np.array(q, float)
        for j in range(1, n_sub + 1):
            qq = ik(spec, prev_target + (target - prev_target) * j / n_sub, qq, posture=posture)
        if np.linalg.norm(fk(spec, qq) - target) < tol and np.abs(_wrap(qq - q)).max() < max_jump:
            break
    return qq


def _minjerk(s):
    """position, velocity, acceleration of the 5th-order smooth step on [0,1]"""
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4, 60 * s - 180 * s**2 + 120 * s**3


@register_task("tracking")
class Tracking(Task):
    name = "tracking"

    def __init__(self, robot: RobotSpec | None = None, ref: str = "lissajous", duration: float = 4.0,
                 center=(0.5, 0.0), size: float = 0.15, glyph: str | None = None, **params):
        super().__init__(robot, ref=ref, **params)
        if ref not in ("lissajous", "glyph", "joint_sines"):
            raise ValueError(ref)
        self.kind, self.duration, self.center, self.size, self.glyph = ref, duration, np.array(center, float), size, glyph
        self.n_steps = int(round(duration / self.robot.dt))
        self.Q = self.Qd = self.Qdd = self.EE = None

    # ---- reference construction ------------------------------------------------
    def _joint_sines(self, seed):
        n, dt = self.robot.n, self.robot.dt
        r = np.random.default_rng(int(seed))
        c, a, w, ph = np.zeros(n), np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
        for j in range(n):                                     # same draw order as spike/sim.py
            A, frac = r.uniform(0.2, 0.6), r.uniform(0.3, 0.7)
            lo, hi = _JR[j] if j < 2 else (-0.8, 0.8)
            a[j] = [A * frac, A * (1 - frac)]
            c[j] = r.uniform(lo + A, hi - A)
            w[j] = 2 * np.pi * r.uniform(0.2, 1.0, size=2)
            ph[j] = r.uniform(0, 2 * np.pi, size=2)
        t = np.arange(self.n_steps)[:, None] * dt              # t = k * DT_CTRL, as in spike
        arg = w[None] * t[:, :, None] + ph[None]
        Q = c + (a[None] * np.sin(arg)).sum(-1)
        Qd = (a[None] * w[None] * np.cos(arg)).sum(-1)
        Qdd = -(a[None] * w[None] ** 2 * np.sin(arg)).sum(-1)
        return Q, Qd, Qdd

    def _tip_path(self, seed, t):
        """tip xz at times t (may extend beyond [0, duration]: the figure continues smoothly there)"""
        r = np.random.default_rng(int(seed))
        if self.kind == "lissajous":
            ax, az = r.uniform(0.6, 1.0, 2) * self.size
            f0 = r.uniform(0.2, 0.45)
            ra, rb = [(1, 1), (1, 2), (2, 3), (3, 2)][r.integers(0, 4)]
            p1, p2 = r.uniform(0, 2 * np.pi, 2)
            return np.stack([self.center[0] + ax * np.sin(2 * np.pi * f0 * ra * t + p1),
                             self.center[1] + az * np.sin(2 * np.pi * f0 * rb * t + p2)], 1)
        name = self.glyph or sorted(GLYPHS)[r.integers(0, len(GLYPHS))]
        pts = np.array(GLYPHS[name], float) * self.size + self.center
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        spl = CubicSpline(s, pts, bc_type="natural")
        u, _, _ = _minjerk(np.clip(t / self.duration, 0, 1))
        return spl(u * s[-1])

    def _build_reference(self, seed):
        spec, dt = self.robot, self.robot.dt
        if self.kind == "joint_sines":
            Q, Qd, Qdd = self._joint_sines(seed)
        else:
            pad = 40                                             # extend the grid so spline end conditions do not bias t=0 / t=T
            tg = (np.arange(-pad, self.n_steps + pad)) * dt
            tip = self._tip_path(seed, tg)
            q = np.array([0.0, 1.2] + [0.4] * (spec.n - 2))      # elbow-positive warm start
            posture = q.copy()
            Qg = np.zeros((len(tg), spec.n))
            for k in range(len(tg)):
                q_new = _track_ik(spec, tip[k], tip[k - 1] if k else fk(spec, q), q, posture)
                # the IK is 2 pi periodic and its Newton steps can wander through many turns: wrap the start into (-pi, pi]
                # and afterwards pick, per joint, the 2 pi branch closest to the previous sample (unwrap)
                q = _wrap(q_new) if k == 0 else q + _wrap(q_new - q)
                Qg[k] = q
            sp = CubicSpline(tg, Qg)
            t = np.arange(self.n_steps) * dt
            Q, Qd, Qdd = sp(t), sp(t, 1), sp(t, 2)
        self.Q, self.Qd, self.Qdd = Q, Qd, Qdd
        self.EE = np.array([fk(spec, q) for q in Q])

    # ---- Task API ---------------------------------------------------------------
    def reset(self, seed, env):
        self._build_reference(seed)
        n = self.robot.n
        r = np.random.default_rng(int(seed) + 7_000_000)         # init error, same draws as spike
        ie = np.concatenate([r.uniform(-0.15, 0.15, n), r.uniform(-0.3, 0.3, n)])
        env.data.qpos[:n] = self.Q[0] + ie[:n]
        env.data.qvel[:n] = self.Qd[0] + ie[n:]

    def reference(self, k):
        k = min(k, self.n_steps - 1)
        return readonly(dict(q=self.Q[k], qd=self.Qd[k], qdd=self.Qdd[k], ee=self.EE[k]))

    def ee_ref(self, k):
        return self.EE[min(k, self.n_steps - 1)]

    def error(self, env):
        return env.data.qpos[:env.n] - self.Q[min(env.k, self.n_steps - 1)]

    def reward(self, env):
        return -float(np.mean(self.error(env) ** 2))

    def info(self):
        return dict(ref=self.kind, duration=self.duration, n_steps=self.n_steps, center=self.center.copy(), size=self.size)

    def log(self, env):
        import mujoco
        return dict(ee=env.data.site_xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tip")][[0, 2]].copy())

    def extra_metrics(self, trace):
        ee, ref = trace.channels.get("ee"), trace.ee_ref
        if ee is None or ref is None:
            return {}
        d = np.linalg.norm(ee - ref[:len(ee)], axis=1)
        if len(d) < trace.n_steps:                      # tracking never ends early except by simulator divergence
            d = np.concatenate([d, np.full(trace.n_steps - len(d), FAIL_EE_ERR)])
        return {"ee_rmse": float(np.sqrt((np.nan_to_num(d, nan=FAIL_EE_ERR) ** 2).mean()))}
