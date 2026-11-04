"""Data types shared by every layer (kept dependency-free)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np


def readonly(d: dict) -> dict:
    """Copy of a dict of arrays with every array copied and frozen (a controller must not edit the task's reference)."""
    out = {}
    for k, v in d.items():
        a = np.array(v, copy=True)
        a.flags.writeable = False
        out[k] = a
    return out


@dataclass
class Obs:
    """What a controller sees at control step k.

    q, qd       : measured joint position / velocity (after sensor perturbations)
    ref         : task reference at this step, dict of READ-ONLY arrays. Keys are task-defined, but by convention
                  'q', 'qd', 'qdd' (joint space) and 'ee' (tip xz position) when meaningful.
    extra       : other measured task quantities (e.g. 'pole_theta', 'pole_omega')
    prev_action : the torque command the controller returned at the previous step (zeros at reset)
    history     : (H, d) array of the previous H observations as `to_vector()` rows, oldest first (H = Scenario.history,
                  default 0 -> shape (0, d)). At reset every row is the first observation.
    ref_preview : `obs.ref_preview(H)` -> dict of (H, ...) arrays, the reference at steps k..k+H-1 (clamped at the end).
    """
    t: float
    k: int
    q: np.ndarray
    qd: np.ndarray
    ref: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    prev_action: np.ndarray | None = None
    history: np.ndarray | None = None
    ref_fn: Callable | None = field(default=None, repr=False, compare=False)

    def replace(self, **kw) -> "Obs":
        return replace(self, **kw)

    def to_vector(self) -> np.ndarray:
        """Flat float32 vector [q, qd, *ref (sorted keys), *extra (sorted keys)] for learned policies.
        (history / prev_action are NOT included; concatenate them yourself if the policy wants them.)"""
        parts = [self.q, self.qd]
        parts += [np.ravel(self.ref[k]) for k in sorted(self.ref)]
        parts += [np.ravel(self.extra[k]) for k in sorted(self.extra)]
        return np.concatenate(parts).astype(np.float32)

    def ref_preview(self, horizon: int) -> dict:
        """Reference at steps k, k+1, ..., k+horizon-1 as {key: (horizon, ...) array}. For iLQR / MPC."""
        if self.ref_fn is None:
            raise RuntimeError("this Obs has no reference function (not produced by ArenaEnv)")
        rows = [self.ref_fn(self.k + j) for j in range(int(horizon))]
        return readonly({key: np.stack([r[key] for r in rows]) for key in rows[0]})


@dataclass(frozen=True)
class ControlContext:
    """Nominal knowledge handed to a controller factory (the controller never sees the true plant).

    robot      : NOMINAL RobotSpec (perturbations such as payload error change only the true plant)
    task       : task registry name
    dt         : control period (s)
    tau_limit  : nominal actuator limit the controller may assume (Nm)
    task_info  : task-provided nominal constants (e.g. pole length, hold point, fall angle, duration)
    n_steps    : episode horizon in control steps
    rng        : numpy Generator re-seeded from the EPISODE seed at every env.reset (use it for any randomness in
                 act(); `Controller.rng` is a second stream, re-seeded by Controller.reset(seed))
    ref_fn     : ref_fn(k) -> reference dict at step k (clamped to the horizon); for MPC / iLQR
    make_model : make_model() -> a fresh NOMINAL mujoco.MjModel (arm + tool + task objects such as the pole)
    """
    robot: object
    task: str
    dt: float
    tau_limit: float
    task_info: dict = field(default_factory=dict)
    n_steps: int = 0
    rng: np.random.Generator | None = None
    ref_fn: Callable | None = None
    make_model: Callable | None = None
