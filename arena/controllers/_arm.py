"""Nominal arm model shared by model-based controllers (built from the NOMINAL RobotSpec only)."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are

from ..robots import build_mjcf


class ArmModel:
    def __init__(self, spec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(spec))
        self.d = mujoco.MjData(self.model)

    def inverse(self, q, qd, qdd) -> np.ndarray:
        """Torque that produces acceleration qdd at (q, qd) under the nominal model (gravity, damping included)."""
        d = self.d
        d.qpos[:] = q
        d.qvel[:] = qd
        d.qacc[:] = qdd
        d.qfrc_applied[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse.copy()

    def mass_diag(self, q) -> np.ndarray:
        d = self.d
        d.qpos[:] = q
        mujoco.mj_forward(self.model, d)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, d, M)
        return np.diag(M).copy()

    def gravity(self, q) -> np.ndarray:
        return self.inverse(q, np.zeros_like(q), np.zeros_like(q))


def lqr_double_integrator(qp: float = 1.6e5, qv: float = 800.0, r: float = 1.0):
    """LQR gain [Kp, Kd] for the per-joint error dynamics e'' = u (spike01 teacher default)."""
    A = np.array([[0, 1], [0, 0.0]])
    B = np.array([[0], [1.0]])
    P = solve_continuous_are(A, B, np.diag([qp, qv]), np.array([[r]]))
    return (B.T @ P).ravel() / r
