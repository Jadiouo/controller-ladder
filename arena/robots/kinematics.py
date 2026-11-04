"""Analytic planar kinematics. Convention matches MuJoCo hinge axis +y: a positive joint angle rotates +x toward -z.

Link i points along angle phi_i = sum_{j<=i} q_j; tip = sum_i L_i (cos phi_i, -sin phi_i) in (x, z).
"""
from __future__ import annotations

import numpy as np

from .spec import RobotSpec


def _phi(q):
    return np.cumsum(q)


def fk(spec: RobotSpec, q) -> np.ndarray:
    phi, L = _phi(np.asarray(q, float)), np.asarray(spec.link_lengths)
    return np.array([(L * np.cos(phi)).sum(), -(L * np.sin(phi)).sum()])


def jacobian(spec: RobotSpec, q) -> np.ndarray:
    """(2, n) tip-position Jacobian."""
    phi, L = _phi(np.asarray(q, float)), np.asarray(spec.link_lengths)
    dx, dz = -L * np.sin(phi), -L * np.cos(phi)       # d tip / d phi_i
    # q_j moves phi_i for all i >= j
    return np.array([np.cumsum(dx[::-1])[::-1], np.cumsum(dz[::-1])[::-1]])


def jdot_qd(spec: RobotSpec, q, qd) -> np.ndarray:
    """Jdot @ qd: the tip acceleration when joint accelerations are zero."""
    phi, L = _phi(np.asarray(q, float)), np.asarray(spec.link_lengths)
    w = _phi(np.asarray(qd, float))
    return np.array([-(L * np.cos(phi) * w ** 2).sum(), (L * np.sin(phi) * w ** 2).sum()])


def ik(spec: RobotSpec, target, q0, iters: int = 200, damping: float = 1e-6, tol: float = 1e-12,
       posture=None, posture_gain: float = 0.05) -> np.ndarray:
    """Damped-least-squares IK from the warm start q0 (keeps the elbow branch of q0). For n>2 an optional
    `posture` pulls the null space toward a preferred configuration."""
    q = np.array(q0, float)
    target = np.asarray(target, float)
    for it in range(iters):
        e = target - fk(spec, q)
        J = jacobian(spec, q)
        JJt = J @ J.T + damping * np.eye(2)
        dq = J.T @ np.linalg.solve(JJt, e)
        if it == 0 and posture is not None and spec.n > 2:   # one gentle null-space step per call keeps the path continuous
            Jp = np.linalg.pinv(J) @ J
            dq += (np.eye(spec.n) - Jp) @ (posture_gain * (np.asarray(posture) - q))
        q += dq
        if np.linalg.norm(e) < tol and np.linalg.norm(dq) < 1e-10:
            break
    return q


def reachable(spec: RobotSpec, target, margin: float = 0.02) -> bool:
    r = float(np.hypot(*target))
    L = list(spec.link_lengths)
    rmax = sum(L)
    rmin = max(0.0, 2 * max(L) - rmax)
    return rmin + margin <= r <= rmax - margin
