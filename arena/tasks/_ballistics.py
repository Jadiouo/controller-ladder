"""Projectile with linear drag in the x-z plane: a = -k v + g, g = (0, -G), k = c / m (1/s).
Closed form with h(t) = (1 - exp(-k t)) / k:   p(t) = p0 + g t / k + (v0 - g / k) h(t),   v(t) = g / k + (v0 - g / k) exp(-k t)."""
from __future__ import annotations

import numpy as np

G = 9.81


def _gvec(g):
    return np.array([0.0, -g])


def h_of(t, k):
    t = np.asarray(t, float)
    return t if k < 1e-9 else (1.0 - np.exp(-k * t)) / k


def position(t, p0, v0, k, g=G):
    t = np.asarray(t, float)[..., None]
    gv = _gvec(g)
    if k < 1e-9:
        return np.asarray(p0) + np.asarray(v0) * t + 0.5 * gv * t ** 2
    return np.asarray(p0) + gv * t / k + (np.asarray(v0) - gv / k) * h_of(t, k)


def velocity(t, v0, k, g=G):
    t = np.asarray(t, float)[..., None]
    gv = _gvec(g)
    if k < 1e-9:
        return np.asarray(v0) + gv * t
    return gv / k + (np.asarray(v0) - gv / k) * np.exp(-k * t)


def launch_velocity(p0, target, T, k, g=G):
    """v0 such that the projectile launched from p0 is at `target` after T seconds."""
    gv = _gvec(g)
    p0, target = np.asarray(p0, float), np.asarray(target, float)
    if k < 1e-9:
        return (target - p0 - 0.5 * gv * T ** 2) / T
    return gv / k + (target - p0 - gv * T / k) / h_of(T, k)


def fit(ts, ps, k, g=G):
    """Least-squares (p0 at t=0, v0) of the drag model from timed position samples ts (N,), ps (N, 2)."""
    ts = np.asarray(ts, float)
    gv = _gvec(g)
    h = h_of(ts, k)
    A = np.stack([np.ones_like(ts), h], 1)
    base = (ts[:, None] * gv / k if k >= 1e-9 else 0.5 * gv * ts[:, None] ** 2)
    if k < 1e-9:
        A = np.stack([np.ones_like(ts), ts], 1)
    sol, *_ = np.linalg.lstsq(A, np.asarray(ps, float) - base, rcond=None)     # rows: [a; w], a = p0, w = v0 - g/k
    p0 = sol[0]
    v0 = sol[1] if k < 1e-9 else sol[1] + gv / k
    return p0, v0


# ---- planar 3-link arm with a prescribed tool orientation (used by both ball tasks) -----------------------------
def level_ik(spec, tip, phi=0.0, elbow_up=True):
    """Analytic IK of an n=3 arm for tip xz position and total angle sum(q) = phi. Returns q (3,) or None if unreachable.
    elbow_up=True is the q2 > 0 branch (the default posture of the other tasks)."""
    L1, L2, L3 = spec.link_lengths
    wx = tip[0] - L3 * np.cos(phi)
    wz = tip[1] + L3 * np.sin(phi)
    r2 = wx * wx + wz * wz
    D = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(D) > 1.0:
        return None
    q2 = np.arccos(D) * (1 if elbow_up else -1)
    th2 = -q2
    th1 = np.arctan2(wz, wx) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
    q1 = -th1
    return np.array([q1, q2, phi - q1 - q2])


def quintic(t0, s0, t1, s1):
    """Quintic polynomial between states s=(q, qd, qdd) (each (n,)) at t0 and t1. Returns f(t) -> (q, qd, qdd); t may be a
    scalar or an array (outputs then have a leading axis). Clamped to [t0, t1]; after t1 the end state is extrapolated at
    constant velocity (zero acceleration)."""
    T = t1 - t0
    q0, v0, a0 = (np.asarray(x, float) for x in s0)
    q1, v1, a1 = (np.asarray(x, float) for x in s1)
    c0, c1, c2 = q0, v0, a0 / 2
    M = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    rhs = np.stack([q1 - c0 - c1 * T - c2 * T ** 2, v1 - c1 - 2 * c2 * T, a1 - 2 * c2])
    c3, c4, c5 = np.linalg.solve(M, rhs)

    def f(t):
        t = np.asarray(t, float)
        s = np.clip(t - t0, 0.0, T)[..., None]
        q = c0 + c1 * s + c2 * s ** 2 + c3 * s ** 3 + c4 * s ** 4 + c5 * s ** 5
        v = c1 + 2 * c2 * s + 3 * c3 * s ** 2 + 4 * c4 * s ** 3 + 5 * c5 * s ** 4
        a = 2 * c2 + 6 * c3 * s + 12 * c4 * s ** 2 + 20 * c5 * s ** 3
        late = (t > t1)[..., None]
        q = np.where(late, q + v * (t - t1)[..., None], q)
        a = np.where(late, 0.0, a)
        return q, v, a
    return f
