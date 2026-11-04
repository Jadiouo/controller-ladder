"""Seed -> episode data, in numpy float64, with the SAME draw order as arena (Tracking._joint_sines, Tracking.reset,
TorquePulses.reset). Everything an episode needs is a table indexed by control step, so the JAX side has no RNG
that could differ from the CPU version ("reset is decided by the seed")."""
from __future__ import annotations

import numpy as np

from arena.robots import RobotSpec

_JR = np.array([[-1.2, 1.2], [0.3, 2.3]])
DURATION = 4.0


def ref_params(seed: int, n: int = 2):
    r = np.random.default_rng(int(seed))
    c, a, w, ph = np.zeros(n), np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
    for j in range(n):
        A, frac = r.uniform(0.2, 0.6), r.uniform(0.3, 0.7)
        lo, hi = _JR[j]
        a[j] = [A * frac, A * (1 - frac)]
        c[j] = r.uniform(lo + A, hi - A)
        w[j] = 2 * np.pi * r.uniform(0.2, 1.0, size=2)
        ph[j] = r.uniform(0, 2 * np.pi, size=2)
    return c, a, w, ph


def ref_table(seed: int, spec: RobotSpec):
    """(Q, Qd, Qdd) each (n_steps, n): identical formulas to Tracking._joint_sines."""
    c, a, w, ph = ref_params(seed, spec.n)
    n_steps = int(round(DURATION / spec.dt))
    t = np.arange(n_steps)[:, None] * spec.dt
    arg = w[None] * t[:, :, None] + ph[None]
    Q = c + (a[None] * np.sin(arg)).sum(-1)
    Qd = (a[None] * w[None] * np.cos(arg)).sum(-1)
    Qdd = -(a[None] * w[None] ** 2 * np.sin(arg)).sum(-1)
    return Q, Qd, Qdd


def init_error(seed: int, n: int = 2):
    r = np.random.default_rng(int(seed) + 7_000_000)
    return np.concatenate([r.uniform(-0.15, 0.15, n), r.uniform(-0.3, 0.3, n)])


def pulse_table(seed: int, spec: RobotSpec, n: int = 3, width: float = 0.1, mag=(4.0, 8.0), margin: float = 0.3):
    dt, N, nj = spec.dt, int(round(DURATION / spec.dt)), spec.n
    out = np.zeros((N, nj))
    r = np.random.default_rng(int(seed) + 9_000_000)
    m = int(margin / dt)
    for _ in range(n):
        k0 = r.integers(m, N - m)
        j = r.integers(0, nj)
        out[k0:k0 + int(width / dt), j] += r.choice([-1, 1]) * r.uniform(*mag)
    return out


def episode_batch(seeds, spec: RobotSpec | None = None, pulses: bool | np.ndarray = False):
    """Stack tables for `seeds`. pulses: False / True / bool array (E,) choosing which episodes get pulses."""
    spec = spec or RobotSpec()
    seeds = np.asarray(seeds)
    E, N, n = len(seeds), int(round(DURATION / spec.dt)), spec.n
    Q, Qd, Qdd = np.zeros((E, N, n)), np.zeros((E, N, n)), np.zeros((E, N, n))
    ie = np.zeros((E, 2 * n))
    pt = np.zeros((E, N, n))
    on = np.broadcast_to(np.asarray(pulses, bool), (E,))
    for i, s in enumerate(seeds):
        Q[i], Qd[i], Qdd[i] = ref_table(s, spec)
        ie[i] = init_error(s, n)
        if on[i]:
            pt[i] = pulse_table(s, spec)
    return dict(q=Q, qd=Qd, qdd=Qdd, ie=ie, pulse=pt, seeds=seeds)
