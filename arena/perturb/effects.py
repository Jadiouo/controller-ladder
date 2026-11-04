"""Concrete weaknesses / disturbances. All are deterministic given the episode seed."""
from __future__ import annotations

import numpy as np

from .base import Perturbation


class PayloadError(Perturbation):
    """True tip payload differs from the nominal one the controller assumes.
    PayloadError(rel=0.3): true = nominal * 1.3.  PayloadError(mass=1.5): true = 1.5 kg exactly."""

    def __init__(self, rel: float | None = None, mass: float | None = None):
        if (rel is None) == (mass is None):
            raise ValueError("give exactly one of rel / mass")
        self.rel, self.mass = rel, mass

    def modify_robot(self, robot):
        m = self.mass if self.mass is not None else robot.payload_mass * (1 + self.rel)
        return robot.with_(payload_mass=m)

    def label(self):
        return f"payload{self.rel:+.0%}" if self.rel is not None else f"payload={self.mass}kg"


class TorqueLimit(Perturbation):
    """True actuators saturate at `cap` Nm (controller still assumes the nominal limit)."""

    def __init__(self, cap: float):
        self.cap = float(cap)
        self.torque_limit = self.cap

    def on_command(self, tau, k):
        return np.clip(tau, -self.cap, self.cap)

    def label(self):
        return f"cap{self.cap:g}Nm"


class Delay(Perturbation):
    """Latency of `steps` control periods (or `ms` milliseconds, converted with the ACTUAL control period of the env at
    reset). `where`: 'command' (actuator latency, default), 'obs' (sensor latency: q, qd and task measurements
    are `steps` old; the reference is not delayed) or 'both'. Until the buffer fills the first value is held
    (same convention as spike02)."""

    def __init__(self, steps: int | None = None, ms: float | None = None, where: str = "command"):
        if (steps is None) == (ms is None):
            raise ValueError("give exactly one of steps / ms")
        if where not in ("command", "obs", "both"):
            raise ValueError(where)
        self.ms, self.where = ms, where
        self.steps = None if steps is None else int(steps)
        self._hist: list = []
        self._ohist: list = []

    @classmethod
    def from_ms(cls, ms: float, dt: float | None = None, where: str = "command"):
        """dt given -> steps resolved now; dt None -> resolved at env.reset with the env's real control period."""
        return cls(steps=int(round(ms / 1000.0 / dt)), where=where) if dt else cls(ms=ms, where=where)

    def reset(self, seed, env):
        self._hist, self._ohist = [], []
        if self.steps is None or self.ms is not None:
            self.steps = int(round(self.ms / 1000.0 / env.dt))

    def _pick(self, hist, new):
        hist.append(new)
        out = hist[max(0, len(hist) - 1 - self.steps)]
        if len(hist) > self.steps + 2:
            hist.pop(0)
        return out

    def on_command(self, tau, k):
        if self.where == "obs":
            return tau
        return self._pick(self._hist, np.array(tau, float))

    def on_obs(self, obs, env):
        if self.where == "command":
            return obs
        old = self._pick(self._ohist, (obs.q, obs.qd, dict(obs.extra)))
        return obs.replace(q=old[0], qd=old[1], extra=old[2])

    def label(self):
        n = f"{self.ms:g}ms" if self.steps is None else f"{self.steps}step"
        return f"delay{n}" + ("" if self.where == "command" else f"[{self.where}]")


class Friction(Perturbation):
    """Un-modelled joint friction: -(coulomb * tanh(qd/eps) + viscous * qd). Scalars or per-joint arrays."""

    def __init__(self, coulomb: float = 0.0, viscous: float = 0.0, eps: float = 0.05):
        self.fc, self.fv, self.eps = coulomb, viscous, eps

    def qfrc(self, qd, k):
        return -(self.fc * np.tanh(qd / self.eps) + self.fv * qd)

    def label(self):
        return f"friction(fc={self.fc},fv={self.fv})"


class SensorNoise(Perturbation):
    """Additive Gaussian noise on measurements. Fields: q, qd (joints), pole_theta, pole_omega (task extras).
    Reference signals are not corrupted. Std of 0 leaves a field untouched."""

    def __init__(self, q: float = 0.0, qd: float = 0.0, pole_theta: float = 0.0, pole_omega: float = 0.0):
        self.std = dict(q=q, qd=qd, pole_theta=pole_theta, pole_omega=pole_omega)
        self.rng = np.random.default_rng(0)

    def reset(self, seed, env):
        self.rng = np.random.default_rng([int(seed), 31337])

    def on_obs(self, obs, env):
        q, qd, extra = obs.q, obs.qd, dict(obs.extra)
        if self.std["q"]:
            q = q + self.rng.normal(0, self.std["q"], q.shape)
        if self.std["qd"]:
            qd = qd + self.rng.normal(0, self.std["qd"], qd.shape)
        for key in ("pole_theta", "pole_omega"):
            if key in extra and self.std[key]:
                extra[key] = extra[key] + self.rng.normal(0, self.std[key], np.shape(extra[key]))
        return obs.replace(q=q, qd=qd, extra=extra)

    def label(self):
        return "noise(" + ",".join(f"{k}={v:g}" for k, v in self.std.items() if v) + ")"


class TorquePulses(Perturbation):
    """`n` rectangular joint-torque pulses per episode (random joint, sign, magnitude U(mag)).
    Same draw order as spike01's sample_dist, seeded with seed + 9_000_000."""

    def __init__(self, n: int = 3, width: float = 0.1, mag=(4.0, 8.0), margin: float = 0.3):
        self.n, self.width, self.mag, self.margin = n, width, mag, margin
        self.table = np.zeros((0, 0))

    def reset(self, seed, env):
        dt, N, nj = env.dt, env.n_steps, env.robot.n
        out = np.zeros((N, nj))
        r = np.random.default_rng(int(seed) + 9_000_000)
        m = int(self.margin / dt)
        for _ in range(self.n):
            k0 = r.integers(m, N - m)
            j = r.integers(0, nj)
            out[k0:k0 + int(self.width / dt), j] += r.choice([-1, 1]) * r.uniform(*self.mag)
        self.table = out

    def qfrc(self, qd, k):
        return self.table[k] if k < len(self.table) else None

    def active(self, k):
        return bool(k < len(self.table) and np.any(self.table[k]))

    def label(self):
        return f"pulses(n={self.n},{self.mag[0]:g}-{self.mag[1]:g}Nm)"


class PoleLength(Perturbation):
    """True pole length differs from the nominal one the controller assumes (task parameter, pole_balance).
    PoleLength(rel=-0.2): true = nominal * 0.8.  PoleLength(length=0.5): true = 0.5 m."""

    def __init__(self, rel: float | None = None, length: float | None = None):
        if (rel is None) == (length is None):
            raise ValueError("give exactly one of rel / length")
        self.rel, self.length = rel, length

    def modify_task(self, task):
        task.L = self.length if self.length is not None else task.L * (1 + self.rel)
        return task

    def label(self):
        return f"pole{self.rel:+.0%}" if self.rel is not None else f"pole={self.length}m"


class BallRestitution(Perturbation):
    """True ball restitution differs from the nominal one (task parameter `restitution` of ball tasks; the task turns it
    into contact parameters in tip_xml/reset). Raises AttributeError for tasks without that parameter."""

    def __init__(self, restitution: float):
        self.restitution = float(restitution)

    def modify_task(self, task):
        if not hasattr(task, "restitution"):
            raise AttributeError(f"task '{task.name}' has no 'restitution' parameter")
        task.restitution = self.restitution
        return task

    def label(self):
        return f"restitution={self.restitution:g}"
