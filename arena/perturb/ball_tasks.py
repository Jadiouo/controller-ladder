"""Task-parameter perturbations for the ball tasks (catch, toss). All change the TRUE task only (`modify_task`); the
controller keeps the nominal `task_info`. `rel` is a relative change of the nominal value (true = nominal * (1 + rel)).

AirDrag(rel)         linear drag coefficient c (N s/m) of the projectile
ObjectMass(rel)      projectile mass (kg): changes the drag deceleration c/m and the load carried before the release
ReleaseDelay(ms)     toss only: the object leaves the gripper `ms` later than scheduled (actuation latency of the gripper)
BallObsNoise(std)    Gaussian noise (m) on the measured projectile position (Obs.extra['ball_pos'])
BallObsDelay(steps)  the measured projectile position is `steps` control periods old (vision latency); q, qd are not delayed
"""
from __future__ import annotations

import numpy as np

from .base import Perturbation


def _need(task, attr):
    if not hasattr(task, attr):
        raise AttributeError(f"task '{task.name}' has no '{attr}' parameter")


class AirDrag(Perturbation):
    def __init__(self, rel: float):
        self.rel = float(rel)

    def modify_task(self, task):
        _need(task, "drag")
        task.drag = task.drag * (1 + self.rel)
        return task

    def label(self):
        return f"drag{self.rel:+.0%}"


class ObjectMass(Perturbation):
    def __init__(self, rel: float):
        self.rel = float(rel)

    def modify_task(self, task):
        _need(task, "ball_mass")
        task.ball_mass = task.ball_mass * (1 + self.rel)
        return task

    def label(self):
        return f"mass{self.rel:+.0%}"


class ReleaseDelay(Perturbation):
    def __init__(self, ms: float):
        self.ms = float(ms)

    def modify_task(self, task):
        _need(task, "release_delay")
        task.release_delay = self.ms / 1000.0
        return task

    def label(self):
        return f"release+{self.ms:g}ms"


class BallObsNoise(Perturbation):
    def __init__(self, std: float):
        self.std = float(std)
        self.rng = np.random.default_rng(0)

    def reset(self, seed, env):
        self.rng = np.random.default_rng([int(seed), 27182])

    def on_obs(self, obs, env):
        if "ball_pos" not in obs.extra or not self.std:
            return obs
        extra = dict(obs.extra)
        extra["ball_pos"] = extra["ball_pos"] + self.rng.normal(0, self.std, 2)
        return obs.replace(extra=extra)

    def label(self):
        return f"ballnoise{self.std * 1000:g}mm"


class BallObsDelay(Perturbation):
    def __init__(self, steps: int):
        self.steps = int(steps)
        self._h: list = []

    def reset(self, seed, env):
        self._h = []

    def on_obs(self, obs, env):
        if "ball_pos" not in obs.extra:
            return obs
        self._h.append(np.array(obs.extra["ball_pos"]))
        old = self._h[max(0, len(self._h) - 1 - self.steps)]
        if len(self._h) > self.steps + 2:
            self._h.pop(0)
        extra = dict(obs.extra)
        extra["ball_pos"] = old
        return obs.replace(extra=extra)

    def label(self):
        return f"balldelay{self.steps}step"
