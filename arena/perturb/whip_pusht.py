"""Task-parameter perturbations for `whip` and `push_t`. All change the TRUE task only (`modify_task`); the controller keeps the nominal
`task_info`. `rel` is a relative change (true = nominal * (1 + rel)).

whip (per-link arrays of the chain; `spread` > 0 multiplies each link by its own lognormal factor exp(spread * z_i), z_i fixed by
`pattern` so the same Scenario always has the same uneven chain):
  ChainMass(rel, spread, pattern)      mass of every chain link
  ChainDamping(rel, spread, pattern)   joint damping of every chain hinge
  ChainLength(rel, spread, pattern)    length of every chain link
push_t:
  TFriction(rel)       friction coefficient mu of the block: the pusher-block contact cone AND the table friction (b scales with mu)
  TMass(rel)           block mass (inertia only: the table friction coefficient b is kept, so a heavier block coasts longer)
  TComOffset(dx, dz)   centre of mass of the block moved by (dx, dz) m in the shape frame (friction acts about the true COM; the controller
                       sees the pose of the outline, not of the COM)
  TPoseNoise(pos, ang) Gaussian noise on the measured block pose (Obs.extra['t_pose'])
"""
from __future__ import annotations

import numpy as np

from .base import Perturbation


def _need(task, attr):
    if not hasattr(task, attr):
        raise AttributeError(f"task '{task.name}' has no '{attr}' parameter")


def _factors(n, rel, spread, pattern):
    z = np.random.default_rng([int(pattern), 99]).standard_normal(n) if spread else np.zeros(n)
    return (1.0 + rel) * np.exp(spread * z)


class _ChainScale(Perturbation):
    attr = ""
    tag = ""

    def __init__(self, rel: float, spread: float = 0.0, pattern: int = 0):
        self.rel, self.spread, self.pattern = float(rel), float(spread), int(pattern)

    def modify_task(self, task):
        _need(task, self.attr)
        a = np.asarray(getattr(task, self.attr), float)
        setattr(task, self.attr, a * _factors(len(a), self.rel, self.spread, self.pattern))
        return task

    def label(self):
        return f"{self.tag}{self.rel:+.0%}" + (f"~{self.spread:g}" if self.spread else "")


class ChainMass(_ChainScale):
    attr, tag = "chain_mass", "chainmass"


class ChainDamping(_ChainScale):
    attr, tag = "chain_damp", "chaindamp"


class ChainLength(_ChainScale):
    attr, tag = "chain_len", "chainlen"


class TFriction(Perturbation):
    def __init__(self, rel: float):
        self.rel = float(rel)

    def modify_task(self, task):
        _need(task, "mu")
        task.mu = task.mu * (1 + self.rel)
        task.table_b = task.table_b * (1 + self.rel)
        return task

    def label(self):
        return f"tfriction{self.rel:+.0%}"


class TMass(Perturbation):
    def __init__(self, rel: float):
        self.rel = float(rel)

    def modify_task(self, task):
        _need(task, "t_mass")
        task.t_mass = task.t_mass * (1 + self.rel)
        return task

    def label(self):
        return f"tmass{self.rel:+.0%}"


class TComOffset(Perturbation):
    def __init__(self, dx: float = 0.0, dz: float = 0.0):
        self.dx, self.dz = float(dx), float(dz)

    def modify_task(self, task):
        _need(task, "com")
        task.com = np.asarray(task.com, float) + np.array([self.dx, self.dz])
        return task

    def label(self):
        return f"tcom({self.dx * 1000:+g},{self.dz * 1000:+g})mm"


class TPoseNoise(Perturbation):
    """Gaussian noise on the measured pose of the block (Obs.extra['t_pose']: `pos` m on x, z; `ang` rad on the angle), a camera that is
    not perfect. The truth (coverage, success) is computed from the real pose."""

    def __init__(self, pos: float = 0.0, ang: float = 0.0):
        self.pos, self.ang = float(pos), float(ang)
        self.rng = np.random.default_rng(0)

    def reset(self, seed, env):
        self.rng = np.random.default_rng([int(seed), 4242])

    def on_obs(self, obs, env):
        if "t_pose" not in obs.extra:
            return obs
        extra = dict(obs.extra)
        extra["t_pose"] = extra["t_pose"] + self.rng.normal(0.0, 1.0, 3) * np.array([self.pos, self.pos, self.ang])
        return obs.replace(extra=extra)

    def label(self):
        return f"tposenoise({self.pos * 1000:g}mm,{self.ang:g}rad)"
