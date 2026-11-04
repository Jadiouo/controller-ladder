"""Scenario = task + nominal robot + perturbation list. The unit that eval and viz take."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

from .env import ArenaEnv
from .perturb import Perturbation
from .robots import RobotSpec
from .tasks import make_task


@dataclass(frozen=True)
class Scenario:
    task: str
    task_kw: dict = field(default_factory=dict)
    robot: RobotSpec | None = None
    perturbations: tuple = ()
    name: str = ""
    history: int = 0            # length of Obs.history (previous observations); 0 = off

    def make_env(self) -> ArenaEnv:
        """A NEW env. Perturbations are deep-copied: they carry per-episode state (delay buffer, noise rng, pulse
        table), so two envs of one Scenario must never share instances."""
        return ArenaEnv(make_task(self.task, self.robot, **self.task_kw), copy.deepcopy(self.perturbations),
                        history=self.history)

    def with_perturb(self, *ps: Perturbation, name: str | None = None) -> "Scenario":
        return replace(self, perturbations=self.perturbations + tuple(ps), name=name or self.name)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        ps = "+".join(p.label() for p in self.perturbations) or "clean"
        return f"{self.task}/{ps}"
