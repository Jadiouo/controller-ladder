"""Task interface + registry.

A Task owns: the episode length, the reference signal, the initial state, the true error, and termination.
It sees the simulator only through `env` (ArenaEnv): env.model, env.data, env.robot (TRUE spec), env.k, env.dt.

Optional pieces (all have defaults):
  error(env)      TRUE error vector for tracking-style metrics; None (the default) = the task has no error signal
  terminated(env) failure condition (episode ends, outcome 'failure')
  succeeded(env)  success condition (episode ends, outcome 'success')
  reward(env)     RL reward after a step (never used by the evaluation metrics)
  metrics         names of the generic protocol metrics this task wants (see eval/metrics.py METRICS)
  extra_metrics   task-specific metrics computed from the Trace (event times, success, tip error, ...)
  primary_metric  the metric a paper table should lead with (rmse for tracking, survival_time for balance, ...)
Episode outcome: 'failure' | 'success' | 'timeout' (horizon reached without either event).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..registry import Registry
from ..robots import RobotSpec

TASKS = Registry("task")
register_task = TASKS.register


class Task(ABC):
    name: str = "task"
    duration: float = 4.0                    # seconds
    higher_is_better: frozenset = frozenset()   # extra metric names where larger = better
    binary_metrics: frozenset = frozenset()     # extra metric names that are 0/1 per episode (bootstrap CI in tables)
    metrics: tuple = ("rmse", "ss_rmse", "settle_time", "recovery_time", "sat", "energy", "alive_frac")
    primary_metric: str = "rmse"
    settle_thr = None        # None -> protocol SETTLE_THR for every error component; else scalar / per-component array
                             # in the UNITS OF THE ERROR COMPONENTS (rad for angles, m for lengths)

    def __init__(self, robot: RobotSpec | None = None, **params):
        self.robot = robot or self.default_robot()      # NOMINAL spec (used for geometry/references)
        self.params = params

    # ---- static description -------------------------------------------------
    @classmethod
    def default_robot(cls) -> RobotSpec:
        return RobotSpec.two_link()

    def tip_xml(self, robot: RobotSpec) -> str:
        """Extra MJCF placed inside the tip body (default: none)."""
        return ""

    def info(self) -> dict:
        """Nominal constants a controller is allowed to know (goes into ControlContext.task_info)."""
        return {}

    # ---- episode ---------------------------------------------------------------
    @abstractmethod
    def reset(self, seed: int, env) -> None:
        """Set the initial state in env.data and build the reference for this seed. Must be a pure function of seed."""

    @abstractmethod
    def reference(self, k: int) -> dict:
        """Reference signals at control step k (dict of arrays)."""

    def measure(self, env) -> dict:
        """Task-specific measured quantities exposed in Obs.extra (before sensor perturbations)."""
        return {}

    def error(self, env) -> np.ndarray | None:
        """TRUE error vector of the current state (what RMSE / settling are computed on), or None for event tasks.
        Keep all components in ONE unit (do not mix rad and m); report other quantities as log channels / extra metrics."""
        return None

    def terminated(self, env) -> bool:
        """Failure condition: the episode ends with outcome 'failure'."""
        return False

    def succeeded(self, env) -> bool:
        """Success condition: the episode ends with outcome 'success'."""
        return False

    def reward(self, env) -> float:
        """RL reward for the step that just happened (state already advanced). Default 0."""
        return 0.0


    def log(self, env) -> dict:
        """Extra per-step ground-truth channels stored in the Trace (name -> scalar/array)."""
        return {}

    def ee_ref(self, k: int):
        """Reference tip position (2,) or None."""
        return None

    def extra_metrics(self, trace) -> dict:
        return {}


def make_task(name: str, robot: RobotSpec | None = None, **params) -> Task:
    return TASKS.get(name)(robot, **params)
