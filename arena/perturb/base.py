"""Perturbation interface. A perturbation can hook four places; unused hooks are no-ops, so any
subset composes by listing them in order on a Scenario / ArenaEnv.

  modify_robot(spec)       -> TRUE plant parameters (the controller keeps the nominal spec)
  modify_task(task)        -> TRUE task parameters (pole length, ball restitution, ...). Receives a private COPY of the
                              nominal task and returns the modified task; the controller keeps the nominal task_info
  reset(seed, env)         -> per-episode setup (own RNG from the episode seed)
  on_obs(obs, env)         -> corrupt what the controller sees
  on_command(tau, k)       -> transform the commanded torque (delay, cap, ...), applied in list order
  qfrc(qd, k)              -> extra generalized force added at every physics substep (friction, pushes)
  active(k)                -> True while a disturbance is acting (used by the recovery-time metric)
"""
from __future__ import annotations

import numpy as np


class Perturbation:
    torque_limit: float | None = None      # set by actuator caps so the env can compute saturation

    def modify_robot(self, robot):
        return robot

    def modify_task(self, task):
        return task

    def reset(self, seed: int, env) -> None:
        pass

    def on_obs(self, obs, env):
        return obs

    def on_command(self, tau: np.ndarray, k: int) -> np.ndarray:
        return tau

    def qfrc(self, qd: np.ndarray, k: int):
        return None

    def active(self, k: int) -> bool:
        return False

    def label(self) -> str:
        return type(self).__name__

    def __repr__(self):
        return self.label()
