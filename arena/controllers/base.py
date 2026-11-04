"""Controller interface + registry.

    ctrl = factory(ctx)      # factory: Callable[[ControlContext], Controller]; ctx carries NOMINAL robot/task info
    ctrl.reset(seed)         # once per episode, with the EPISODE seed; re-seeds ctrl.rng (and clears internal state)
    tau = ctrl.act(obs)      # obs: arena.types.Obs  ->  torque (n,)

Randomness inside a controller must come from `self.rng` (re-seeded by reset(seed)) or `self.ctx.rng` (re-seeded by the
env at every reset): never from the global numpy/torch RNG, otherwise two runs of the same episode differ.
Subclasses that override reset must accept `seed=None` and call `super().reset(seed)`.

Controllers never see the true plant. A controller that does not support a task should raise
NotImplementedError in __init__ (the demo/eval will report it instead of silently scoring garbage).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from ..registry import Registry
from ..types import ControlContext, Obs

CONTROLLERS = Registry("controller")
register_controller = CONTROLLERS.register


class Controller(ABC):
    def __init__(self, ctx: ControlContext):
        self.ctx = ctx
        self.seed = None
        self.rng = np.random.default_rng([0, 0xC0DE])

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng([0 if seed is None else int(seed), 0xC0DE])

    @abstractmethod
    def act(self, obs: Obs) -> np.ndarray:
        ...


def controller_factory(name: str, **kw):
    """Factory for a registered controller with fixed hyper-parameters: controller_factory('pid', kp=300)."""
    cls = CONTROLLERS.get(name)
    return partial(cls, **kw)


def as_factory(c):
    """Accept a registered name, a Controller subclass or any callable ctx -> Controller."""
    return controller_factory(c) if isinstance(c, str) else c
