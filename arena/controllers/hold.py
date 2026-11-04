"""Multi-rate wrapper: the base controller is queried every `every` control steps and its output is held in between,
or (chunk=True) the base returns an action chunk (>= every, n) that is executed row by row (diffusion policy / ACT style)."""
from __future__ import annotations

import numpy as np

from ..types import Obs
from .base import Controller, controller_factory, register_controller


@register_controller("hold")
class HoldWrapper(Controller):
    def __init__(self, ctx, base="pid", every: int = 4, chunk: bool = False, base_kw=None):
        super().__init__(ctx)
        if every < 1:
            raise ValueError("every must be >= 1")
        fac = controller_factory(base, **(base_kw or {})) if isinstance(base, str) else base
        self.base, self.every, self.chunk = fac(ctx), int(every), chunk
        self._i, self._buf = 0, None

    def reset(self, seed=None):
        super().reset(seed)
        self.base.reset(seed)
        self._i, self._buf = 0, None

    def act(self, obs: Obs):
        j = self._i % self.every
        if j == 0:
            out = np.asarray(self.base.act(obs), float)
            if self.chunk:
                if out.ndim != 2 or len(out) < 1:
                    raise ValueError("chunk=True needs the base controller to return a (T, n) action chunk")
                self._buf = np.vstack([out[:self.every], np.repeat(out[-1:], max(0, self.every - len(out)), 0)])
            else:
                self._buf = np.repeat(out.reshape(1, -1), self.every, 0)
        self._i += 1
        return self._buf[j].copy()
