"""L6: learned student (BC-MLP / MIP / diffusion / flow) loaded from a checkpoint of `arena.learn.train`.

The student sees ONLY the `Obs` it is handed (window of `Obs.history` + current row -> features); it never touches
`ctx.make_model`, the task or the env.  Torch is imported lazily so the core arena does not need it.

  factory = controller_factory("student", ckpt="ckpt/mip_s0.pt", exec_k=4)          # receding horizon, handled inside
  factory = controller_factory("hold", base=controller_factory("student", ckpt=..., chunk_out=True), every=4, chunk=True)
`exec_k` = how many actions of each predicted chunk are executed before re-planning (default = the whole chunk).
`chunk_out=True` makes act() return the (Ta, n) chunk itself, for HoldWrapper(chunk=True, every=k).
The env must be built with `Scenario(history=H)` where H is the history the student was trained with.
Randomness (diffusion / flow initial noise) comes from a torch Generator seeded from `self.rng` in reset(seed).
Timing (latency axis B): `infer_times` = wall seconds of every network call (one per chunk), `timing()` summarises.
"""
from __future__ import annotations

import time

import numpy as np

from ..types import Obs
from .base import Controller, register_controller


@register_controller("student")
class Student(Controller):
    def __init__(self, ctx, ckpt=None, loaded=None, exec_k: int | None = None, chunk_out: bool = False,
                 device: str = "cpu", sample_kw: dict | None = None):
        super().__init__(ctx)
        import torch
        from ..learn.train import load_checkpoint
        self.torch = torch
        self.L = loaded if loaded is not None else load_checkpoint(ckpt, device)
        self.dev = self.L.device
        self.chunk = self.L.policy.chunk
        self.exec_k = int(exec_k or self.chunk)
        if not 1 <= self.exec_k <= self.chunk:
            raise ValueError("exec_k must be in [1, chunk]")
        self.chunk_out, self.sample_kw = chunk_out, dict(sample_kw or {})
        self.tau_limit = float(ctx.tau_limit)
        self.infer_times: list = []
        self._gen = None
        self._buf, self._i = None, 0

    def reset(self, seed=None):
        super().reset(seed)
        self._gen = self.torch.Generator(device=self.dev).manual_seed(int(self.rng.integers(2 ** 62)))
        self._buf, self._i = None, 0

    def _infer(self, obs: Obs) -> np.ndarray:
        torch = self.torch
        feat = self.L.spec.expand(self.L.spec.window(obs))[None]
        t0 = time.perf_counter()
        x = torch.as_tensor(feat, device=self.dev)
        if self._gen is None:
            self.reset(self.seed)
        out = self.L.policy.act_chunk(x, self._gen, **self.sample_kw)[0].cpu().numpy()
        self.infer_times.append(time.perf_counter() - t0)
        return np.clip(out.astype(float), -self.tau_limit, self.tau_limit)

    def act(self, obs: Obs):
        if self.chunk_out:
            return self._infer(obs)[:self.exec_k]
        if self._buf is None or self._i >= self.exec_k:
            self._buf, self._i = self._infer(obs), 0
        a = self._buf[self._i]
        self._i += 1
        return a.copy()

    def timing(self) -> dict:
        t = np.asarray(self.infer_times) * 1e3
        return dict(n=len(t), mean_ms=float(t.mean()), p95_ms=float(np.percentile(t, 95)), max_ms=float(t.max())) if len(t) else {}
