"""L5 interface: residual wrapper  tau = clip(base(obs) + bound * clip(policy(features), -1, 1)).

`policy` is any callable features -> array(n). Ported from spike02/03a (residual PPO) so a trained actor can be
dropped in without retraining; `MLPPolicy` runs a saved tanh-MLP (torch state_dict or numpy arrays) in pure numpy.
"""
from __future__ import annotations

import numpy as np

from ..types import Obs
from .base import Controller, controller_factory, register_controller


def spike_features(obs: Obs, tau_base: np.ndarray) -> np.ndarray:
    """spike02 `feats`: [q, qd, qr, qdr, qddr/3, tau_teacher/30, (q-qr)*20, (qd-qdr)*3]  (2-DoF -> 16 dims)."""
    r = obs.ref
    return np.concatenate([obs.q, obs.qd, r["q"], r["qd"], r["qdd"] / 3, tau_base / 30,
                           (obs.q - r["q"]) * 20, (obs.qd - r["qd"]) * 3]).astype(np.float32)


class MLPPolicy:
    """tanh MLP forward pass in numpy. `state` maps '0.weight','0.bias','2.weight',... (torch Sequential keys)."""

    def __init__(self, state: dict):
        idx = sorted({int(k.split(".")[0]) for k in state})
        self.layers = [(np.asarray(_np(state[f"{i}.weight"])), np.asarray(_np(state[f"{i}.bias"]))) for i in idx]

    def __call__(self, x):
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x


def _np(v):
    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


@register_controller("residual")
class Residual(Controller):
    def __init__(self, ctx, base="ctc_lqr", policy=None, bound=16.0, features=spike_features, base_kw=None):
        super().__init__(ctx)
        fac = controller_factory(base, **(base_kw or {})) if isinstance(base, str) else base
        self.base = fac(ctx)
        self.policy = policy if policy is not None else (lambda x: np.zeros(ctx.robot.n))
        self.bound, self.features = bound, features
        self.max_residual = 0.0

    def reset(self, seed=None):
        super().reset(seed)
        self.base.reset(seed)
        self.max_residual = 0.0

    def act(self, obs: Obs):
        tb = self.base.act(obs)
        m = np.clip(np.asarray(self.policy(self.features(obs, tb)), float), -1.0, 1.0)
        out = np.clip(tb + self.bound * m, -self.ctx.tau_limit, self.ctx.tau_limit)
        self.max_residual = max(self.max_residual, float(np.abs(out - tb).max()))
        return out
