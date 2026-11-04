"""L4: sampling-based MPC (MPPI) on the NOMINAL MuJoCo model.

Every `replan_every` steps: K control sequences  U_nom + eps  (eps ~ N(0, sigma^2), temporally smoothed by an
exponential filter with coefficient `smooth`) are rolled out from the measured state with `mujoco.rollout` (C, one
thread by default, no extra processes), scored with the same quadratic cost as iLQR (reference from obs.ref_preview,
Riccati terminal cost) and combined with softmax weights exp(-(S - S_min) / lam). The first control of the updated
sequence is applied (held between re-plans), the sequence is shifted for the next call. All randomness comes from
`self.rng` (re-seeded by reset(seed)): the same seed gives the same episode.
The nominal sequence is initialised with the feed-forward torque of the reference (nominal inverse dynamics).
"""
from __future__ import annotations

import numpy as np

from ..types import Obs
from ._ocp import NominalOCP, TimedController
from .base import register_controller


@register_controller("mppi")
class MPPI(TimedController):
    def __init__(self, ctx, samples=128, horizon=25, sigma=None, lam=None, smooth=0.5, iters=1, replan_every=1, nthread=1,
                 weights=None):
        super().__init__(ctx)
        self.ocp = NominalOCP(ctx, weights)
        self.model = self.ocp.model
        self.K, self.H, self.smooth, self.iters = int(samples), int(horizon), float(smooth), int(iters)
        self.every, self.nthread = int(replan_every), int(nthread)
        self.sigma = float(sigma if sigma is not None else 0.1 * ctx.tau_limit)
        self.lam = lam            # None -> adaptive: lam = 0.1 * (median sample cost - min), recomputed every solve
        self.U = None
        self.j = 0

    def reset(self, seed=None):
        super().reset(seed)
        self.U, self.j = None, 0

    def _solve(self, obs: Obs):
        o, H, K = self.ocp, self.H, self.K
        x0 = o.state_from_obs(obs)
        Xref, Uref = o.reference(obs, H)
        Pf = o.terminal_cost(Xref[-1], Uref[-1])
        if self.U is None:
            self.U = Uref.copy()
        else:                                                       # shift by the steps executed, pad with the new feed-forward
            s = min(self.j, H)
            self.U = np.concatenate([self.U[s:], Uref[H - s:]]) if s < H else Uref.copy()
        for _ in range(self.iters):
            eps = self.rng.normal(0.0, self.sigma, (K, H, o.nu))
            for t in range(1, H):                                   # temporal smoothing (keeps the std roughly constant)
                eps[:, t] = self.smooth * eps[:, t - 1] + np.sqrt(1 - self.smooth ** 2) * eps[:, t]
            eps[0] = 0.0                                            # the nominal sequence itself is one sample
            Us = np.clip(self.U[None] + eps, -o.lim, o.lim)
            Xs = o.rollout_batch(x0, Us, self.nthread)             # (K, H, nx): state after each control
            Xfull = np.concatenate([np.broadcast_to(x0, (K, 1, o.nx)), Xs], 1)
            S = o.cost_traj(Xfull, Us, Xref, Uref, Pf)
            S = np.where(np.isfinite(S), S, np.inf)
            lam = self.lam if self.lam is not None else max(0.1 * (np.median(S) - S.min()), 1e-9)
            w = np.exp(-(S - S.min()) / lam)
            w /= w.sum()
            self.U = np.clip(self.U + np.einsum("k,kht->ht", w, Us - self.U[None]), -o.lim, o.lim)
        self.j = 0
        self.last_info["cost"], self.last_info["ess"] = float(S.min()), float(1.0 / np.sum(w ** 2))

    def _act(self, obs: Obs):
        if self.U is None or self.j >= self.every:
            self._solve(obs)
        u = self.U[self.j]
        self.j += 1
        return u
