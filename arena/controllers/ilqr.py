"""L3/L4: iLQR on the NOMINAL MuJoCo model.

ilqr      : offline trajectory optimisation over the whole episode at the first call (reference from obs.ref_preview,
            warm start = time-varying LQR around the reference), then execution of  u_k = u*_k + K_k (x_k - x*_k)
            with the time-varying gains K_k (no re-planning). The offline planning time is reported separately
            (`offline_time`), it is not part of the per-step latency.
ilqr_mpc  : receding horizon: every `replan_every` steps iLQR is re-solved over `horizon` steps from the measured state
            (warm start = previous plan shifted, closed-loop re-rolled), the terminal cost is the Riccati cost of the
            reference end; between solves the plan's feedback law is applied.
Cost, weights, torque clipping and the model: see controllers/_ocp.py. Only `ctx.make_model()` (nominal) is used.
"""
from __future__ import annotations

import time

import numpy as np

from ..types import Obs
from ._ocp import NominalOCP, TimedController
from .base import register_controller


@register_controller("ilqr")
class ILQR(TimedController):
    def __init__(self, ctx, weights=None, max_iter=20, tol=1e-4):
        super().__init__(ctx)
        self.ocp = NominalOCP(ctx, weights)
        self.model = self.ocp.model
        self.max_iter, self.tol = max_iter, tol
        self.plan = None

    def reset(self, seed=None):
        super().reset(seed)
        self.plan = None

    def _make_plan(self, obs: Obs):
        t0, c0 = time.perf_counter(), time.thread_time()
        o = self.ocp
        T = max(self.ctx.n_steps - obs.k, 1)
        Xref, Uref = o.reference(obs, T)
        Pf = o.terminal_cost(Xref[-1], Uref[-1])
        self.plan = o.ilqr(o.state_from_obs(obs), Xref, Uref, Pf, max_iter=self.max_iter, tol=self.tol)
        self.plan["k0"] = obs.k
        self.last_info["offline_s"], self.last_info["offline_cpu_s"] = time.perf_counter() - t0, time.thread_time() - c0
        self.last_info["iters"], self.last_info["cost"] = self.plan["iters"], self.plan["J"]

    def _act(self, obs: Obs):
        if self.plan is None:
            self._make_plan(obs)
        p = self.plan
        j = min(max(obs.k - p["k0"], 0), len(p["U"]) - 1)
        x = self.ocp.state_from_obs(obs)
        return p["U"][j] + p["K"][j] @ (x - p["X"][j])


@register_controller("ilqr_mpc")
class ILQRMPC(TimedController):
    def __init__(self, ctx, horizon=40, replan_every=2, max_iter=3, tol=1e-3, weights=None):
        super().__init__(ctx)
        self.ocp = NominalOCP(ctx, weights)
        self.model = self.ocp.model
        self.H, self.every, self.max_iter, self.tol = int(horizon), int(replan_every), max_iter, tol
        if self.every < 1 or self.every > self.H:
            raise ValueError("need 1 <= replan_every <= horizon")
        self.plan, self.j = None, 0

    def reset(self, seed=None):
        super().reset(seed)
        self.plan, self.j = None, 0

    def _solve(self, obs: Obs):
        o, H = self.ocp, self.H
        x0 = o.state_from_obs(obs)
        Xref, Uref = o.reference(obs, H)
        Pf = o.terminal_cost(Xref[-1], Uref[-1])
        warm = None
        if self.plan is not None:           # shift the previous plan by the steps executed since, pad with the new reference
            s = self.j
            p = self.plan
            m = len(p["U"]) - s
            Xb, Ub, Kb = Xref.copy(), Uref.copy(), np.repeat(p["K"][-1:], H, 0)
            Xb[:m + 1], Ub[:m], Kb[:m] = p["X"][s:], p["U"][s:], p["K"][s:]
            warm = (Xb, Ub, Kb)
        self.plan = o.ilqr(x0, Xref, Uref, Pf, warm=warm, max_iter=self.max_iter, tol=self.tol)
        self.j = 0
        self.last_info["iters"], self.last_info["cost"] = self.plan["iters"], self.plan["J"]

    def _act(self, obs: Obs):
        if self.plan is None or self.j >= self.every:
            t0 = time.perf_counter()
            self._solve(obs)
            self.last_info["solve_s"] = time.perf_counter() - t0
        p, j = self.plan, self.j
        self.j += 1
        return p["U"][j] + p["K"][j] @ (self.ocp.state_from_obs(obs) - p["X"][j])
