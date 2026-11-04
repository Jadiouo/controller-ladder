"""Shared optimal-control machinery for iLQR, iLQR-MPC and MPPI (NOMINAL model only).

`NominalOCP(ctx)` owns a private copy of the nominal MuJoCo model (`ctx.make_model()`; perturbations such as
PayloadError never reach it) and provides
  * one-control-step dynamics  x' = f(x, u)  with x = [qpos, qvel] (nv each; the pole hinge is a state for pole_balance),
    implemented as ONE RK4 step of length ctx.dt (the plant takes `substeps` RK4 steps of dt_sim; the difference is ~1e-6),
  * batched finite-difference linearisation of f along a whole trajectory (one `mujoco.rollout` call),
  * the quadratic cost  l = 1/2 dx'Q dx + 1/2 du'R du  (dx = x - x_ref, du = u - u_ref) for `tracking` and `pole_balance`,
  * reference states / feed-forward torques from `obs.ref_preview` (u_ref = nominal inverse dynamics),
  * a terminal cost from the discrete Riccati equation at the end of the horizon,
  * `ilqr(...)`: the iLQR solver (regularised backward pass, line-search forward pass, torque clipping).
`TimedController` records per-step compute time (see its docstring).
"""
from __future__ import annotations

import time

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout
from scipy.linalg import solve_discrete_are

from ..types import Obs
from .base import Controller

DEFAULT_WEIGHTS = {
    "tracking": dict(q=1e4, qd=1e2, r=1e-3),
    # theta, omega (pole angle / rate, world frame); q, qd = arm joint posture / velocity (keeps the tip near the hold point)
    "pole_balance": dict(theta=2e3, omega=1e2, q=20.0, qd=2.0, r=1e-2),
}


class NominalOCP:
    def __init__(self, ctx, weights: dict | None = None):
        if ctx.task not in DEFAULT_WEIGHTS:
            raise NotImplementedError(f"iLQR / MPC / MPPI do not support task {ctx.task}")
        if ctx.make_model is None:
            raise NotImplementedError("no nominal model in the control context")
        self.ctx = ctx
        self.model = ctx.make_model()                    # NOMINAL (fresh copy, owned by the controller)
        self.model.opt.timestep = ctx.dt                 # one RK4 step per control period
        self.data = mujoco.MjData(self.model)
        self.n, self.nv = ctx.robot.n, self.model.nv
        self.nx, self.nu = 2 * self.nv, self.model.nu
        self.pole = ctx.task == "pole_balance"
        self.lim = float(ctx.tau_limit)
        self.ns = int(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        self.w = dict(DEFAULT_WEIGHTS[ctx.task], **(weights or {}))
        self.Q, self.R = self._cost_matrices()
        self._eps = 1e-6

    # ---------------------------------------------------------------- cost
    def _cost_matrices(self):
        n, nv, w = self.n, self.nv, self.w
        Q = np.zeros((self.nx, self.nx))
        for i in range(n):
            Q[i, i], Q[nv + i, nv + i] = w["q"], w["qd"]
        if self.pole:       # world pole angle theta = sum(arm q) + hinge, omega likewise
            c = np.ones(nv)
            Q[:nv, :nv] += w["theta"] * np.outer(c, c)
            Q[nv:, nv:] += w["omega"] * np.outer(c, c)
        return Q, w["r"] * np.eye(self.nu)

    def cost_traj(self, X, U, Xref, Uref, Pf):
        """cost of a trajectory X (T+1, nx), U (T, nu); also batched: leading dims allowed on X/U."""
        dx, du = X - Xref, U - Uref
        run = 0.5 * np.einsum("...ti,ij,...tj->...", dx[..., :-1, :], self.Q, dx[..., :-1, :]) \
            + 0.5 * np.einsum("...ti,ij,...tj->...", du, self.R, du)
        return run + 0.5 * np.einsum("...i,ij,...j->...", dx[..., -1, :], Pf, dx[..., -1, :])

    # ---------------------------------------------------------------- state / reference
    def state_from_obs(self, obs: Obs) -> np.ndarray:
        n, nv = self.n, self.nv
        x = np.zeros(self.nx)
        x[:n], x[nv:nv + n] = obs.q, obs.qd
        if self.pole:       # hinge (relative) angle / rate from the measured world pole angle / rate
            x[n] = float(obs.extra["pole_theta"]) - float(np.sum(obs.q))
            x[nv + n] = float(obs.extra["pole_omega"]) - float(np.sum(obs.qd))
        return x

    def reference(self, obs: Obs, T: int):
        """Xref (T+1, nx), Uref (T, nu) from obs.ref_preview (state of the reference, nominal inverse-dynamics torque)."""
        n, nv = self.n, self.nv
        pv = obs.ref_preview(T + 1)
        q, qd, qdd = pv["q"], pv["qd"], pv["qdd"]
        X = np.zeros((T + 1, self.nx))
        X[:, :n], X[:, nv:nv + n] = q, qd
        acc = np.zeros((T + 1, nv))
        acc[:, :n] = qdd
        if self.pole:
            X[:, n], X[:, nv + n] = -q.sum(1), -qd.sum(1)         # upright pole: theta = omega = 0
            acc[:, n] = -qdd.sum(1)
        U = np.zeros((T, self.nu))
        d, m = self.data, self.model
        for t in range(T):
            d.qpos[:], d.qvel[:], d.qacc[:] = X[t, :nv], X[t, nv:], acc[t]
            d.qfrc_applied[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_inverse(m, d)
            U[t] = np.clip(d.qfrc_inverse[:self.nu], -self.lim, self.lim)
        return X, U

    # ---------------------------------------------------------------- dynamics
    def step(self, x, u):
        d = self.data
        d.qpos[:], d.qvel[:], d.ctrl[:] = x[:self.nv], x[self.nv:], u
        d.qfrc_applied[:] = 0
        mujoco.mj_step(self.model, d)
        return np.concatenate([d.qpos, d.qvel])

    def _full_state(self, X):
        S = np.zeros(X.shape[:-1] + (self.ns,))
        S[..., 1:] = X
        return S

    def rollout_batch(self, x0, U, nthread: int = 1):
        """Open-loop rollouts of K control sequences from x0: U (K, H, nu) -> X (K, H, nx) (states AFTER each control)."""
        K = U.shape[0]
        S0 = np.repeat(self._full_state(x0)[None], K, 0)
        st, _ = mj_rollout.rollout(self.model, [mujoco.MjData(self.model) for _ in range(nthread)] if nthread > 1
                                   else self.data, S0, np.clip(U, -self.lim, self.lim))
        return st[..., 1:]

    def linearize(self, X, U):
        """Forward finite differences of f along (X[:T], U): A (T, nx, nx), B (T, nx, nu), one batched rollout call
        (eps = 1e-6: truncation ~1e-6 relative, checked against central differences in the tests)."""
        T, nx, nu, e = len(U), self.nx, self.nu, self._eps
        P = nx + nu + 1
        base = np.concatenate([X[:T], U], 1)                       # (T, nx+nu)
        pert = np.repeat(base[:, None, :], P, 1)                    # (T, P, nx+nu); row 0 = unperturbed
        idx = np.arange(nx + nu)
        pert[:, 1 + idx, idx] += e
        S0 = self._full_state(pert[..., :nx].reshape(T * P, nx))
        ctrl = pert[..., nx:].reshape(T * P, 1, nu)
        st, _ = mj_rollout.rollout(self.model, self.data, S0, ctrl)
        Y = st[:, 0, 1:].reshape(T, P, nx)
        J = np.swapaxes((Y[:, 1:] - Y[:, :1]) / e, 1, 2)            # (T, nx, nx+nu)
        return J[:, :, :nx], J[:, :, nx:]

    def terminal_cost(self, xref_end, uref_end):
        """Pf = DARE solution for the dynamics linearised at the end of the reference (fallback: 10 Q)."""
        A, B = self.linearize(xref_end[None], uref_end[None])
        try:
            P = solve_discrete_are(A[0], B[0], self.Q + 1e-9 * np.eye(self.nx), self.R)
            if np.isfinite(P).all():
                return 0.5 * (P + P.T)
        except Exception:
            pass
        return 10.0 * self.Q

    # ---------------------------------------------------------------- iLQR
    def forward(self, x0, Xbar, Ubar, k, K, alpha, Xref, Uref, Pf):
        T = len(Ubar)
        X, U = np.zeros((T + 1, self.nx)), np.zeros((T, self.nu))
        x = x0.copy()
        for t in range(T):
            u = np.clip(Ubar[t] + alpha * k[t] + K[t] @ (x - Xbar[t]), -self.lim, self.lim)
            X[t], U[t] = x, u
            x = self.step(x, u)
            if not np.isfinite(x).all() or np.abs(x).max() > 1e4:
                return None, None, np.inf
        X[T] = x
        return X, U, float(self.cost_traj(X, U, Xref, Uref, Pf))

    def backward(self, X, U, Xref, Uref, Pf, A, B, mu):
        T, nx, nu, Q, R = len(U), self.nx, self.nu, self.Q, self.R
        k, K = np.zeros((T, nu)), np.zeros((T, nu, nx))
        Vx, Vxx = Pf @ (X[T] - Xref[T]), Pf.copy()
        dv1 = dv2 = 0.0
        for t in range(T - 1, -1, -1):
            At, Bt = A[t], B[t]
            Qx = Q @ (X[t] - Xref[t]) + At.T @ Vx
            Qu = R @ (U[t] - Uref[t]) + Bt.T @ Vx
            VB = Vxx @ Bt
            Quu = R + Bt.T @ VB + mu * np.eye(nu)
            Qux = VB.T @ At
            Qxx = Q + At.T @ Vxx @ At
            try:
                L = np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                return None
            sol = np.linalg.solve(Quu, np.column_stack([Qu, Qux]))
            kt, Kt = -sol[:, 0], -sol[:, 1:]
            k[t], K[t] = kt, Kt
            Vx = Qx + Kt.T @ Quu @ kt + Kt.T @ Qu + Qux.T @ kt
            Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt
            Vxx = 0.5 * (Vxx + Vxx.T)
            dv1 += kt @ Qu
            dv2 += 0.5 * kt @ Quu @ kt
        return k, K, dv1, dv2

    def ilqr(self, x0, Xref, Uref, Pf, warm=None, max_iter=20, tol=1e-4, mu0=1e-4):
        """Solve min_U J from x0. `warm` = (Xbar, Ubar, K) of a previous plan aligned to this horizon (closed-loop re-rolled
        from x0); otherwise the first nominal is the closed-loop roll-out of the TV-LQR around the reference.
        Returns dict(X, U, K, J, iters, J0)."""
        T = len(Uref)
        Z = np.zeros((T, self.nu))
        if warm is None:                                                     # TV-LQR around the reference trajectory
            Xbar, Ubar, K = Xref.copy(), Uref.copy(), np.zeros((T, self.nu, self.nx))
            A, B = self.linearize(Xbar, Ubar)
            bw = self.backward(Xbar, Ubar, Xref, Uref, Pf, A, B, mu0)
            K = bw[1] if bw is not None else K
        else:
            Xbar, Ubar, K = warm
        X, U, J = self.forward(x0, Xbar, Ubar, Z, K, 0.0, Xref, Uref, Pf)
        if X is None:                                                        # diverged: fall back to the feed-forward
            X, U, J = self.forward(x0, Xref, Uref, Z, np.zeros_like(K), 0.0, Xref, Uref, Pf)
            K = np.zeros_like(K)
        J0, mu, it = J, mu0, 0
        for it in range(1, max_iter + 1):
            A, B = self.linearize(X, U)
            bw = None
            while bw is None and mu < 1e8:
                bw = self.backward(X, U, Xref, Uref, Pf, A, B, mu)
                if bw is None:
                    mu = max(mu * 10, 1e-3)
            if bw is None:
                break
            k, K_new, dv1, dv2 = bw
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
                Xn, Un, Jn = self.forward(x0, X, U, k, K_new, alpha, Xref, Uref, Pf)
                if Jn < J:
                    accepted = True
                    break
            if not accepted:
                mu = max(mu * 10, 1e-3)
                if mu > 1e6:
                    break
                continue
            dJ = J - Jn
            X, U, J, K = Xn, Un, Jn, K_new
            mu = max(mu / 5, 1e-7)
            if dJ < tol * max(abs(J), 1e-9):
                break
        return dict(X=X, U=U, K=K, J=J, J0=J0, iters=it)


class TimedController(Controller):
    """Controller that records the wall-clock time of every `act` call.

    compute_times : list of wall-clock seconds per `act` call (online latency; the offline plan of `ilqr` is NOT in it)
    cpu_times     : same with the thread CPU time (less sensitive to other jobs on a busy machine)
    offline_time  : seconds spent in offline planning (iLQR), 0 otherwise
    last_info     : dict of the last call (solve_time, iterations, cost, ...), updated every step
    info_log      : list of the `last_info` dicts, one per step (the 'trace channel' for the latency axis)
    timing()      : dict(mean_ms, p95_ms, max_ms, cpu_mean_ms, cpu_p95_ms, n, offline_s)
    Subclasses implement `_act(obs)` and may put things in `self.last_info`.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self._clear_timing()

    def _clear_timing(self):
        self.compute_times, self.cpu_times, self.info_log, self.last_info, self.offline_time = [], [], [], {}, 0.0

    def reset(self, seed=None):
        super().reset(seed)
        self._clear_timing()

    def act(self, obs: Obs):
        t0, c0 = time.perf_counter(), time.thread_time()
        self.last_info = {}
        tau = self._act(obs)
        off = self.last_info.get("offline_s", 0.0)
        dt = time.perf_counter() - t0 - off
        self.cpu_times.append(time.thread_time() - c0 - self.last_info.get("offline_cpu_s", 0.0))
        self.offline_time += self.last_info.get("offline_s", 0.0)
        self.last_info["step_time"] = dt
        self.compute_times.append(dt)
        self.info_log.append(self.last_info)
        return np.clip(np.asarray(tau, float), -self.ctx.tau_limit, self.ctx.tau_limit)

    def timing(self) -> dict:
        a, c = np.asarray(self.compute_times, float) * 1e3, np.asarray(self.cpu_times, float) * 1e3
        if len(a) == 0:
            return dict(mean_ms=np.nan, p95_ms=np.nan, max_ms=np.nan, cpu_mean_ms=np.nan, cpu_p95_ms=np.nan, n=0,
                        offline_s=self.offline_time)
        return dict(mean_ms=float(a.mean()), p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()),
                    cpu_mean_ms=float(c.mean()), cpu_p95_ms=float(np.percentile(c, 95)), n=len(a),
                    offline_s=self.offline_time)
