"""Vectorised (vmap) MJX version of arena.ArenaEnv for the 2-DoF tracking task (joint_sines reference).

Step order is the one of arena/env.py:
  command -> [Delay: command from `delay` steps ago, first command held during warm-up] -> TorqueLimit clip(cap)
  -> actuator clip(+-tau_max) -> `substeps` physics steps (RK4, dt_sim), each with
  qfrc_applied = -(fc*tanh(qd/eps) + fv*qd) + pulse[k]  -> k += 1.
The error / observation at step k is the state BEFORE action k, as in arena's Trace.
All envs run in lock-step (shared k); an episode is `n_steps` = 800 control steps.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from arena.robots import RobotSpec

from .model import ArmModel

EPS_TANH = 0.05
DMAX = 8     # largest supported delay (control steps)


class Params(NamedTuple):
    """per-env true-plant parameters and episode tables (leading axis E)"""
    payload: jax.Array      # kg
    cap: jax.Array          # Nm (TorqueLimit; use tau_max for none)
    fc: jax.Array           # Coulomb friction (Nm)
    fv: jax.Array           # viscous friction (Nm s/rad)
    delay: jax.Array        # int32 control steps
    q: jax.Array            # (E,N,n) reference tables
    qd: jax.Array
    qdd: jax.Array
    ie: jax.Array           # (E,2n) initial error
    pulse: jax.Array        # (E,N,n) torque pulse table (Nm)


class State(NamedTuple):
    d: mjx.Data             # batched
    buf: jax.Array          # (E, DMAX+1, n) command history
    k: jax.Array            # int32 scalar, control step index


def make_params(batch: dict, dtype, payload=1.5, cap=80.0, fc=0.0, fv=0.0, delay=0) -> Params:
    E = batch["q"].shape[0]
    full = lambda v, dt=dtype: jnp.broadcast_to(jnp.asarray(v, dt), (E,))
    return Params(full(payload), full(cap), full(fc), full(fv), full(delay, jnp.int32),
                  *(jnp.asarray(batch[k], dtype) for k in ("q", "qd", "qdd", "ie", "pulse")))


class MjxTracking:
    def __init__(self, spec: RobotSpec | None = None, dtype=jnp.float32):
        self.spec = spec or RobotSpec()
        self.arm = ArmModel(self.spec)
        self.dtype = dtype
        self.n, self.nsub, self.tau_max = self.spec.n, self.spec.substeps, self.spec.tau_max
        self.n_steps = int(round(4.0 / self.spec.dt))
        self.d0 = mjx.make_data(self.arm.m)
        self.m_nom = self.arm.m

    # ---------------------------------------------------------------- reset / step
    def reset(self, p: Params) -> State:
        E, n = p.q.shape[0], self.n
        d = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (E,) + x.shape), self.d0)
        qpos = (p.q[:, 0] + p.ie[:, :n]).astype(d.qpos.dtype)
        qvel = (p.qd[:, 0] + p.ie[:, n:]).astype(d.qvel.dtype)
        d = d.replace(qpos=qpos, qvel=qvel)
        return State(d, jnp.zeros((E, DMAX + 1, n), self.dtype), jnp.zeros((), jnp.int32))

    def _step_one(self, m, p_pay, cap, fc, fv, delay, pulse_k, d, buf, k, cmd):
        n = self.n
        buf = jnp.where(k == 0, jnp.broadcast_to(cmd, buf.shape), buf)
        buf = jnp.concatenate([buf[1:], cmd[None]], 0)
        out = buf[DMAX - delay]
        ex = jnp.clip(jnp.clip(out, -cap, cap), -self.tau_max, self.tau_max)

        def sub(dd, _):
            f = -(fc * jnp.tanh(dd.qvel / EPS_TANH) + fv * dd.qvel) + pulse_k
            dd = dd.replace(ctrl=ex.astype(dd.ctrl.dtype), qfrc_applied=f.astype(dd.qfrc_applied.dtype))
            return mjx.step(m, dd), None

        d, _ = jax.lax.scan(sub, d, None, length=self.nsub)
        return d, buf, ex

    def make_vstep(self, p: Params):
        """returns step(state, cmd (E,n)) -> (state, applied (E,n), sat (E,)) closed over the batched model"""
        mb, axes = self.arm.batched(p.payload)
        v = jax.vmap(self._step_one, in_axes=(axes, 0, 0, 0, 0, 0, 0, 0, 0, None, 0))
        limit = jnp.minimum(p.cap, self.tau_max)

        def step(s: State, cmd):
            cmd = jnp.nan_to_num(cmd, nan=0.0, posinf=self.tau_max, neginf=-self.tau_max).astype(self.dtype)
            kk = jnp.minimum(s.k, self.n_steps - 1)
            pk = p.pulse[:, kk]
            d, buf, ex = v(mb, p.payload, p.cap, p.fc, p.fv, p.delay, pk, s.d, s.buf, s.k, cmd)
            sat = (jnp.abs(ex) >= limit[:, None] - 1e-9).any(-1)
            return State(d, buf, s.k + 1), ex, sat

        return step

    def ref(self, p: Params, k):
        kk = jnp.minimum(k, self.n_steps - 1)
        return p.q[:, kk], p.qd[:, kk], p.qdd[:, kk]
