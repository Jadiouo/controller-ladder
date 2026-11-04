"""Computed torque + LQR gains (the arena `ctc_lqr` tracking controller / spike01 teacher) in JAX."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from arena.controllers._arm import lqr_double_integrator


class CtcLqr:
    def __init__(self, env, qp=1.6e5, qv=800.0, kp_scale=1.0, kd_scale=1.0):
        self.env = env
        Kp, Kd = lqr_double_integrator(qp, qv)
        self.Kp, self.Kd = float(Kp) * kp_scale, float(Kd) * kd_scale
        self.d0 = env.d0
        self.m = env.m_nom          # NOMINAL model (the controller never sees the true plant)

    def inverse(self, q, qd, qdd):
        """(E,n) -> (E,n) nominal inverse dynamics (gravity, damping, armature included)"""
        def one(q, qd, qdd):
            d = self.d0.replace(qpos=q.astype(self.d0.qpos.dtype), qvel=qd.astype(self.d0.qvel.dtype),
                                qacc=qdd.astype(self.d0.qacc.dtype))
            return mjx.inverse(self.m, d).qfrc_inverse
        return jax.vmap(one)(q, qd, qdd)

    def act(self, q, qd, qr, qdr, qddr):
        v = qddr + self.Kp * (qr - q) + self.Kd * (qdr - qd)
        return jnp.clip(self.inverse(q, qd, v), -self.env.tau_max, self.env.tau_max)
