"""Whole-episode rollouts on the device (lax.scan over the control steps)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

SS_START = 100       # steady state: t >= 0.5 s at 200 Hz (arena SS_START_S / dt)


def rollout(env, policy, p, record: bool = False):
    """policy(state, q, qd, qr, qdr, qddr) -> command (E,n). Metrics follow arena.eval.metrics (rmse over the 800 states
    BEFORE each action, ss_rmse t>=0.5 s, energy = mean over steps of sum tau^2, sat = fraction of steps at the limit)."""
    s0 = env.reset(p)
    step = env.make_vstep(p)

    def body(carry, _):
        s, e2s, e2ss, en, sat = carry
        q, qd = s.d.qpos, s.d.qvel
        qr, qdr, qddr = env.ref(p, s.k)
        cmd = policy(s, q, qd, qr, qdr, qddr)
        e2 = ((q - qr) ** 2).mean(-1)
        s2, ex, sa = step(s, cmd)
        carry = (s2, e2s + e2, e2ss + jnp.where(s.k >= SS_START, e2, 0.0), en + (ex ** 2).sum(-1), sat + sa)
        return carry, ((q, qd, ex) if record else None)

    z = jnp.zeros(p.q.shape[0], env.dtype)
    (sT, e2s, e2ss, en, sat), rec = jax.lax.scan(body, (s0, z, z, z, z), None, length=env.n_steps)
    N = env.n_steps
    out = dict(rmse=jnp.sqrt(e2s / N), ss_rmse=jnp.sqrt(e2ss / (N - SS_START)), energy=en / N, sat=sat / N)
    if record:
        out["q"], out["qd"], out["tau"] = rec
    return out


def replay(env, p, taus):
    """Apply a fixed torque sequence taus (E,T,n); returns q, qd after every control step, (E,T,n)."""
    s0 = env.reset(p)
    step = env.make_vstep(p)

    def body(s, tau):
        s2, ex, _ = step(s, tau)
        return s2, (s2.d.qpos, s2.d.qvel)

    _, (q, qd) = jax.lax.scan(body, s0, jnp.swapaxes(jnp.asarray(taus, env.dtype), 0, 1))
    return jnp.swapaxes(q, 0, 1), jnp.swapaxes(qd, 0, 1)
