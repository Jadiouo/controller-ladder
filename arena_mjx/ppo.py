"""Residual PPO on MJX: same algorithm / hyper-parameters as experiments/spike02/run_rl.py (reimplemented in JAX).

action = clip(tau_ctc + bound * clip(a, -1, 1), +-tau_max); obs = spike02 `feats` (16-d); reward =
-tracking_w * sum_j (q-qref)^2 (after the step) - res_pen * |bound*clip(a)|^2; horizon 200 of an 800-step episode;
gamma .99, lambda .95, clip .2, Adam 3e-4 (global grad-norm clip .5), 10 epochs, minibatch 2048, tanh 128x128 actor and
critic, log-std init -1.5, orthogonal init (sqrt2 hidden, 0.01 actor head, 1.0 critic head). Net weights are stored with
torch Sequential naming ('0.weight','2.weight','4.weight') so that arena.controllers.residual.MLPPolicy can run them.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class HP:
    nenv: int = 64
    horizon: int = 200
    bound: float = 16.0
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 10
    mb: int = 2048
    tracking_w: float = 200.0
    res_pen: float = 1e-4
    init_logstd: float = -1.5


def feats(q, qd, qr, qdr, qddr, tT):
    return jnp.concatenate([q, qd, qr, qdr, qddr / 3, tT / 30, (q - qr) * 20, (qd - qdr) * 3], -1).astype(jnp.float32)


def init_mlp(key, i, o, out_scale, h=128):
    ini = jax.nn.initializers.orthogonal
    k1, k2, k3 = jax.random.split(key, 3)
    return [(ini(np.sqrt(2))(k1, (h, i)), jnp.zeros(h)), (ini(np.sqrt(2))(k2, (h, h)), jnp.zeros(h)),
            (ini(out_scale)(k3, (o, h)), jnp.zeros(o))]


def mlp(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = jnp.tanh(x)
    return x


def init_params(key, hp: HP, n=2):
    ka, kc = jax.random.split(key)
    return dict(actor=init_mlp(ka, 8 * n, n, 0.01), critic=init_mlp(kc, 8 * n, 1, 1.0),
                logstd=jnp.full((n,), hp.init_logstd, jnp.float32))


def actor_state_dict(params):
    """torch-Sequential-style numpy dict for arena MLPPolicy"""
    out = {}
    for idx, (W, b) in zip((0, 2, 4), params["actor"]):
        out[f"{idx}.weight"], out[f"{idx}.bias"] = np.asarray(W, np.float32), np.asarray(b, np.float32)
    return out


def _logp(a, mu, logstd):
    return (-0.5 * ((a - mu) / jnp.exp(logstd)) ** 2 - logstd - 0.9189385).sum(-1)


def make_iteration(env, ctc, hp: HP):
    """returns jitted (params, opt_state, state, p, key) -> (params, opt_state, state, key, metrics) and the optimizer"""
    N, H, B = hp.nenv, hp.horizon, hp.bound
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(hp.lr))

    def obs(s, p):
        q, qd = s.d.qpos, s.d.qvel
        qr, qdr, qddr = env.ref(p, s.k)
        tT = ctc.act(q, qd, qr, qdr, qddr)
        return feats(q, qd, qr, qdr, qddr, tT), tT

    def iteration(params, opt_state, s, p, key):
        step = env.make_vstep(p)
        x0, tT0 = obs(s, p)

        def body(carry, _):
            s, x, tT, key = carry
            key, k1 = jax.random.split(key)
            mu = mlp(params["actor"], x)
            act = mu + jnp.exp(params["logstd"]) * jax.random.normal(k1, mu.shape)
            lp = _logp(act, mu, params["logstd"])
            v = mlp(params["critic"], x)[:, 0]
            res = B * jnp.clip(act, -1, 1)
            cmd = jnp.clip(tT + res, -env.tau_max, env.tau_max)
            s2, _, _ = step(s, cmd)
            qr2 = env.ref(p, s2.k)[0]
            r = -hp.tracking_w * ((s2.d.qpos - qr2) ** 2).sum(-1) - hp.res_pen * (res ** 2).sum(-1)
            done = jnp.where(s2.k >= env.n_steps, 1.0, 0.0) * jnp.ones(N)
            x2, tT2 = obs(s2, p)
            return (s2, x2, tT2, key), (x, act, lp, v, r.astype(jnp.float32), done)

        (sT, xT, _, key), (X, A, LP, V, R, D) = jax.lax.scan(body, (s, x0, tT0, key), None, length=H)
        lastv = mlp(params["critic"], xT)[:, 0] * (1 - D[-1])
        nextv = jnp.concatenate([V[1:], lastv[None]], 0)

        def gae(g, t):
            r, v, nv, nd = t
            delta = r + hp.gamma * nv * (1 - nd) - v
            g = delta + hp.gamma * hp.lam * (1 - nd) * g
            return g, g

        _, adv = jax.lax.scan(gae, jnp.zeros(N), (R, V, nextv, D), reverse=True)
        ret = adv + V
        X, A, LP = X.reshape(H * N, -1), A.reshape(H * N, -1), LP.reshape(-1)
        advf, retf = adv.reshape(-1), ret.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std(ddof=1) + 1e-8)     # torch .std() is unbiased
        n = H * N

        def loss_fn(prm, x, a, lp0, ad, rt):
            mu = mlp(prm["actor"], x)
            ratio = jnp.exp(_logp(a, mu, prm["logstd"]) - lp0)
            pl = -jnp.minimum(ratio * ad, jnp.clip(ratio, 1 - hp.clip, 1 + hp.clip) * ad).mean()
            vl = 0.5 * ((mlp(prm["critic"], x)[:, 0] - rt) ** 2).mean()
            return pl + vl

        def epoch(carry, k):
            prm, ost = carry
            perm = jax.random.permutation(k, n)
            for i in range(0, n, hp.mb):
                idx = perm[i:i + hp.mb]
                g = jax.grad(loss_fn)(prm, X[idx], A[idx], LP[idx], advf[idx], retf[idx])
                upd, ost = opt.update(g, ost, prm)
                prm = optax.apply_updates(prm, upd)
            return (prm, ost), None

        key, ke = jax.random.split(key)
        (params, opt_state), _ = jax.lax.scan(epoch, (params, opt_state), jax.random.split(ke, hp.epochs))
        metrics = dict(meanR=R.mean(), std=jnp.exp(params["logstd"]).mean(), absres=jnp.abs(B * jnp.clip(A, -1, 1)).mean())
        return params, opt_state, sT, key, metrics

    return jax.jit(iteration), opt
