"""Gymnasium-compatible thin wrapper around ArenaEnv so PPO / SAC (stable-baselines3, CleanRL, ...) can train on a Scenario.

    env = ArenaGymEnv(Scenario("pole_balance").with_perturb(Delay(1)))
    obs, info = env.reset()                 # episode seed drawn from TRAIN_SEEDS (never CAL / TEST)
    obs, reward, terminated, truncated, info = env.step(action)     # action in [-1, 1]^n  ->  tau = action * tau_limit

observation = Obs.to_vector() (float32); reward = Task.reward; terminated / truncated as in ArenaEnv.step.
`reset(seed=s)` uses exactly episode seed s (so the same wrapper can replay a CAL seed); `reset()` draws from `seed_pool`.
If gymnasium is not installed a minimal stand-in with the same reset/step signatures is used (install gymnasium for
real training: `uv pip install --python .venv/bin/python gymnasium`).
"""
from __future__ import annotations

import numpy as np

from .eval.protocol import TRAIN_SEEDS
from .scenario import Scenario

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:                                           # pragma: no cover - exercised when gymnasium is absent
    class spaces:                                             # noqa: N801
        class Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low, self.high, self.dtype = np.broadcast_to(low, shape).astype(dtype), np.broadcast_to(high, shape).astype(dtype), dtype
                self.shape = tuple(shape)

    class _Base:
        np_random = None

        def reset(self, *, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)


class ArenaGymEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, scenario: Scenario, seed_pool=TRAIN_SEEDS, normalize_action: bool = True):
        self.scenario, self.env = scenario, scenario.make_env()
        self.seed_pool, self.normalize_action = np.asarray(seed_pool), normalize_action
        obs = self.env.reset(int(self.seed_pool[0]))
        d = len(obs.to_vector())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float32)
        lim = 1.0 if normalize_action else self.env.tau_limit
        self.action_space = spaces.Box(-lim, lim, shape=(self.env.n,), dtype=np.float32)
        self._rng = np.random.default_rng(0)

    @property
    def unwrapped_env(self):
        """The underlying ArenaEnv (ctx, task, model, ...)."""
        return self.env

    def reset(self, *, seed=None, options=None):
        if _Base is not object and hasattr(_Base, "reset"):
            super().reset(seed=seed)
        if seed is not None:
            ep = int(seed)
            self._rng = np.random.default_rng(ep)
        else:
            ep = int(self._rng.choice(self.seed_pool))
        obs = self.env.reset(ep)
        return obs.to_vector(), dict(episode_seed=ep)

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(self.env.n), -1.0, 1.0) if self.normalize_action \
            else np.asarray(action, float)
        tau = a * self.env.tau_limit if self.normalize_action else a
        obs, reward, terminated, truncated, info = self.env.step(tau)
        return obs.to_vector(), reward, terminated, truncated, info
