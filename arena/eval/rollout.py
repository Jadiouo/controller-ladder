"""One episode -> Trace. The single place where controller, env and recording meet."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..controllers.base import as_factory
from ..scenario import Scenario


@dataclass
class Trace:
    """Everything one episode produced. Row k of err / q / channels is the state BEFORE action k.

    outcome : 'success' | 'failure' | 'timeout'.  failure = task failure OR simulator / controller divergence;
              timeout = the horizon was reached with neither event.  (derived from terminated/diverged if left empty)
    final   : the state AFTER the last executed action (the one that triggered a termination): dict(err=..., q=...,
              channels={name: value}); None for hand-made traces.
    """
    dt: float
    n_steps: int                 # nominal horizon (padded length)
    t_end: float                 # time at which the episode ended
    terminated: bool             # ended before the horizon or by an event (failure / success / divergence)
    err: np.ndarray              # (T, m) true error vector BEFORE each action, T = executed steps; m = 0 if the task has none
    tau_cmd: np.ndarray          # (T, n) controller output (non-finite values replaced by the env)
    tau_applied: np.ndarray      # (T, n) after perturbation chain and clipping
    sat: np.ndarray              # (T,) bool
    pulse: np.ndarray            # (T,) bool, a disturbance is active
    ee_ref: np.ndarray | None = None
    channels: dict = field(default_factory=dict)    # task ground-truth logs, each (T, ...)
    q: np.ndarray | None = None  # (T, n) true joint positions
    diverged: bool = False       # simulation (or the controller output) went non-finite
    outcome: str = ""
    final: dict | None = None

    def __post_init__(self):
        if not self.outcome:        # hand-made traces: an early end without further information is a failure
            self.outcome = "failure" if (self.diverged or self.terminated) else "timeout"

    @property
    def failure(self) -> bool:
        return self.outcome == "failure"

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def timeout(self) -> bool:
        return self.outcome == "timeout"

    def _raw(self, name):
        if name == "err":
            return self.err, (None if self.final is None else self.final.get("err"))
        if name == "q":
            return self.q, (None if self.final is None else self.final.get("q"))
        if name in self.channels:
            return self.channels[name], (None if self.final is None else self.final.get("channels", {}).get(name))
        raise KeyError(name)

    def series(self, name: str, fail: float = 0.0) -> np.ndarray:
        """The per-step series `name` ('err', 'q' or a channel) as an (n_steps, ...) array with ONE rule for episodes
        that end early: append the terminal state, then fill the remaining steps with the constant `fail` (outcome
        'failure', also replaces non-finite values) or by holding the terminal state (outcome 'success')."""
        a, fin = self._raw(name)
        a = np.asarray(a, float)
        if fin is not None and len(a) < self.n_steps:
            a = np.concatenate([a, np.asarray(fin, float)[None]])
        a = np.nan_to_num(a, nan=fail, posinf=fail, neginf=-fail)
        short = self.n_steps - len(a)
        if short > 0:
            fill = np.repeat(a[-1:], short, 0) if (self.outcome == "success" and len(a)) \
                else np.full((short,) + a.shape[1:], fail)
            a = np.concatenate([a, fill])
        return a[:self.n_steps]

    def series_channel(self, name: str, fail: float = 0.0):
        """series() for a task log channel, or None if the task did not log it."""
        return self.series(name, fail) if name in self.channels else None


def run_episode(scenario: Scenario, controller, seed: int, env=None, frame_cb=None) -> Trace:
    """controller: registered name or factory(ctx). frame_cb(env, k) is called before each action (used by viz)."""
    env = env or scenario.make_env()
    ctrl = as_factory(controller)(env.ctx)
    obs = env.reset(seed)
    ctrl.reset(int(seed))
    task = env.task
    rows = dict(err=[], tau_cmd=[], tau=[], sat=[], pulse=[], ee_ref=[], q=[], ch={})
    outcome, div, k, final = "timeout", False, 0, None
    for k in range(env.n_steps):
        e = env.error()
        rows["err"].append(np.zeros(0) if e is None else np.asarray(e, float))
        rows["pulse"].append(env.disturbed())
        rows["q"].append(env.data.qpos[:env.n].copy())
        ref = task.ee_ref(k)
        rows["ee_ref"].append(None if ref is None else np.asarray(ref))
        for key, v in task.log(env).items():
            rows["ch"].setdefault(key, []).append(np.asarray(v))
        if frame_cb is not None:
            frame_cb(env, k)
        tau = np.asarray(ctrl.act(obs), float)
        obs, _, terminated, truncated, info = env.step(tau)
        rows["tau_cmd"].append(info["cmd"])
        rows["tau"].append(info["applied"])
        rows["sat"].append(info["saturated"])
        if terminated or truncated:
            div = info["diverged"]
            outcome = info["outcome"]
            e = env.error()                    # the state that ended the episode (may be non-finite after divergence)
            final = dict(err=np.zeros(0) if e is None else np.asarray(e, float), q=env.data.qpos[:env.n].copy(),
                         channels={key: np.asarray(v) for key, v in task.log(env).items()})
            break
    if rows["ee_ref"] and rows["ee_ref"][0] is None:
        ee_ref = None
    else:
        ee_ref = np.array(rows["ee_ref"])
    return Trace(dt=env.dt, n_steps=env.n_steps, t_end=(k + 1) * env.dt, terminated=outcome in ("failure", "success"),
                 diverged=div, outcome=outcome, final=final,
                 err=np.array(rows["err"]), tau_cmd=np.array(rows["tau_cmd"]), tau_applied=np.array(rows["tau"]),
                 sat=np.array(rows["sat"], bool), pulse=np.array(rows["pulse"], bool), ee_ref=ee_ref,
                 channels={k_: np.array(v) for k_, v in rows["ch"].items()}, q=np.array(rows["q"]))
