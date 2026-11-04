"""ArenaEnv: one MuJoCo plant for (task, perturbations). Gym-like:
reset(seed) -> obs;  step(tau) -> (obs, reward, terminated, truncated, info).

Order inside step(tau_cmd):
  0. non-finite commands are replaced (nan -> 0, +-inf -> +-tau_max) and the episode is flagged diverged (failure)
  1. on_command hooks in list order (e.g. Delay then TorqueLimit)
  2. clip to the actuator range, flag saturation against the smallest effective limit
  3. `substeps` physics steps, each with qfrc_applied = sum of perturbation forces
  4. next observation = task measurement -> on_obs hooks (+ prev_action, history, reference preview)

terminated = the task reached a failure ('failure') or success ('success') state, or the simulation diverged;
truncated  = the horizon was reached without either ('timeout').  info['outcome'] is one of those strings or None.

The env holds two task objects: `nominal_task` (what the controller is told about: ctx.task_info, ctx.make_model) and
`task` (a private copy after the perturbations' modify_task, i.e. the TRUE task the physics and metrics use).
"""
from __future__ import annotations

import copy
from collections import deque

import mujoco
import numpy as np

from .robots import build_mjcf
from .types import ControlContext, Obs, readonly

DIVERGED_REWARD = -10.0


class ArenaEnv:
    def __init__(self, task, perturbations=(), history: int = 0):
        self.nominal_task = task
        self.perts = list(perturbations)
        self.task = copy.deepcopy(task)
        for p in self.perts:
            self.task = p.modify_task(self.task)
        self.nominal = task.robot
        robot = self.nominal
        for p in self.perts:
            robot = p.modify_robot(robot)
        self.robot = robot                      # TRUE plant
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(robot, self.task.tip_xml(robot)))
        self.data = mujoco.MjData(self.model)
        self.dt = robot.dt
        self.n_steps = int(round(self.task.duration / self.dt))
        limits = [robot.tau_max] + [p.torque_limit for p in self.perts if p.torque_limit is not None]
        self.tau_limit = float(min(limits))
        self.k = 0
        self.history = int(history)
        self._hist: deque = deque(maxlen=max(self.history, 1))
        self._prev_action = np.zeros(robot.n)
        self._ctrl_rng = np.random.default_rng(0)
        self._obs: Obs | None = None

    # ---- what a controller may know ----------------------------------------------
    def _make_nominal_model(self):
        return mujoco.MjModel.from_xml_string(build_mjcf(self.nominal, self.nominal_task.tip_xml(self.nominal)))

    def _ref_fn(self, k):
        return readonly(self.task.reference(int(min(max(k, 0), self.n_steps - 1))))

    @property
    def ctx(self) -> ControlContext:
        return ControlContext(robot=self.nominal, task=self.task.name, dt=self.nominal.dt,
                              tau_limit=self.nominal.tau_max, task_info=self.nominal_task.info(), n_steps=self.n_steps,
                              rng=self._ctrl_rng, ref_fn=self._ref_fn, make_model=self._make_nominal_model)

    @property
    def n(self) -> int:
        return self.robot.n

    # ---- gym-like API -----------------------------------------------------
    def reset(self, seed: int) -> Obs:
        mujoco.mj_resetData(self.model, self.data)
        self.k = 0
        self._ctrl_rng.bit_generator.state = np.random.default_rng([int(seed), 424242]).bit_generator.state
        self._prev_action = np.zeros(self.n)
        self._hist.clear()
        self.task.reset(int(seed), self)
        mujoco.mj_forward(self.model, self.data)
        for p in self.perts:
            p.reset(int(seed), self)
        self._obs = self._observe(push=True)
        return self._obs

    def _history_array(self, cur: np.ndarray) -> np.ndarray:
        H = self.history
        if H == 0:
            return np.zeros((0, len(cur)), np.float32)
        rows = list(self._hist)[-H:]
        pad = [rows[0] if rows else cur] * (H - len(rows))
        return np.stack(pad + rows).astype(np.float32)

    def _observe(self, push: bool = False) -> Obs:
        n = self.n
        obs = Obs(t=self.k * self.dt, k=self.k, q=self.data.qpos[:n].copy(), qd=self.data.qvel[:n].copy(),
                  ref=readonly(self.task.reference(min(self.k, self.n_steps - 1))), extra=self.task.measure(self),
                  ref_fn=self._ref_fn)
        for p in self.perts:
            obs = p.on_obs(obs, self)
        vec = obs.to_vector()
        obs = obs.replace(prev_action=self._prev_action.copy(), history=self._history_array(vec))
        if push:
            self._hist.append(vec)
        return obs

    def error(self):
        return self.task.error(self)

    def disturbed(self) -> bool:
        return any(p.active(self.k) for p in self.perts)

    def step(self, tau_cmd):
        raw = np.array(tau_cmd, float).reshape(self.n)
        bad_cmd = not np.isfinite(raw).all()
        cmd = np.nan_to_num(raw, nan=0.0, posinf=self.robot.tau_max, neginf=-self.robot.tau_max)
        tau = cmd
        for p in self.perts:
            tau = p.on_command(tau, self.k)
        ex = np.clip(tau, -self.robot.tau_max, self.robot.tau_max)
        sat = bool((np.abs(ex) >= self.tau_limit - 1e-9).any())
        d = self.data
        d.ctrl[:self.n] = ex
        for _ in range(self.robot.substeps):
            f = np.zeros(self.model.nv)
            for p in self.perts:
                fp = p.qfrc(d.qvel[:self.n], self.k)
                if fp is not None:
                    f[:self.n] += fp
            d.qfrc_applied[:] = f
            mujoco.mj_step(self.model, d)
        self.k += 1
        self._prev_action = cmd.copy()
        diverged = bad_cmd or not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        failure = diverged or bool(self.task.terminated(self))
        success = (not failure) and bool(self.task.succeeded(self))
        terminated = failure or success
        truncated = (not terminated) and self.k >= self.n_steps
        outcome = "failure" if failure else "success" if success else "timeout" if truncated else None
        if not diverged:
            self._obs = self._observe(push=True)
        reward = DIVERGED_REWARD if diverged else float(self.task.reward(self))
        return self._obs, reward, terminated, truncated, dict(applied=ex, saturated=sat, diverged=diverged,
                                                              outcome=outcome, cmd=cmd)
