"""Whip: a 3-link arm with a free-swinging chain at the tip has to hit a target point with the CHAIN TIP. The target lies outside the arm's
reach (1.0 m) but inside reach + chain length (2.0 m): the chain has to be flung out.

Plant. 3-link arm (0.40 / 0.35 / 0.25 m, tip payload 0.2 kg) + a planar chain of `n_links` capsules (default 10 x 0.10 m, 0.015 kg each,
no joint torque, joint damping `chain_damp`) hinged at the arm tip (axes about +y, gravity on). The chain state is NOT observed
(Obs.q / qd are the arm only); the chain tip position is only logged for the metrics.
Seed -> target (x, z) = d (cos a, sin a) with d ~ U(1.15, 1.70) m from the shoulder, a ~ U(-0.45, 0.45) rad; initial arm posture
Q_START + U(-0.02, 0.02); the chain hangs straight down. Obs.ref['target'] = target (x, z). The chain tip passes through the target within
`r_hit` = 0.08 m at any time before `duration` = 2.5 s: success (the episode ends).
Metrics: `success` (0/1, primary), `min_dist` (m, closest approach of the chain tip, over the whole episode), `hit_speed` (m/s, chain tip
speed at the first hit; NaN without a hit: a faster tip = more whip-like), `energy`, `sat`.
True task parameters a perturbation may change (per link arrays of length n_links): chain_mass (kg), chain_damp (N m s/rad), chain_len (m).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..robots import RobotSpec
from .base import Task, register_task

Q_START = np.array([0.5, 0.5, 0.3])      # arm posture at t = 0 (pointing down-forward, chain hanging below the hand)


@register_task("whip")
class Whip(Task):
    name = "whip"
    duration = 2.5
    metrics = ("energy", "sat", "alive_frac")
    binary_metrics = frozenset({"success"})
    higher_is_better = frozenset({"success", "hit_speed"})
    primary_metric = "success"

    def __init__(self, robot: RobotSpec | None = None, duration: float = 2.5, n_links: int = 10, link_len: float = 0.10,
                 link_mass: float = 0.015, damp: float = 0.0004, r_hit: float = 0.08, **params):
        super().__init__(robot, **params)
        self.duration, self.n_links, self.r_hit = duration, int(n_links), r_hit
        self.chain_len = np.full(self.n_links, float(link_len))
        self.chain_mass = np.full(self.n_links, float(link_mass))
        self.chain_damp = np.full(self.n_links, float(damp))
        self._nom = dict(chain_len=self.chain_len.copy(), chain_mass=self.chain_mass.copy(), chain_damp=self.chain_damp.copy())
        self.q_start = Q_START.copy()
        self.target = np.array([1.4, 0.0])
        self.L_arm = float(sum(self.robot.link_lengths))

    @classmethod
    def default_robot(cls):
        return RobotSpec.three_link(payload_mass=0.2)

    # ---- model ----------------------------------------------------------------------------------
    def tip_xml(self, robot):
        tx, tz = (float(v) for v in self.target)
        xml = ""
        for i in reversed(range(self.n_links)):
            L, m, c = float(self.chain_len[i]), float(self.chain_mass[i]), float(self.chain_damp[i])
            pos = "0 0 0" if i == 0 else f"{float(self.chain_len[i - 1])!r} 0 0"
            site = f'<site name="whip_tip" pos="{L!r} 0 0" size="0.012" rgba="1 0.2 0.2 1"/>' if i == self.n_links - 1 else ""
            xml = (f'<body name="c{i + 1}" pos="{pos}"><joint name="cj{i + 1}" type="hinge" axis="0 1 0" damping="{c!r}"/>'
                   f'<geom type="capsule" fromto="0 0 0 {L!r} 0 0" size="0.008" mass="{m!r}" rgba="0.9 0.75 0.2 1" contype="0" conaffinity="0"/>'
                   f'{site}{xml}</body>')
        return (xml + "<!--@world-->"
                f'<geom name="target" type="sphere" size="{self.r_hit!r}" pos="{tx!r} -0.05 {tz!r}" rgba="0.2 0.8 0.3 0.35" contype="0" conaffinity="0"/>'
                '<geom name="pedestal" type="box" size="0.06 0.06 0.455" pos="0 0 -0.455" rgba="0.40 0.42 0.46 1" contype="0" conaffinity="0"/>')

    def info(self):
        n = self._nom
        return dict(n_links=self.n_links, chain_len=n["chain_len"].copy(), chain_mass=n["chain_mass"].copy(),
                    chain_damp=n["chain_damp"].copy(), r_hit=self.r_hit, duration=self.duration, q_start=self.q_start.copy(),
                    arm_reach=self.L_arm, gravity=9.81)

    # ---- episode ------------------------------------------------------------------------------------
    def reset(self, seed, env):
        n = env.n
        r = np.random.default_rng([int(seed), 5150])
        d, a = r.uniform(1.15, 1.70), r.uniform(-0.45, 0.45)
        self.target = np.array([d * np.cos(a), d * np.sin(a)])
        r2 = np.random.default_rng([int(seed), 5151])
        q = self.q_start + r2.uniform(-0.02, 0.02, n)
        env.data.qpos[:n] = q
        env.data.qpos[n] = np.pi / 2 - q.sum()               # first chain link points straight down; the rest hangs along it
        m = env.model
        m.geom_pos[m.geom("target").id] = [self.target[0], -0.05, self.target[1]]
        mujoco.mj_forward(m, env.data)

    def reference(self, k):
        return {"target": self.target.copy()}

    def ee_ref(self, k):
        return None

    def chain_tip(self, env):
        p = env.data.site_xpos[env.model.site("whip_tip").id]
        return np.array([p[0], p[2]])

    # ---- events ---------------------------------------------------------------------------------------
    def succeeded(self, env):
        return bool(np.linalg.norm(self.chain_tip(env) - self.target) < self.r_hit)

    def reward(self, env):
        return -float(np.linalg.norm(self.chain_tip(env) - self.target)) + (5.0 if self.succeeded(env) else 0.0)

    def log(self, env):
        return dict(whip_tip=self.chain_tip(env), target=self.target.copy())

    def extra_metrics(self, trace):
        out = dict(success=float(trace.outcome == "success"), min_dist=float("nan"), hit_speed=float("nan"))
        ch = trace.channels
        if "whip_tip" not in ch or len(ch["whip_tip"]) == 0:
            return out
        tip = np.asarray(ch["whip_tip"], float)
        if trace.final is not None:
            tip = np.vstack([tip, np.asarray(trace.final["channels"]["whip_tip"], float)[None]])
        tgt = np.asarray(ch["target"][0], float)
        dist = np.linalg.norm(tip - tgt, axis=1)
        out["min_dist"] = float(dist.min())
        hit = np.where(dist < self.r_hit)[0]
        if len(hit) and len(tip) > 1:
            j = max(int(hit[0]), 1)
            out["hit_speed"] = float(np.linalg.norm(tip[j] - tip[j - 1]) / trace.dt)
        return out
