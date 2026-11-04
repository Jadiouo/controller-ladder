"""Classical teacher for `push_t`: sampling-based push planning on a quasi-static model + operational-space computed torque.

Knowledge (NOMINAL only): goal pose, nominal friction coefficient mu, the block's limit-surface radius rho, pusher radius, the arm model
(gravity off, as the task says). It does NOT know the table friction, the mass or the COM offset of the TRUE block.
Loop (closed loop at primitive level; the observation is the measured shape pose Obs.extra['t_pose']):
  1. PLAN  (`_pusht_plan.PushPlanner`): beam search (depth 4) over push primitives = (contact point on the outline, push direction inside the
     friction cone, push distance), rolled out with the quasi-static ellipsoidal-limit-surface model; the first primitive of the cheapest
     leaf (pose error + a cost per primitive + a little travel) whose approach path exists is executed.
  2. APPROACH: visibility-graph path around the outline offset by the pusher radius, to a gate point in front of the contact point.
  3. PUSH: straight line at 8 cm/s along the chosen direction; no re-plan while pushing.
  4. HOLD 0.2 s so that the block stops; the pose samples of the hold are averaged (a noise filter), then back to 1 with that pose. If no
     candidate has an approach path the pusher backs off 6 cm from the block and plans again.
Low level: tip reference moving along the path (speed profile with acceleration limit, slows down when the tip lags), operational-space
computed torque  qdd = J^+ (kp e + kd ed - Jdot qd),  tau = inverse dynamics (nominal arm, joint damping included).
"""
from __future__ import annotations

import numpy as np

from ..robots import fk, jacobian, jdot_qd
from ..tasks import _pusht_geom as G
from ..types import Obs
from ._arm import ArmModel
from ._pusht_plan import PushPlanner, approach_path, pose_cost
from .base import Controller, register_controller

BOX_X, BOX_Z = (0.30, 0.92), (-0.34, 0.34)       # the pusher stays inside this box (well inside the arm's reach)


@register_controller("push_t_teacher")
class PushTTeacher(Controller):
    def __init__(self, ctx, kp=900.0, kd=60.0, v_approach=0.40, v_push=0.08, a_max=2.0, depth=4, beam=10, n_try=6,
                 travel_w=0.3, hold_s=0.2):
        super().__init__(ctx)
        if ctx.task != "push_t":
            raise NotImplementedError(f"push_t_teacher does not support task {ctx.task}")
        info = ctx.task_info
        self.arm = ArmModel(ctx.robot)
        self.arm.model.opt.gravity[:] = [0.0, 0.0, -float(info.get("gravity", 0.0))]
        self.goal = np.asarray(info["goal"], float)
        self.r_p = float(info["pusher_radius"])
        self.planner = PushPlanner(info["mu"], info["rho2"], self.r_p, self.goal, depth=depth, beam=beam)
        self.kp, self.kd, self.v_app, self.v_push, self.a_max = kp, kd, v_approach, v_push, a_max
        self.n_try, self.travel_w, self.hold_n = n_try, travel_w, int(round(hold_s / ctx.dt))
        self.plan_log = []
        self._clear()

    def _clear(self):
        self.legs, self.leg, self.s, self.vel = [], 0, 0.0, 0.0       # legs: (a, b, speed, is_push)
        self.hold, self.pose_at_plan, self.k_plan = 0, None, 0
        self.hold_pt = None
        self.retreats = 0
        self.buf = []

    def reset(self, seed=None):
        super().reset(seed)
        self._clear()
        self.plan_log = []

    # ---- planning -------------------------------------------------------------------------------------------
    def _rank(self, pose):
        return self.planner.plan(pose)

    def _filtered_pose(self, pose):
        """Mean of the samples taken while the block rested (skipping the first 10 settling steps): averages out measurement noise."""
        b = np.array(self.buf[10:]) if len(self.buf) > 12 else None
        if b is None:
            return pose
        return np.array([b[:, 0].mean(), b[:, 1].mean(), np.arctan2(np.sin(b[:, 2]).mean(), np.cos(b[:, 2]).mean())])

    def _plan(self, obs, tip, pose):
        ranked = self._rank(pose)
        cands = []
        for i, cost in ranked[: self.n_try]:
            s, u, dist, nout = self.planner.world_primitive(pose, i)
            gate = s + nout * 0.06
            end = s + u * (dist + 0.004)
            pts = [s, gate, end]
            if not all(BOX_X[0] < p[0] < BOX_X[1] and BOX_Z[0] < p[1] < BOX_Z[1] for p in pts):
                continue
            path = approach_path(tip, s, gate, pose, self.r_p)
            if path is None:
                continue
            length = sum(np.linalg.norm(path[j + 1] - path[j]) for j in range(len(path) - 1))
            cands.append((cost + self.travel_w * length, i, path, end, cost))
        if not cands:
            return False
        cands.sort(key=lambda c: c[0])
        _, i, path, end, cost = cands[0]
        self.legs = [(path[j], path[j + 1], self.v_app, False) for j in range(len(path) - 1)]
        self.legs.append((path[-1], end, self.v_push, True))
        self.leg, self.s, self.vel = 0, 0.0, 0.0
        self.pose_at_plan, self.k_plan = pose.copy(), obs.k
        self.buf = []
        self.plan_log.append(dict(k=obs.k, pose=pose.copy(), cost=cost, prim=int(i), now=float(pose_cost(pose, self.goal))))
        return True

    # ---- reference generation -----------------------------------------------------------------------------------
    def _advance(self, tip):
        """Move the reference along the legs; returns (p_ref, v_ref) and whether the last leg is finished."""
        a, b, spd, is_push = self.legs[self.leg]
        ln = float(np.linalg.norm(b - a))
        rem = ln - self.s
        nxt = self.legs[self.leg + 1][2] if self.leg + 1 < len(self.legs) else 0.0
        v_lim = np.sqrt(nxt ** 2 + 2 * self.a_max * max(rem, 0.0))
        if is_push:
            v_lim = spd if rem > 0.0 else 0.0
        pref = a + (b - a) * (self.s / max(ln, 1e-9))
        lag = float(np.linalg.norm(pref - tip))
        gate = float(np.clip(1.0 - (lag - 0.01) / 0.03, 0.05, 1.0))        # wait for the tip when it lags the reference
        v_goal = min(spd, v_lim) * gate
        dt = self.ctx.dt
        self.vel = float(np.clip(v_goal, self.vel - self.a_max * dt, self.vel + self.a_max * dt))
        self.s += self.vel * dt
        done = False
        if self.s >= ln - 1e-9:
            if self.leg + 1 < len(self.legs):
                self.s -= ln
                self.leg += 1
            else:
                self.s, done = ln, True
        a, b, _, _ = self.legs[self.leg]
        ln = float(np.linalg.norm(b - a))
        pref = a + (b - a) * (self.s / max(ln, 1e-9))
        vref = (b - a) / max(ln, 1e-9) * self.vel
        return pref, vref, done

    def act(self, obs: Obs):
        spec, q, qd = self.ctx.robot, obs.q, obs.qd
        J = jacobian(spec, q)
        tip, vel = fk(spec, q), J @ qd
        pose = np.asarray(obs.extra["t_pose"], float)
        if not self.legs:
            self.buf.append(pose)                                          # block at rest during the hold: collect samples
        if not self.legs and self.hold <= 0:
            self.hold = 0
            pose = self._filtered_pose(pose)
            if not self._plan(obs, tip, pose):                                  # nothing feasible: retreat from the block, then try again
                away = tip - pose[:2]
                end = np.clip(tip + 0.06 * away / max(np.linalg.norm(away), 1e-9), [BOX_X[0], BOX_Z[0]], [BOX_X[1], BOX_Z[1]])
                self.legs, self.leg, self.s, self.vel = [(tip.copy(), end, self.v_app, False)], 0, 0.0, 0.0
                self.pose_at_plan, self.k_plan = pose.copy(), obs.k
                self.retreats += 1
                self.buf = []
        if self.legs and obs.k - self.k_plan > int(10.0 / self.ctx.dt):      # safety net: a plan never lives longer than 10 s
            self.legs, self.hold_pt, self.hold = [], tip.copy(), 1
        if self.legs:
            pref, vref, done = self._advance(tip)
            if done:
                self.hold_pt, self.legs, self.hold = pref.copy(), [], self.hold_n
                self.buf = []
        else:
            pref, vref = self.hold_pt if self.hold_pt is not None else tip, np.zeros(2)
            self.hold = max(self.hold - 1, 0)
        a = self.kp * (pref - tip) + self.kd * (vref - vel)
        qdd = np.linalg.pinv(J, rcond=1e-3) @ (a - jdot_qd(spec, q, qd))
        return np.clip(self.arm.inverse(q, qd, qdd), -self.ctx.tau_limit, self.ctx.tau_limit)


@register_controller("push_t_random")
class PushTRandom(PushTTeacher):
    """Baseline: same primitives, same approach paths and low-level control, but the primitive is drawn at random (no model-based search).
    It shows how much of the teacher's success comes from the planner and not from the controller / the primitive set."""

    def _rank(self, pose):
        idx = self.rng.permutation(len(self.planner.D))
        return [(int(i), 0.0) for i in idx]
