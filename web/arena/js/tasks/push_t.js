// Task plugin: Push-T (2-link arm pushes a T-shaped block onto a goal pose).
// JS port of arena/tasks/push_t.py (plant, coverage, success) and arena/controllers/pusht_teacher.py (quasi-static push planner with beam search
// + visibility-graph approach + operational-space computed torque). The planner is ported in full and runs in the browser (about 0.1-0.5 s per
// plan, once per push), nothing is precomputed. Geometry and planner: ./push_t_plan.js.
import { ArmModel } from "../sim.js";
import { clip, rng, loadXml, fk2, jac2, jdotqd2, inv2, mulv2 } from "./common.js";
import * as G from "./push_t_plan.js";

const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

/** damped least-squares IK of arena/robots/kinematics.py (n = 2, no posture term) */
export function ik2(L, target, q0) {
  const q = q0.slice();
  for (let it = 0; it < 200; it++) {
    const x = fk2(L, q), e = [target[0] - x[0], target[1] - x[1]], J = jac2(L, q);
    const a = J[0][0] * J[0][0] + J[0][1] * J[0][1] + 1e-6, b = J[0][0] * J[1][0] + J[0][1] * J[1][1], d = J[1][0] * J[1][0] + J[1][1] * J[1][1] + 1e-6;
    const det = a * d - b * b, s0 = (d * e[0] - b * e[1]) / det, s1 = (-b * e[0] + a * e[1]) / det;
    const dq = [J[0][0] * s0 + J[1][0] * s1, J[0][1] * s0 + J[1][1] * s1];
    q[0] += dq[0]; q[1] += dq[1];
    if (Math.hypot(...e) < 1e-12 && Math.hypot(...dq) < 1e-10) break;
  }
  return q;
}

const BOX_X = [0.30, 0.92], BOX_Z = [-0.34, 0.34];       // the pusher stays inside this box
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

class PushTTeacher {
  constructor(arm, cfg, random = false) {
    const t = cfg.teacher;
    this.arm = arm; this.cfg = cfg; this.random = random;
    this.L = cfg.link_lengths; this.goal = cfg.goal; this.rp = cfg.pusher_radius; this.tauLimit = cfg.tau_max; this.dt = cfg.dt;
    this.planner = new G.PushPlanner(cfg.mu, G.RHO2, this.rp, this.goal, t.depth, t.beam);
    Object.assign(this, { kp: t.kp, kd: t.kd, vApp: t.v_approach, vPush: t.v_push, aMax: t.a_max, nTry: t.n_try, travelW: t.travel_w, holdN: Math.round(t.hold_s / cfg.dt) });
    this.rand = rng(12345);
    this.reset();
  }
  reset(seed = 0) {
    this.legs = []; this.leg = 0; this.s = 0; this.vel = 0;
    this.hold = 0; this.poseAtPlan = null; this.kPlan = 0; this.holdPt = null; this.retreats = 0; this.buf = [];
    this.planLog = [];
    this.rand = rng(seed * 131 + 7);
  }
  setParam(id, v) { if (id === "depth") this.planner.depth = v | 0; else this[id] = v; }
  phase() { return this.legs.length ? (this.legs[this.leg][3] ? "pushing" : "approach") : this.hold > 0 ? "settling" : "planning"; }
  _rank(pose) {
    if (!this.random) return this.planner.plan(pose);
    const idx = Array.from({ length: this.planner.M }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(this.rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    return idx.map((i) => [i, 0]);
  }
  _filtered(pose) {
    if (this.buf.length <= 12) return pose;
    const b = this.buf.slice(10), n = b.length;
    return [b.reduce((s, p) => s + p[0], 0) / n, b.reduce((s, p) => s + p[1], 0) / n,
      Math.atan2(b.reduce((s, p) => s + Math.sin(p[2]), 0) / n, b.reduce((s, p) => s + Math.cos(p[2]), 0) / n)];
  }
  _plan(k, tip, pose) {
    const cands = [];
    for (const [i, cost] of this._rank(pose).slice(0, this.nTry)) {
      const { s, u, dist, nout } = this.planner.worldPrimitive(pose, i);
      const gate = [s[0] + nout[0] * 0.06, s[1] + nout[1] * 0.06], end = [s[0] + u[0] * (dist + 0.004), s[1] + u[1] * (dist + 0.004)];
      if (![s, gate, end].every((p) => BOX_X[0] < p[0] && p[0] < BOX_X[1] && BOX_Z[0] < p[1] && p[1] < BOX_Z[1])) continue;
      const path = G.approachPath(tip, s, gate, pose, this.rp);
      if (!path) continue;
      let length = 0;
      for (let j = 0; j + 1 < path.length; j++) length += dist2(path[j + 1], path[j]);
      cands.push([cost + this.travelW * length, i, path, end, cost]);
    }
    if (!cands.length) return false;
    cands.sort((a, b) => a[0] - b[0]);
    const [, i, path, end, cost] = cands[0];
    this.legs = [];
    for (let j = 0; j + 1 < path.length; j++) this.legs.push([path[j], path[j + 1], this.vApp, false]);
    this.legs.push([path[path.length - 1], end, this.vPush, true]);
    this.leg = 0; this.s = 0; this.vel = 0; this.poseAtPlan = pose.slice(); this.kPlan = k; this.buf = [];
    this.planLog.push({ k, pose: pose.slice(), cost, prim: i, now: G.poseCost(pose, this.goal) });
    return true;
  }
  _advance(tip) {
    let [a, b, spd, isPush] = this.legs[this.leg];
    let ln = dist2(b, a);
    const rem = ln - this.s, nxt = this.leg + 1 < this.legs.length ? this.legs[this.leg + 1][2] : 0;
    let vLim = Math.sqrt(nxt * nxt + 2 * this.aMax * Math.max(rem, 0));
    if (isPush) vLim = rem > 0 ? spd : 0;
    const f = this.s / Math.max(ln, 1e-9), pref0 = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    const lag = dist2(pref0, tip), gate = Math.min(1, Math.max(0.05, 1.0 - (lag - 0.01) / 0.03));
    const vGoal = Math.min(spd, vLim) * gate, dt = this.dt;
    this.vel = Math.min(Math.max(vGoal, this.vel - this.aMax * dt), this.vel + this.aMax * dt);
    this.s += this.vel * dt;
    let done = false;
    if (this.s >= ln - 1e-9) {
      if (this.leg + 1 < this.legs.length) { this.s -= ln; this.leg += 1; } else { this.s = ln; done = true; }
    }
    [a, b] = this.legs[this.leg];
    ln = dist2(b, a);
    const g = this.s / Math.max(ln, 1e-9);
    return { pref: [a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g], vref: [((b[0] - a[0]) / Math.max(ln, 1e-9)) * this.vel, ((b[1] - a[1]) / Math.max(ln, 1e-9)) * this.vel], done };
  }
  act(obs) {
    const { q, qd } = obs, J = jac2(this.L, q), tip = fk2(this.L, q), vel = mulv2(J, qd);
    let pose = obs.extra.t_pose.slice();
    if (!this.legs.length) this.buf.push(pose);
    if (!this.legs.length && this.hold <= 0) {
      this.hold = 0;
      pose = this._filtered(pose);
      if (!this._plan(obs.k, tip, pose)) {                          // nothing feasible: retreat from the block, then plan again
        const away = [tip[0] - pose[0], tip[1] - pose[1]], an = Math.max(Math.hypot(...away), 1e-9);
        const end = [Math.min(Math.max(tip[0] + (0.06 * away[0]) / an, BOX_X[0]), BOX_X[1]), Math.min(Math.max(tip[1] + (0.06 * away[1]) / an, BOX_Z[0]), BOX_Z[1])];
        this.legs = [[tip.slice(), end, this.vApp, false]]; this.leg = 0; this.s = 0; this.vel = 0;
        this.poseAtPlan = pose.slice(); this.kPlan = obs.k; this.retreats += 1; this.buf = [];
      }
    }
    if (this.legs.length && obs.k - this.kPlan > Math.trunc(10.0 / this.dt)) { this.legs = []; this.holdPt = tip.slice(); this.hold = 1; }   // a plan never lives longer than 10 s
    let pref, vref;
    if (this.legs.length) {
      const r = this._advance(tip);
      pref = r.pref; vref = r.vref;
      if (r.done) { this.holdPt = pref.slice(); this.legs = []; this.hold = this.holdN; this.buf = []; }
    } else {
      pref = this.holdPt ?? tip; vref = [0, 0];
      this.hold = Math.max(this.hold - 1, 0);
    }
    const a = [this.kp * (pref[0] - tip[0]) + this.kd * (vref[0] - vel[0]), this.kp * (pref[1] - tip[1]) + this.kd * (vref[1] - vel[1])];
    const jd = jdotqd2(this.L, q, qd);
    const qdd = mulv2(inv2(J), [a[0] - jd[0], a[1] - jd[1]]);
    return this.arm.inverse(q, qd, qdd).map((t) => clip(t, this.tauLimit));
  }
}

export class PushTSession {
  constructor(mj, assets) {
    this.mj = mj;
    this.template = assets.xml;
    this.cfg = assets.cfg;
    const c = this.cfg;
    Object.assign(this, { dt: c.dt, substeps: c.substeps, tauMax: c.tau_max, duration: c.duration, nSteps: Math.round(c.duration / c.dt), n: c.n });
    this.arm = new ArmModel(mj, assets.nominal);
    this.weak = { delay: 0, noise: 0, friction: 0, mass: 0, comx: 0, comz: 0 };
    this.opt = { start: "random (seed)" };
    this.dragPose = null;
    this.model = this.data = null;
    this.controllers = { teacher: new PushTTeacher(this.arm, c), random: new PushTTeacher(this.arm, c, true) };
    this.active = "teacher";
    this._build();
    this.reset(1000);
  }
  // true plant parameters from the weakness settings (arena/perturb/whip_pusht.py: TFriction, TMass, TComOffset)
  plant() {
    const c = this.cfg, w = this.weak;
    return { mu: c.mu * (1 + w.friction / 100), b: c.table_b * (1 + w.friction / 100), m: c.t_mass * (1 + w.mass / 100), com: [w.comx / 1000, w.comz / 1000] };
  }
  /** arena/tasks/push_t.py tip_xml, plus a massless sphere that only draws the pusher (the app does not draw cylinders) */
  _xml() {
    const c = this.cfg, { mu, b, m, com } = this.plant(), rho2 = G.RHO2;
    const pusher = `<geom name="pusher" type="cylinder" size="${c.pusher_radius} 0.03" euler="90 0 0" mass="0.05" rgba="0.9 0.35 0.2 1" contype="1" conaffinity="1" friction="0.001 0.005 0.0001" solref="0.005 1"/>` +
      `<geom name="pusher_vis" type="sphere" size="0.03" pos="0 -0.05 0" mass="0" rgba="0.9 0.35 0.2 1" contype="0" conaffinity="0"/>`;
    const fl = c.coulomb_frac * mu * m * c.g, [gx, gz, gth] = c.goal, degr = (gth * 180) / Math.PI;
    let tb = `<body name="tbody" gravcomp="1"><joint name="tx" type="slide" axis="1 0 0" damping="${b}" frictionloss="${fl}"/>` +
      `<joint name="tz" type="slide" axis="0 0 1" damping="${b}" frictionloss="${fl}"/>` +
      `<joint name="tth" type="hinge" axis="0 1 0" damping="${b * rho2}" frictionloss="${fl * Math.sqrt(rho2)}"/>` +
      `<inertial pos="0 0 0" mass="${m}" diaginertia="${m * rho2} ${m * rho2} ${m * rho2}"/>`;
    G.RECTS.forEach(([cu, cv, hu, hv], i) => {
      tb += `<geom name="${["tbar", "tstem"][i]}" type="box" size="${hu} 0.03 ${hv}" pos="${cu - com[0]} 0 ${cv - com[1]}" rgba="0.25 0.45 0.85 1" contype="1" conaffinity="1" friction="${mu} 0.005 0.0001" solref="0.005 1"/>`;
    });
    tb += "</body>";
    const goal = G.RECTS.map(([cu, cv, hu, hv], i) =>
      `<geom name="goal_${["bar", "stem"][i]}" type="box" size="${hu} 0.002 ${hv}" rgba="0.3 0.8 0.35 0.55" contype="0" conaffinity="0" ` +
      `pos="${gx + Math.cos(gth) * cu + Math.sin(gth) * cv} 0.036 ${gz - Math.sin(gth) * cu + Math.cos(gth) * cv}" euler="0 ${degr} 0"/>`).join("");
    const world = '<geom name="table" type="box" size="0.55 0.004 0.40" pos="0.62 0.045 0" rgba="0.82 0.78 0.68 1" contype="0" conaffinity="0"/>' + goal + tb;
    return this.template.replace("@@PUSHER@@", pusher).replace("@@WORLD@@", world);
  }
  _build() {
    const old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel), pose: this.shapePose() } : null;
    this.data?.delete();
    this.model?.delete();
    this.model = loadXml(this.mj, this._xml(), "/pusht_plant.xml");
    this.data = new this.mj.MjData(this.model);
    const id = (type, nm) => this.mj.mj_name2id(this.model, type.value, nm);
    this.tbody = id(this.mj.mjtObj.mjOBJ_BODY, "tbody");
    this.geomIds = Object.fromEntries(["pusher_vis", "table", "goal_bar", "goal_stem", "tbar", "tstem"].map((nm) => [id(this.mj.mjtObj.mjOBJ_GEOM, nm), nm]));
    if (old) {
      old.q.forEach((x, i) => (this.data.qpos[i] = x));
      old.v.forEach((x, i) => (this.data.qvel[i] = x));
      this._setBlock(old.pose);                             // body frame = COM frame: re-express the shape pose with the new COM offset
    }
    this.mj.mj_forward(this.model, this.data);
  }
  _setBlock(pose) {
    const n = this.n, c = this.plant().com, r = G.rotv(pose[2], c);
    this.data.qpos[n] = pose[0] + r[0]; this.data.qpos[n + 1] = pose[1] + r[1]; this.data.qpos[n + 2] = pose[2];
  }
  shapePose() {
    const n = this.n, q = this.data.qpos, th = q[n + 2], r = G.rotv(th, this.plant().com);
    return [q[n] - r[0], q[n + 1] - r[1], th];
  }
  // ---- plugin interface ----
  setController(id) { this.active = id; }
  setParam(ctl, id, v) { this.controllers[ctl].setParam(id, v); }
  setWeakness(id, v) {
    this.weak[id] = v;
    if (id === "friction" || id === "mass" || id === "comx" || id === "comz") this._build();
  }
  setOption(id, v) { this.opt[id] = v; }
  controllerInfo() { return ""; }
  /** initial block pose and pusher position from the seed (own JS RNG, same distribution as Python _sample), or from the pose the user dropped */
  _sample(seed, dropped) {
    const c = this.cfg, goal = c.goal, r = rng(seed * 2654435761 + 7701);
    const u = (a) => (2 * r() - 1) * a;
    let pose;
    if (dropped) pose = [Math.min(Math.max(dropped[0], 0.34), 0.86), Math.min(Math.max(dropped[1], -0.26), 0.26), dropped[2]];
    else {
      for (let i = 0; i < 500; i++) {
        pose = [goal[0] + u(0.10), goal[1] + u(0.10), goal[2] + u(c.max_rot)];
        const poly = G.polyWorld(pose);
        if (G.coverage(pose, goal) <= 0.3 && poly.every((p) => p[0] > 0.34 && p[0] < 0.86 && Math.abs(p[1]) < 0.26)) break;
      }
    }
    const poly = G.polyWorld(pose);
    let p;
    for (let i = 0; i < 500; i++) {
      p = [0.40 + 0.45 * r(), u(0.25)];
      if (G.closestOnPolygon(p, poly).d > 0.10 && !G.insidePolygon(G.rotT(pose[2], [p[0] - pose[0], p[1] - pose[1]]), G.POLY)) break;
    }
    return { pose, pusher: p };
  }
  /** `init` = {q:[q1,q2,x,z,th], qd?} (validation: exact Python initial state; q[2:4] is the COM position) */
  reset(seed, init = null) {
    const c = this.cfg, n = this.n;
    this.seed = seed;
    this.mj.mj_resetData(this.model, this.data);
    if (init) {
      init.q.forEach((x, i) => (this.data.qpos[i] = x));
    } else {
      const s = this._sample(seed, this.opt.start.startsWith("where") ? this.dragPose : null);
      const q = ik2(c.link_lengths, s.pusher, [0.4, 1.4]);
      this.data.qpos[0] = q[0]; this.data.qpos[1] = q[1];
      this._setBlock(s.pose);
    }
    this.data.xfrc_applied.fill(0);
    this.mj.mj_forward(this.model, this.data);
    this.k = 0;
    this.done = false;
    this.outcome = null;
    this.message = "";
    this.recordNow = true;
    this.cov = this.covNow();
    this.maxCov = this.cov;
    this.moved = false;
    this.lastTau = null;
    this.hist = [];
    this.noiseRng = rng(seed * 977 + 4242);
    this.pathLen = 0;
    this.prevTip = this.tipPos();
    Object.values(this.controllers).forEach((x) => x.reset(seed));
    this._observe();
  }
  covNow() { return G.coverage(this.shapePose(), this.cfg.goal); }
  tipPos() { return fk2(this.cfg.link_lengths, [this.data.qpos[0], this.data.qpos[1]]); }
  _observe() {
    const d = this.data, w = this.weak;
    let pose = this.shapePose();
    if (w.noise > 0) {
      const sg = w.noise / 1000, sa = 0.0025 * w.noise;     // sigma mm on x, z; 0.0025 rad per mm on the angle
      pose = [pose[0] + sg * gauss(this.noiseRng), pose[1] + sg * gauss(this.noiseRng), pose[2] + sa * gauss(this.noiseRng)];
    }
    const cur = { q: [d.qpos[0], d.qpos[1]], qd: [d.qvel[0], d.qvel[1]], pose };
    this.hist.push(cur);                                    // Delay(where="obs"): q, qd and the block pose are `steps` old; until full the first value is held
    const old = this.hist[Math.max(0, this.hist.length - 1 - w.delay)];
    if (this.hist.length > w.delay + 2) this.hist.shift();
    this.obs = { k: this.k, q: old.q, qd: old.qd, extra: { t_pose: old.pose } };
  }
  get ctrl() { return this.controllers[this.active]; }
  step(hook) {
    let tau;
    if (this.done) {                                       // episode over: the arm just comes to rest
      const q = [this.data.qpos[0], this.data.qpos[1]], qd = [this.data.qvel[0], this.data.qvel[1]];
      tau = this.arm.inverse(q, qd, qd.map((v) => -20 * v)).map((t) => clip(t, this.tauMax));
    } else tau = this.ctrl.act(this.obs).map((t) => clip(t, this.tauMax));
    const d = this.data, n = this.n, tb = this.tbody;
    d.ctrl[0] = tau[0]; d.ctrl[1] = tau[1];
    let grabbed = false;
    for (let s = 0; s < this.substeps; s++) {
      if (hook) hook(this);
      if (d.xfrc_applied[6 * tb] !== 0 || d.xfrc_applied[6 * tb + 2] !== 0) grabbed = true;
      this.mj.mj_step(this.model, d);
    }
    this.k += 1;
    this.lastTau = tau;
    this.recordNow = !this.done;
    if (grabbed) { this.moved = true; this.dragPose = this.shapePose(); }
    const tip = this.tipPos();
    if (!this.done) {
      this.pathLen += dist2(tip, this.prevTip);
      this.cov = this.covNow();
      this.maxCov = Math.max(this.maxCov, this.cov);
      const p = this.shapePose(), c = this.cfg;
      const out = !(c.x_range[0] < p[0] && p[0] < c.x_range[1] && c.z_range[0] < p[1] && p[1] < c.z_range[1]);
      const by = this.moved ? " (you moved the block)" : "";
      if (out && this.cov < c.cov_thr) { this.done = true; this.outcome = "failure"; this.message = `The T left the work area${by}`; }
      else if (this.cov >= c.cov_thr) { this.done = true; this.outcome = "success"; this.message = `Success: coverage ${(this.cov * 100).toFixed(0)} % in ${(this.k * this.dt).toFixed(1)} s${by}`; }
      else if (this.k >= this.nSteps) { this.done = true; this.outcome = "failure"; this.message = `Timeout: coverage ${(this.cov * 100).toFixed(0)} % (needs ${(c.cov_thr * 100).toFixed(0)} %)${by}`; }
    }
    this.prevTip = tip;
    this._observe();
    return tau;
  }
  // ---- display ----
  chartValue() { return this.cov; }
  refPath() { return null; }
  target() { return this.cfg.goal.slice(0, 2); }
  trailPoint() { return this.tipPos(); }
  metrics() {
    const pct = (x) => `${(x * 100).toFixed(0)} %`;
    return [
      ["time", `${(this.k * this.dt).toFixed(1)} s`],
      ["coverage", pct(this.cov)],
      ["best coverage", pct(this.maxCov)],
      ["pushes planned", String(this.ctrl.planLog.length)],
      ["phase", this.done ? "-" : this.ctrl.phase()],
      ["result", this.done ? (this.outcome === "success" ? "SUCCESS" : "failed") : "running"],
    ];
  }
  geomColor(i) {
    switch (this.geomIds[i]) {
      case "pusher_vis": return 0xe8590c;
      case "table": return 0xddd5c0;
      case "goal_bar": case "goal_stem": return 0x7bd88f;
      case "tbar": case "tstem": return 0x3b6fd8;
      default: return this.model.geom_bodyid[i] === this.n + 1 && this.model.geom_type[i] === this.mj.mjtGeom.mjGEOM_SPHERE.value ? 0x6b7280 : null;   // tip payload
    }
  }
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]], pose = this.shapePose();
    const w = (u, v) => { const r = G.rotv(pose[2], [u, v]); return [pose[0] + r[0], pose[1] + r[1]]; };
    const T = { fmax: 14, k: 160, damp: 12 };               // the block weighs 0.4 kg and has table friction: moderate pulls
    return [
      { body: 1, a: p(1), b: p(2) },
      { body: 2, a: p(2), b: p(3) },
      { body: this.tbody, a: w(-G.W / 2, G.V_BAR), b: w(G.W / 2, G.V_BAR), ...T },
      { body: this.tbody, a: w(0, G.V_BAR), b: w(0, G.V_STEM - G.L / 2), ...T },
    ];
  }
  dispose() { this.data?.delete(); this.model?.delete(); this.arm.data.delete(); this.arm.model.delete(); }
}

const fmtPct = (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %";
const fmtMm = (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " mm";

export default {
  id: "push_t",
  title: "Push-T",
  summary:
    "A 2-link arm moves a round pusher on a table (seen from above) and has to push a T-shaped block onto the green goal pose; success is 90 % overlap with the goal. " +
    "The classical teacher plans pushes with a quasi-static friction model (beam search over contact point, direction and length), drives around the block, pushes, looks again and re-plans. " +
    "The whole planner runs in your browser. Drag the T with the mouse to change where it starts, or try a delayed or noisy camera.",
  assets: { xml: "assets/push_t.xml", nominal: "assets/push_t_nominal.xml", cfg: "assets/push_t.json" },
  camera: { target: [0.6, 0], distance: 2.0, fov: 32 },
  chart: { label: "coverage of the goal (%)", scale: 100, signed: false, fixedMax: 100, guides: [90], guideLabel: "success at 90 %" },
  legendNote: "orange ball = pusher, blue = T block, green = goal pose (red dot = its centroid), black line = pusher path; coverage = overlap area / goal area",
  interaction: { fmax: 40, k: 200, damp: 15, hitRadius: 0.07 },
  controllers: [
    {
      id: "teacher", label: "Push planner", color: "#2563eb", name: "Quasi-static push planner + OSC",
      note: "Beam search over push primitives on an ellipsoidal limit-surface model (knows the nominal friction coefficient 0.5, not the table friction, mass or centre of mass), visibility-graph approach, push at constant speed, rest 0.2 s, re-plan from the filtered block pose. Operational-space computed torque on the arm.",
      params: [
        { id: "vPush", label: "push speed", min: 0.03, max: 0.2, step: 0.01, value: 0.08, fmt: (v) => v.toFixed(2) + " m/s" },
        { id: "depth", label: "planning depth", min: 1, max: 4, step: 1, value: 4, fmt: (v) => v.toFixed(0) + " pushes" },
        { id: "kp", label: "tip stiffness kp", min: 200, max: 2000, step: 50, value: 900, fmt: (v) => v.toFixed(0) },
      ],
    },
    {
      id: "random", label: "Random push", color: "#dc2626", name: "Random push primitive (no search)",
      note: "Baseline: same approach paths and low-level control, but the push is drawn at random among the feasible primitives instead of searched with the model.",
      params: [],
    },
  ],
  weaknesses: [
    { id: "delay", label: "sensor delay (q, qd, block pose)", min: 0, max: 10, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
    { id: "noise", label: "camera noise (block pose)", min: 0, max: 10, step: 1, value: 0, fmt: (v) => `${v.toFixed(0)} mm / ${(v * 0.25).toFixed(1)} deg` },
    { id: "friction", label: "true friction (pusher + table)", min: -50, max: 100, step: 10, value: 0, fmt: fmtPct },
    { id: "mass", label: "true block mass", min: -50, max: 200, step: 10, value: 0, fmt: fmtPct },
    { id: "comx", label: "block centre of mass x", min: -30, max: 30, step: 5, value: 0, fmt: fmtMm },
    { id: "comz", label: "block centre of mass z", min: -30, max: 30, step: 5, value: 0, fmt: fmtMm },
  ],
  weaknessNote:
    "Weaknesses change the plant or the sensors; the planner always assumes the nominal block. In Python the teacher succeeds on 20/20 seeds nominally and on 0/20 with a 30 ms (6-step) sensor delay, " +
    "while friction, mass, centre of mass and noise barely hurt it. Drag the blue T (or the arm) with the mouse; then set \"restart from\" to \"where I dropped the T\" to start the next episodes from that pose.",
  options: [{ id: "start", label: "restart from", choices: ["random (seed)", "where I dropped the T"], value: "random (seed)" }],
  create: (mj, assets) => new PushTSession(mj, assets),
};
