// Task plugin: toss (the arm holds a ball and must throw it into a basket, some baskets beyond its reach).
// JS port of arena/tasks/toss.py and arena/controllers/toss_teacher.py (ballistic inversion -> release state -> joint quintic -> computed torque).
import { ArmModel, lqrDoubleIntegrator } from "../sim.js";
import { clip, rng, loadXml } from "./common.js";
import * as bal from "./_ball_common.js";
import { clamp, withBall, rgbaColor } from "./_ball_common.js";

const T_GRID = Array.from({ length: 1500 }, (_, i) => i * 0.002);   // np.arange(0, 3, 0.002)
const maxAbs = (a) => a.reduce((m, x) => Math.max(m, Math.abs(x)), 0);

/** x where the flight from p0 with v0 crosses z = zRim while falling (NaN if it never does); last crossing, as in Python */
function crossX(p0, v0, k, g, zRim) {
  let last = -1, zPrev = 0, zi = 0, zi1 = 0;
  for (let i = 0; i < T_GRID.length; i++) {
    const z = bal.position(T_GRID[i], p0, v0, k, g)[1] - zRim;
    if (i > 0 && zPrev >= 0 && z < 0) { last = i - 1; zi = zPrev; zi1 = z; }
    zPrev = z;
  }
  if (last < 0) return NaN;
  const f = zi / (zi - zi1), xa = bal.position(T_GRID[last], p0, v0, k, g)[0], xb = bal.position(T_GRID[last + 1], p0, v0, k, g)[0];
  return xa + f * (xb - xa);
}

/** launch speed s at angle theta so that the drag-ballistic flight from p0 crosses the rim plane at targetX (null if none in [sLo, sHi]) */
export function solveLaunch(p0, theta, targetX, zRim, k, g, sLo = 0.5, sHi = 14) {
  const d = [Math.cos(theta), Math.sin(theta)];
  const f = (s) => { const x = crossX(p0, [s * d[0], s * d[1]], k, g, zRim); return Number.isFinite(x) ? x - targetX : -10; };
  const a = f(sLo), b = f(sHi);
  if (a * b > 0) return null;
  try { return bal.brentq(f, sLo, sHi, 1e-4); } catch { return null; }
}

/** Classical toss teacher (nominal knowledge: gravity, drag rate, ball mass, scheduled release time, the basket from the reference) */
export class TossTeacher {
  constructor(hold, free, cfg, params) {
    Object.assign(this, { hold, free, cfg, L: cfg.link_lengths, tauLimit: cfg.tau_max });
    this.p = { ...params };
    this._gains();
    this.reset();
  }
  _gains() { [this.Kp, this.Kd] = lqrDoubleIntegrator(this.p.qp, this.p.qv); }
  /** gains act at once; brake_T and the release-delay compensation shape the plan and apply from the next throw */
  setParam(id, v) { this.p[id] = v; if (id === "qp" || id === "qv") this._gains(); }
  get tr() { return this.cfg.release_time + this.p.comp_ms / 1000; }
  reset() { this.seg1 = this.seg2 = null; this.info = {}; this.tPlan = this.tr; this.brakeT = this.p.brake_T; }
  _plan(q0, basket) {
    const [xb, zb] = basket, zRim = zb + this.cfg.basket_h, tr = this.tr, lim = this.tauLimit, k = this.cfg.drag_rate, g = this.cfg.gravity;
    const ts = Array.from({ length: 14 }, (_, i) => (i === 13 ? tr : (i * tr) / 13));
    let best = null;
    for (const R of [0.5, 0.65, 0.8, 0.9]) {
      for (const aDeg of [-10, 10, 30, 50, 70]) {
        const al = (aDeg * Math.PI) / 180, pr = [R * Math.cos(al), R * Math.sin(al)];
        for (const thDeg of [30, 40, 50, 60, 70]) {
          const th = (thDeg * Math.PI) / 180;
          const s = solveLaunch(pr, th, xb, zRim, k, g);
          if (s === null) continue;
          const vr = [s * Math.cos(th), s * Math.sin(th)];
          for (const phi of [-0.9, -0.45, 0.0, 0.45, 0.9]) {
            const qr = bal.levelIk(this.L, pr, phi);
            if (qr === null) continue;
            const [jx, jz] = bal.jac3(this.L, qr);
            const JJt = [[0, 0], [0, 0]];
            for (let i = 0; i < 3; i++) { JJt[0][0] += jx[i] * jx[i]; JJt[0][1] += jx[i] * jz[i]; JJt[1][1] += jz[i] * jz[i]; }
            JJt[1][0] = JJt[0][1];
            const y = bal.solve(JJt, [[vr[0]], [vr[1]]]).map((r) => r[0]);
            const qdr = [0, 1, 2].map((i) => jx[i] * y[0] + jz[i] * y[1]);
            const seg = bal.quintic(0, [q0, [0, 0, 0], [0, 0, 0]], tr, [qr, qdr, [0, 0, 0]]);
            let tauPk = 0, qdPk = 0;
            for (const t of ts) {
              const [Q, Qd, Qdd] = seg(t);
              tauPk = Math.max(tauPk, maxAbs(this.hold.inverse(Q, Qd, Qdd)));
              qdPk = Math.max(qdPk, maxAbs(Qd));
            }
            const cost = tauPk / lim + 0.01 * qdPk;
            if (best === null || cost < best[0]) best = [cost, seg, qr, qdr, { p_r: pr, theta: th, speed: s, phi, tau_ratio: tauPk / lim }];
          }
        }
      }
    }
    return best;
  }
  _start(obs) {
    const best = this._plan(obs.q, obs.basket);
    this.tPlan = this.tr; this.brakeT = this.p.brake_T;
    if (best === null) { const qh = obs.q.slice(); this.seg1 = () => [qh, [0, 0, 0], [0, 0, 0]]; this.seg2 = null; return; }
    const [, seg1, qr, qdr, info] = best;
    this.seg1 = seg1; this.info = info;
    const stop = qr.map((q, i) => q + 0.4 * qdr[i] * this.brakeT);
    this.seg2 = bal.quintic(this.tPlan, [qr, qdr, [0, 0, 0]], this.tPlan + this.brakeT, [stop, [0, 0, 0], [0, 0, 0]]);
  }
  /** obs = {t, q, qd, basket: [xb, zb]} -> joint torques */
  act(obs) {
    if (this.seg1 === null) this._start(obs);
    const t = obs.t, tr = this.tPlan;
    let q, qd, qdd;
    if (t <= tr + 1e-9 || this.seg2 === null) [q, qd, qdd] = this.seg1(Math.min(t, tr));
    else {
      [q, qd, qdd] = this.seg2(t);
      if (t > tr + this.brakeT) { qd = [0, 0, 0]; qdd = [0, 0, 0]; }
    }
    const model = t < tr ? this.hold : this.free;
    const v = [0, 1, 2].map((j) => qdd[j] + this.Kp * (q[j] - obs.q[j]) + this.Kd * (qd[j] - obs.qd[j]));
    return model.inverse(obs.q, obs.qd, v).map((x) => clip(x, this.tauLimit));
  }
}

const DEFAULTS = { qp: 1.6e5, qv: 800, brake_T: 0.6, comp_ms: 0 };

export class TossSession {
  constructor(mj, assets) {
    this.mj = mj; this.xml = assets.xml; this.cfg = assets.cfg;
    const c = this.cfg;
    Object.assign(this, { dt: c.dt, substeps: c.substeps, tauMax: c.tau_max, duration: c.duration, nSteps: Math.round(c.duration / c.dt) });
    this.hold = new ArmModel(mj, assets.hold);
    this.free = new ArmModel(mj, assets.free);
    this.teacher = new TossTeacher(this.hold, this.free, c, DEFAULTS);
    this.mass = c.ball_mass; this.drag = c.drag; this.releaseDelay = 0;
    this.basketMode = "random";
    this.hits = 0; this.attempts = 0;
    this.model = this.data = null;
    this.basket = [1.0, -0.3];
    this._build();
    this.active = "teacher";
    this.reset(1000);
  }
  _build() {
    const old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel), eq: this.data.eq_active[0] } : null;
    this.data?.delete(); this.model?.delete();
    this.model = loadXml(this.mj, withBall(this.xml, "obj", ["ox", "oz"], this.mass, this.drag), "/plant.xml");
    this.data = new this.mj.MjData(this.model);
    const id = (type, n) => this.mj.mj_name2id(this.model, type.value, n);
    this.tipSite = id(this.mj.mjtObj.mjOBJ_SITE, "tip");
    this.objBody = id(this.mj.mjtObj.mjOBJ_BODY, "obj");
    this.gFloor = id(this.mj.mjtObj.mjOBJ_GEOM, "basket_floor");
    this.gWall = [id(this.mj.mjtObj.mjOBJ_GEOM, "basket_wall0"), id(this.mj.mjtObj.mjOBJ_GEOM, "basket_wall1")];
    this._placeBasket();
    if (old) {
      old.q.forEach((x, i) => (this.data.qpos[i] = x)); old.v.forEach((x, i) => (this.data.qvel[i] = x));
      this.data.eq_active[0] = old.eq;
      this.mj.mj_forward(this.model, this.data);
    }
  }
  _placeBasket() {
    const [xb, zb] = this.basket, hw = this.cfg.basket_hw, h = this.cfg.basket_h, w = 0.016, gp = this.model.geom_pos;
    const set = (g, x, y, z) => { gp[3 * g] = x; gp[3 * g + 1] = y; gp[3 * g + 2] = z; };
    set(this.gFloor, xb, 0, zb - w / 2);
    [-1, 1].forEach((s, i) => set(this.gWall[i], xb + s * (hw + w / 2), 0, zb + h / 2));
  }
  // ---- plugin interface ----
  setController() {}
  setParam(ctl, id, v) { this.teacher.setParam(id, v); }
  setWeakness(id, v) {
    if (id === "mass") { this.mass = this.cfg.ball_mass * (1 + v / 100); this._build(); }
    else if (id === "drag") { this.drag = this.cfg.drag * (1 + v / 100); this._build(); }
    else if (id === "release") this.releaseDelay = v / 1000;
  }
  setOption(id, v) { if (id === "basket") this.basketMode = v.startsWith("near") ? "near" : v.startsWith("far") ? "far" : "random"; }
  controllerInfo() { return "The release speed, angle and posture are chosen at the first step from the basket position (red dot)."; }
  /** basket and posture from the seed (JS RNG, same distributions as Python) or explicitly {q: [3 joints + object xz], basket} (validation) */
  reset(seed, init = null) {
    const c = this.cfg, n = 3;
    this.seed = seed;
    this.mj.mj_resetData(this.model, this.data);
    let q;
    if (init) { this.basket = init.basket.slice(); q = init.q; } else {
      const r = rng(seed), u = (a, b) => a + (b - a) * r();
      let xb = u(0.7, 1.8);
      const zb = u(-0.5, -0.1);
      if (this.basketMode === "near") xb = 0.7 + (xb - 0.7) * 0.35;
      else if (this.basketMode === "far") xb = 1.8 - (1.8 - xb) * 0.35;
      this.basket = [xb, zb];
      q = c.q_start.map((x) => x + u(-0.02, 0.02));
    }
    this._placeBasket();
    for (let i = 0; i < n; i++) this.data.qpos[i] = q[i];
    this.data.eq_active[0] = 1;
    this.mj.mj_forward(this.model, this.data);
    const s = this.data.site_xpos, tip = [s[3 * this.tipSite], s[3 * this.tipSite + 2]];
    this.data.qpos[3] = init ? q[3] : tip[0] - c.arm_reach;
    this.data.qpos[4] = init ? q[4] : tip[1];
    this.data.xfrc_applied.fill(0);
    this.mj.mj_forward(this.model, this.data);
    this.k = 0; this.done = false; this.outcome = null; this.message = ""; this.recordNow = true;
    this.released = false; this.landing = null; this.prevObj = null; this.lastTau = null;
    this.teacher.reset();
    this._observe();
  }
  objPos() { return [this.cfg.arm_reach + this.data.qpos[3], this.data.qpos[4]]; }
  objVel() { return [this.data.qvel[3], this.data.qvel[4]]; }
  tipPos() { const s = this.data.site_xpos; return [s[3 * this.tipSite], s[3 * this.tipSite + 2]]; }
  _observe() {
    const d = this.data;
    this.obs = { t: this.k * this.dt, q: [d.qpos[0], d.qpos[1], d.qpos[2]], qd: [d.qvel[0], d.qvel[1], d.qvel[2]], basket: this.basket };
  }
  _inside() {
    const p = this.objPos(), [xb, zb] = this.basket, c = this.cfg;
    return Math.abs(p[0] - xb) < c.basket_hw && zb < p[1] && p[1] < zb + c.basket_h;
  }
  step(hook) {
    const tau = this.teacher.act(this.obs).map((x) => clip(x, this.tauMax));
    const d = this.data;
    for (let i = 0; i < 3; i++) d.ctrl[i] = tau[i];
    for (let s = 0; s < this.substeps; s++) { if (hook) hook(this); this.mj.mj_step(this.model, d); }
    this.k += 1;
    this._observe();
    this.lastTau = tau;
    this.recordNow = !this.done;
    // task events in the order of Toss.terminated / succeeded
    if (!this.released && this.k * this.dt >= this.cfg.release_time + this.releaseDelay - 1e-9) { d.eq_active[0] = 0; this.released = true; }
    const p = this.objPos();
    if (this.released && this.landing === null && this.prevObj) {
      const zr = this.basket[1] + this.cfg.basket_h, a = this.prevObj;
      if (a[1] >= zr && zr > p[1]) { const f = (a[1] - zr) / (a[1] - p[1]); this.landing = Math.abs(a[0] + f * (p[0] - a[0]) - this.basket[0]); }
    }
    this.prevObj = p;
    if (!this.done) {
      const inside = this._inside();
      if (this.released && (p[1] < this.cfg.floor_z || p[0] > 3) && !inside) this._finish("failure", `Missed the basket${this.landing !== null ? ` by ${(this.landing * 100).toFixed(0)} cm` : ""}`);
      else if (inside) this._finish("success", "In the basket");
      else if (this.k >= this.nSteps) this._finish("failure", "Time out");
    }
    return tau;
  }
  _finish(outcome, message) {
    this.done = true; this.outcome = outcome; this.message = message;
    this.attempts += 1; if (outcome === "success") this.hits += 1;
  }
  // ---- display ----
  chartValue() { const p = this.objPos(); return Math.hypot(p[0] - this.basket[0], p[1] - this.basket[1]); }
  refPath() { return null; }
  target() { return this.basket; }
  trailPoint() { return this.objPos(); }
  metrics() {
    const v = this.objVel();
    return [
      ["time", `${(this.k * this.dt).toFixed(2)} s`],
      ["ball", this.released ? "in flight" : "held"],
      ["ball speed", `${Math.hypot(v[0], v[1]).toFixed(1)} m/s`],
      ["basket", `${this.basket[0].toFixed(2)} m away (reach ${this.cfg.arm_reach.toFixed(2)})`],
      ["landing error", this.landing === null ? "-" : `${(this.landing * 100).toFixed(0)} cm`],
      ["hits", `${this.hits} / ${this.attempts}`],
      ["result", this.done ? (this.outcome === "success" ? "SCORED" : "MISSED") : "running"],
    ];
  }
  geomColor(i) { return rgbaColor(this.model, i); }
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]], o = p(this.objBody);
    return [
      { body: 1, a: p(1), b: p(2) }, { body: 2, a: p(2), b: p(3) }, { body: 3, a: p(3), b: p(4) },
      { body: this.objBody, a: o, b: [o[0] + 1e-4, o[1]], fmax: 4, k: 60, damp: 0.6 },
    ];
  }
  dispose() { this.data?.delete(); this.model?.delete(); for (const a of [this.hold, this.free]) { a.data.delete(); a.model.delete(); } }
}

export default {
  id: "toss",
  title: "Toss",
  summary:
    "The arm holds a ball and must throw it into a basket whose position changes every round; far baskets are out of the arm's reach, so it has to throw. " +
    "The classical controller solves the flight backwards (which speed and angle drop the ball into the basket), swings the arm to that release state, and the gripper opens on a fixed schedule. " +
    "Add release lag or a wrong ball mass and watch the throw fall short.",
  assets: { xml: "assets/toss.xml", hold: "assets/toss_nominal_hold.xml", free: "assets/toss_nominal_free.xml", cfg: "assets/toss.json" },
  camera: { target: [0.85, -0.25], distance: 3.7, fov: 32 },
  chart: { label: "ball to basket distance (cm)", scale: 100, signed: false, minRange: 0.5, maxCap: 3.5 },
  legendNote: "red dot = basket centre, orange = ball, black = ball path, blue = basket",
  interaction: { fmax: 120, k: 220, damp: 18, hitRadius: 0.12 },
  controllers: [
    {
      id: "teacher", label: "Invert flight + CTC", color: "#2563eb", name: "Ballistic inversion + joint trajectory + computed torque",
      note: "Searches release points, launch angles and arm postures for the one that needs the least torque, then follows a smooth joint trajectory to that release state with computed torque. Stiffness acts at once; braking time and the release-lag compensation apply from the next throw.",
      params: [
        { id: "qp", label: "tracking stiffness", min: 3, max: 6, step: 0.1, value: Math.log10(1.6e5), scale: "log10", fmt: (v) => v.toExponential(1) },
        { id: "brake_T", label: "braking time (next throw)", min: 0.2, max: 1.0, step: 0.05, value: 0.6, fmt: (v) => v.toFixed(2) + " s" },
        { id: "comp_ms", label: "release-lag compensation (next throw)", min: 0, max: 100, step: 5, value: 0, fmt: (v) => v.toFixed(0) + " ms" },
      ],
      info: true,
    },
  ],
  weaknesses: [
    { id: "release", label: "gripper release lag", min: 0, max: 100, step: 5, value: 0, fmt: (v) => v.toFixed(0) + " ms" },
    { id: "mass", label: "true ball mass", min: -50, max: 100, step: 5, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
    { id: "drag", label: "true air drag", min: -100, max: 200, step: 10, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
  ],
  weaknessNote: "The controller plans with the nominal 0.15 kg ball, drag and the scheduled release time (0.8 s); the sliders change the real world only. A 40 ms lag means the ball leaves 8 control steps late, while the arm is already swinging, so it lands off target; the compensation slider tells the planner about a lag you know. Drag the ball or a link with the mouse; Restart picks a new basket.",
  options: [{ id: "basket", label: "basket", choices: ["random distance", "near", "far"], value: "random distance" }],
  create: (mj, assets) => new TossSession(mj, assets),
};
