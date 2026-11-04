// Task plugin: catch (a ball is thrown at the arm; a 3-link arm with a shallow basket must catch it and keep it for 0.5 s).
// JS port of arena/tasks/catch.py and arena/controllers/catch_teacher.py (ballistic prediction + intercept planning + IK + computed torque).
import { ArmModel, lqrDoubleIntegrator } from "../sim.js";
import { clip, rng, loadXml, DelayLine } from "./common.js";
import * as bal from "./_ball_common.js";
import { pyRound, clamp, withBall, rgbaColor } from "./_ball_common.js";

const R_MIN = 0.15, R_MAX = 0.70;   // allowed wrist radius at the planned contact
const minjerk = (s) => { s = clamp(s, 0, 1); return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5; };
const maxAbs = (a) => a.reduce((m, x) => Math.max(m, Math.abs(x)), 0);

/** Classical catch teacher: nominal model knowledge only (drag rate, gravity, ball radius, home posture, arm model). */
export class CatchTeacher {
  constructor(arm, cfg, params) {
    this.arm = arm; this.cfg = cfg; this.L = cfg.link_lengths; this.dt = cfg.dt; this.tauLimit = cfg.tau_max;
    this.p = { ...params };
    this.k = cfg.drag_rate; this.g = cfg.gravity; this.rb = cfg.ball_radius; this.qHome = cfg.q_home;
    this._gains();
    this.reset();
  }
  _gains() { [this.Kp, this.Kd] = lqrDoubleIntegrator(this.p.qp, this.p.qv); }
  setParam(id, v) { this.p[id] = v; if (id === "qp" || id === "qv") this._gains(); }
  reset() {
    this.ts = []; this.ps = [];
    this.tc = null; this.tPre = null; this.plan = null;
    this.lastPlanT = -1; this.donePlanning = false; this.info = {};
  }
  _ballFn() {
    const ts = this.ts, ps = this.ps, lim = ts[ts.length - 1] - this.p.fit_window;
    const idx = ts.map((t, i) => (t >= lim ? i : -1)).filter((i) => i >= 0);
    const [p0, v0] = bal.fit(idx.map((i) => ts[i]), idx.map((i) => ps[i]), this.k, this.g);
    const k = this.k, g = this.g;
    return [(t) => bal.position(t, p0, v0, k, g), (t) => bal.velocity(t, v0, k, g),
      (t) => { const v = bal.velocity(t, v0, k, g); return [-k * v[0], -k * v[1] - g]; }];
  }
  _wristR(p) { return Math.hypot(p[0] - this.L[2], p[1]); }
  /** level IK with projection onto the reachable annulus: returns [q, projection distance] */
  _ikProj(p) {
    const [L1, L2, L3] = this.L;
    let wx = p[0] - L3, wz = p[1];
    const r = Math.hypot(wx, wz), rc = clamp(r, 0.08, 0.745);
    wx = (wx * rc) / Math.max(r, 1e-9); wz = (wz * rc) / Math.max(r, 1e-9);
    const D = clamp((rc * rc - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1);
    const q2 = Math.acos(D);
    const th1 = Math.atan2(wz, wx) - Math.atan2(-L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
    const q1 = -th1;
    return [[q1, q2, -q1 - q2], Math.abs(r - rc)];
  }
  _build(t0, ps, vs, as, tc, ball) {
    const c = this.p, dt = this.dt, rb = this.rb, L = this.L;
    const [pos, vel, acc] = ball;
    const tPre = tc - c.tau_s;
    const n = Math.ceil((tc + c.brake_T - t0) / dt) + 1;
    const pre = ((q) => [q[0], q[1] - (rb + c.d_pre)])(pos(tPre));
    const reach = bal.quintic(t0, [ps, vs, as], tPre, [pre, vel(tPre), acc(tPre)]);
    const pc0 = pos(tc), pc = [pc0[0], pc0[1] - rb], vtc = vel(tc);
    let stop = pc;
    for (const f of [0.4, 0.3, 0.2, 0.12, 0.06, 0.0]) {
      stop = [pc[0] + f * vtc[0] * c.brake_T, pc[1] + f * vtc[1] * c.brake_T];
      if (this._wristR(stop) <= 0.70) break;
    }
    const brake = bal.quintic(tc, [pc, vtc, acc(tc)], tc + c.brake_T, [stop, [0, 0], [0, 0]]);
    const plan = { t0, t: [], Q: [], Qd: [], Qdd: [], P: [], V: [], A: [], tc };
    let proj = 0, qdMax = 0;
    for (let i = 0; i < n; i++) {
      const t = t0 + i * dt;
      let P, V, A;
      if (t < tPre) [P, V, A] = reach(t);
      else if (t <= tc) {
        const s = (t - tPre) / c.tau_s, m = minjerk(s);
        const m1 = 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4, m2 = 60 * s - 180 * s ** 2 + 120 * s ** 3;
        const d = c.d_pre * (1 - m), d1 = (-c.d_pre * m1) / c.tau_s, d2 = (-c.d_pre * m2) / c.tau_s ** 2;
        const pp = pos(t), vv = vel(t), aa = acc(t);
        P = [pp[0], pp[1] - (rb + d)]; V = [vv[0], vv[1] - d1]; A = [aa[0], aa[1] - d2];
      } else [P, V, A] = brake(t);
      const [Q, dp] = this._ikProj(P);
      proj = Math.max(proj, dp);
      const phi = [Q[0], Q[0] + Q[1], Q[0] + Q[1] + Q[2]];
      const dx = L.map((l, j) => -l * Math.sin(phi[j])), dz = L.map((l, j) => -l * Math.cos(phi[j]));
      const J3 = [[dx[0] + dx[1] + dx[2], dx[1] + dx[2], dx[2]], [dz[0] + dz[1] + dz[2], dz[1] + dz[2], dz[2]], [1, 1, 1]];
      const JJt = J3.map((r) => J3.map((s) => r.reduce((a, x, k) => a + x * s[k], 0)));
      for (let j = 0; j < 3; j++) JJt[j][j] += 1e-6;
      const apply = (rhs) => {   // J3^T (J3 J3^T + eps I)^-1 rhs
        const y = bal.solve(JJt, rhs.map((x) => [x])).map((r) => r[0]);
        return [0, 1, 2].map((j) => J3[0][j] * y[0] + J3[1][j] * y[1] + J3[2][j] * y[2]);
      };
      const Qd = apply([V[0], V[1], 0]);
      const w = [Qd[0], Qd[0] + Qd[1], Qd[0] + Qd[1] + Qd[2]];
      const jd = [-(L[0] * Math.cos(phi[0]) * w[0] ** 2 + L[1] * Math.cos(phi[1]) * w[1] ** 2 + L[2] * Math.cos(phi[2]) * w[2] ** 2),
        L[0] * Math.sin(phi[0]) * w[0] ** 2 + L[1] * Math.sin(phi[1]) * w[1] ** 2 + L[2] * Math.sin(phi[2]) * w[2] ** 2, 0];
      const Qdd = apply([A[0] - jd[0], A[1] - jd[1], -jd[2]]);
      qdMax = Math.max(qdMax, maxAbs(Qd));
      plan.t.push(t); plan.Q.push(Q); plan.Qd.push(Qd); plan.Qdd.push(Qdd); plan.P.push(P); plan.V.push(V); plan.A.push(A);
    }
    plan.proj = proj; plan.qdMax = qdMax;
    return plan;
  }
  _cost(plan, stride = 2) {
    let tau = 0;
    for (let i = 0; i < plan.t.length; i += stride) tau = Math.max(tau, maxAbs(this.arm.inverse(plan.Q[i], plan.Qd[i], plan.Qdd[i])) / this.tauLimit);
    plan.parts = [tau, plan.qdMax / this.p.qd_max, plan.proj];
    return Math.max(tau, plan.qdMax / this.p.qd_max) + 5 * plan.proj;
  }
  _contactOk(pos, tc) {
    return [tc, tc - this.p.tau_s].every((t) => { const q = pos(t), r = this._wristR([q[0], q[1] - this.rb]); return R_MIN <= r && r <= R_MAX; });
  }
  _startState(t) {
    if (!this.plan) return [bal.fk3(this.L, this.qHome), [0, 0], [0, 0]];
    const i = clamp(pyRound((t - this.plan.t0) / this.dt), 0, this.plan.t.length - 1);
    return [this.plan.P[i], this.plan.V[i], this.plan.A[i]];
  }
  _replan(t) {
    const c = this.p, ball = this._ballFn(), [ps, vs, as] = this._startState(t), pos = ball[0];
    if (this.tc === null) {
      let best = null;
      const start = t + c.tmin, delta = (start + 0.01) - start, N = Math.ceil((t + 1.1 - start) / 0.01);
      for (let i = 0; i < N; i++) {
        const tc = start + i * delta;
        if (!this._contactOk(pos, tc)) continue;
        const plan = this._build(t, ps, vs, as, tc, ball), cost = this._cost(plan);
        if (best === null || cost < best[0]) best = [cost, plan];
        if (cost <= c.torque_frac) { best = [cost, plan]; break; }
      }
      if (best === null) return;
      this.plan = best[1]; this.tc = best[1].tc;
      this.info = { cost: best[0], parts: best[1].parts };
    } else this.plan = this._build(t, ps, vs, as, this.tc, ball);
    this.tPre = this.tc - c.tau_s;
    this.lastPlanT = t;
  }
  /** obs = {t, q, qd, ball: [x, z]} -> joint torques */
  act(obs) {
    const t = obs.t, c = this.p;
    this.ts.push(t); this.ps.push(obs.ball.slice());
    if (t >= c.t_first && this.ts.length >= 8 && !this.donePlanning && t - this.lastPlanT >= c.replan_dt - 1e-9) {
      if (this.tc !== null && this.tPre - t < 0.06) this.donePlanning = true;
      else this._replan(t);
    }
    let qr, qdr, qddr;
    if (!this.plan) { qr = this.qHome; qdr = [0, 0, 0]; qddr = [0, 0, 0]; } else {
      const pl = this.plan, i = clamp(pyRound((t - pl.t0) / this.dt), 0, pl.t.length - 1);
      qr = pl.Q[i]; qdr = pl.Qd[i]; qddr = pl.Qdd[i];
      if (t > pl.t[pl.t.length - 1] + 1e-9) { qdr = [0, 0, 0]; qddr = [0, 0, 0]; }
    }
    const v = [0, 1, 2].map((j) => qddr[j] + this.Kp * (qr[j] - obs.q[j]) + this.Kd * (qdr[j] - obs.qd[j]));
    return this.arm.inverse(obs.q, obs.qd, v).map((x) => clip(x, this.tauLimit));
  }
}

const DEFAULTS = { qp: 1.6e5, qv: 800, fit_window: 0.25, t_first: 0.10, replan_dt: 0.03, tmin: 0.20, tau_s: 0.08, d_pre: 0.03, brake_T: 0.3, torque_frac: 0.85, qd_max: 25 };

export class CatchSession {
  constructor(mj, assets) {
    this.mj = mj; this.xml = assets.xml; this.cfg = assets.cfg;
    const c = this.cfg;
    Object.assign(this, { dt: c.dt, substeps: c.substeps, tauMax: c.tau_max, duration: c.duration, nSteps: Math.round(c.duration / c.dt) });
    this.arm = new ArmModel(mj, assets.nominal);
    this.teacher = new CatchTeacher(this.arm, c, DEFAULTS);
    this.obsDelay = new DelayLine();
    this.mass = c.ball_mass; this.drag = c.drag;
    this.repeat = false;
    this.model = this.data = null;
    this.caught = 0; this.attempts = 0;
    this._build();
    this.active = "teacher";
    this.reset(1000);
  }
  _build() {
    const old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel) } : null;
    this.data?.delete(); this.model?.delete();
    this.model = loadXml(this.mj, withBall(this.xml, "ball", ["bx", "bz"], this.mass, this.drag), "/plant.xml");
    this.data = new this.mj.MjData(this.model);
    if (old) {
      old.q.forEach((x, i) => (this.data.qpos[i] = x)); old.v.forEach((x, i) => (this.data.qvel[i] = x));
      this.mj.mj_forward(this.model, this.data);
    }
    const id = (type, n) => this.mj.mj_name2id(this.model, type.value, n);
    this.tipSite = id(this.mj.mjtObj.mjOBJ_SITE, "tip");
    this.ballGeom = id(this.mj.mjtObj.mjOBJ_GEOM, "ball");
    this.ballBody = id(this.mj.mjtObj.mjOBJ_BODY, "ball");
    this.netGeoms = ["net_floor", "net_wall0", "net_wall1"].map((n) => id(this.mj.mjtObj.mjOBJ_GEOM, n));
  }
  // ---- plugin interface ----
  setController() {}
  setParam(ctl, id, v) { this.teacher.setParam(id, v); }
  setWeakness(id, v) {
    if (id === "mass") { this.mass = this.cfg.ball_mass * (1 + v / 100); this._build(); }
    else if (id === "drag") { this.drag = this.cfg.drag * (1 + v / 100); this._build(); }
    else if (id === "delay") this.obsDelay.steps = v | 0;
  }
  setOption(id, v) { if (id === "throw") this.repeat = v.startsWith("same"); }
  controllerInfo() { return "The catch point (red dot) is planned about 0.1 s after the throw and refined every 30 ms until the basket starts its final approach."; }
  /** launch from the seed (JS RNG, same distributions as Python) or from an explicit init {q: [3 joints + ball xz], qd: [... ball vxz]} (validation) */
  reset(seed, init = null) {
    const c = this.cfg, n = 3;
    if (this.repeat && !init) seed = 1000;
    this.seed = seed;
    this.mj.mj_resetData(this.model, this.data);
    let q, qd;
    if (init) { q = init.q; qd = init.qd; } else {
      const r = rng(seed), u = (a, b) => a + (b - a) * r();
      const x0 = u(1.25, 1.55), z0 = u(0.4, 0.8), aim = [u(0.45, 0.75), u(0.10, 0.40)], T = u(0.55, 0.75);
      const v0 = bal.launchVelocity([x0, z0], aim, T, c.drag_rate, c.gravity);
      q = [...c.q_home.map((x) => x + u(-0.02, 0.02)), x0, z0];
      qd = [0, 0, 0, v0[0], v0[1]];
    }
    for (let i = 0; i < n + 2; i++) { this.data.qpos[i] = q[i]; this.data.qvel[i] = qd[i]; }
    this.data.xfrc_applied.fill(0);
    this.mj.mj_forward(this.model, this.data);
    this.k = 0; this.done = false; this.outcome = null; this.message = ""; this.recordNow = true;
    this.dwellN = 0; this.minDist = Infinity; this.contact = false;
    this.obsDelay.clear(); this.teacher.reset();
    this.lastTau = null;
    this._observe();
  }
  ballPos() { return [this.data.qpos[3], this.data.qpos[4]]; }
  tipPos() { const s = this.data.site_xpos; return [s[3 * this.tipSite], s[3 * this.tipSite + 2]]; }
  _observe() {
    const d = this.data;
    this.obs = { t: this.k * this.dt, q: [d.qpos[0], d.qpos[1], d.qpos[2]], qd: [d.qvel[0], d.qvel[1], d.qvel[2]], ball: this.obsDelay.push(this.ballPos()) };
  }
  _inBasket() {
    const d = this.data, s = this.tipSite, R = d.site_xmat, o = 9 * s, p = this.ballPos();
    const rx = p[0] - d.site_xpos[3 * s], ry = 0 - d.site_xpos[3 * s + 1], rz = p[1] - d.site_xpos[3 * s + 2];
    const lx = R[o] * rx + R[o + 3] * ry + R[o + 6] * rz, lz = R[o + 2] * rx + R[o + 5] * ry + R[o + 8] * rz;   // R^T r
    return Math.abs(lx) < 0.10 && lz > 0 && lz < 0.075;
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
    if (!this.done) {
      const b = this.ballPos(), tip = this.tipPos();
      this.minDist = Math.min(this.minDist, Math.hypot(b[0] - tip[0], b[1] - tip[1]));
      this.dwellN = this._inBasket() ? this.dwellN + 1 : 0;
      const held = this.dwellN * this.dt;
      if ((b[1] < this.cfg.floor_z || b[0] < -1) && held < this.cfg.dwell) this._finish("failure", `Missed: the ball ${b[1] < this.cfg.floor_z ? "hit the floor" : "flew past"}`);
      else if (held >= this.cfg.dwell) this._finish("success", `Caught and held for ${this.cfg.dwell.toFixed(1)} s`);
      else if (this.k >= this.nSteps) this._finish("failure", "Time out: never settled in the basket");
    }
    return tau;
  }
  _finish(outcome, message) {
    this.done = true; this.outcome = outcome; this.message = message;
    this.attempts += 1; if (outcome === "success") this.caught += 1;
  }
  // ---- display ----
  chartValue() { const b = this.ballPos(), t = this.tipPos(); return Math.hypot(b[0] - t[0], b[1] - t[1]); }
  refPath() { return null; }
  target() {
    const pl = this.teacher.plan;
    if (!pl) return this.tipPos();
    const i = clamp(pyRound((this.teacher.tc - pl.t0) / this.dt), 0, pl.t.length - 1);
    return pl.P[i];
  }
  trailPoint() { return this.ballPos(); }
  metrics() {
    const b = this.ballPos(), t = this.tipPos();
    return [
      ["time", `${(this.k * this.dt).toFixed(2)} s`],
      ["ball to basket", `${(Math.hypot(b[0] - t[0], b[1] - t[1]) * 100).toFixed(0)} cm`],
      ["in basket", `${(this.dwellN * this.dt).toFixed(2)} / ${this.cfg.dwell.toFixed(1)} s`],
      ["caught", `${this.caught} / ${this.attempts}`],
      ["torque", this.lastTau ? `${this.lastTau.map((x) => x.toFixed(0)).join(", ")} Nm` : "-"],
      ["result", this.done ? (this.outcome === "success" ? "CAUGHT" : "MISSED") : this.dwellN > 0 ? "settling" : "in flight"],
    ];
  }
  geomColor(i) { return rgbaColor(this.model, i); }
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]], b = this.ballBody, bp = p(b);
    return [
      { body: 1, a: p(1), b: p(2) }, { body: 2, a: p(2), b: p(3) }, { body: 3, a: p(3), b: p(4) },
      { body: b, a: bp, b: [bp[0] + 1e-4, bp[1]], fmax: 4, k: 60, damp: 0.6 },   // a gentle shove on the ball in flight
    ];
  }
  dispose() { this.data?.delete(); this.model?.delete(); this.arm.data.delete(); this.arm.model.delete(); }
}

const fmt = (d, u = "") => (v) => v.toFixed(d) + u;

export default {
  id: "catch",
  title: "Catch",
  summary:
    "A ball is thrown toward the arm and it must catch it in a shallow basket and keep it there for half a second. " +
    "The classical controller watches the ball for a moment, predicts its flight (gravity plus air drag), picks an interception point the arm can reach, " +
    "and moves the basket there at the ball's speed so the landing is soft. Break it with sensor delay or a wrong ball mass, or shove the ball with the mouse.",
  assets: { xml: "assets/catch.xml", nominal: "assets/catch_nominal.xml", cfg: "assets/catch.json" },
  camera: { target: [0.55, 0.0], distance: 3.3, fov: 32 },
  chart: { label: "ball to basket distance (cm)", scale: 100, signed: false, minRange: 0.3, maxCap: 1.6 },
  legendNote: "red dot = planned catch point, orange = ball, black = ball path, green = basket",
  interaction: { fmax: 120, k: 220, damp: 18, hitRadius: 0.12 },
  controllers: [
    {
      id: "teacher", label: "Predict + intercept", color: "#2563eb", name: "Ballistic prediction + intercept + computed torque",
      note: "Fits the drag-ballistic flight to the last fraction of a second of ball positions, picks the earliest catch point the arm can reach within its torque budget, plans a smooth path that arrives with the ball's velocity, then brakes. Joint torques come from computed torque with the nominal model.",
      params: [
        { id: "fit_window", label: "fit window", min: 0.1, max: 0.5, step: 0.01, value: 0.25, fmt: fmt(2, " s") },
        { id: "tau_s", label: "ride-along time", min: 0.02, max: 0.2, step: 0.01, value: 0.08, fmt: fmt(2, " s") },
        { id: "d_pre", label: "pre-contact gap", min: 0, max: 0.08, step: 0.005, value: 0.03, fmt: (v) => (v * 100).toFixed(1) + " cm" },
        { id: "brake_T", label: "braking time", min: 0.1, max: 0.6, step: 0.05, value: 0.3, fmt: fmt(2, " s") },
        { id: "torque_frac", label: "torque budget", min: 0.3, max: 1.2, step: 0.05, value: 0.85, fmt: (v) => (v * 100).toFixed(0) + " %" },
        { id: "qp", label: "tracking stiffness", min: 3, max: 6, step: 0.1, value: Math.log10(1.6e5), scale: "log10", fmt: (v) => v.toExponential(1) },
      ],
      info: true,
    },
  ],
  weaknesses: [
    { id: "delay", label: "ball sensor delay", min: 0, max: 8, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
    { id: "mass", label: "true ball mass", min: -50, max: 100, step: 5, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
    { id: "drag", label: "true air drag", min: -100, max: 200, step: 10, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
  ],
  weaknessNote: "The controller assumes the nominal 0.1 kg ball and drag; these sliders change the real ball only. A 10 ms (2 step) camera delay makes the ball look like it is behind its true position, so the predicted catch point lands off target. Drag the orange ball to shove it; Restart throws again.",
  options: [{ id: "throw", label: "throw", choices: ["new random throw", "same throw every time"], value: "new random throw" }],
  create: (mj, assets) => new CatchSession(mj, assets),
};
