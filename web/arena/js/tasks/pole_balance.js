// Task plugin: pole balance (inverted pendulum on a passive hinge at the arm tip).
// JS port of arena/tasks/pole_balance.py, arena/controllers/pid.py (pole branch) and arena/controllers/ctc_lqr.py (PoleLQR + CtcLqr).
import { ArmModel } from "../sim.js";
import { clip, rng, loadXml, withPayload, DelayLine, fk2, jac2, jdotqd2, inv2, mulv2, lqr } from "./common.js";

/** LQR gain for the linearised pole driven by the tip x-acceleration: state [x - x_hold, xdot, theta, omega] */
export function poleGain(L, g, qDiag, r) {
  const A = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, g / L, 0]];
  const B = [[0], [1], [0], [-1 / L]];
  return lqr(A, B, qDiag.map((q, i) => qDiag.map((_, j) => (i === j ? q : 0))), [[r]]);
}

class PolePID {
  constructor(arm, cfg, tauMax, dt) {
    Object.assign(this, { arm, cfg, tauMax, dt, ...cfg.pid });
    this.reset();
  }
  reset() { this.zth = 0; this.kd = null; }
  setParam(id, v) { this[id] = v; if (id === "wn") this.kd = null; }
  act(obs, extra) {
    const { cfg } = this, n = 2;
    if (this.kd === null) this.kd = this.arm.massDiag(cfg.q_hold).map((m) => 2 * this.wn * m);
    const th = extra.theta, om = extra.omega;
    this.zth = clip(this.zth + th * this.dt, 0.2);
    const v = [this.c_i * this.zth + this.c_p * th + this.c_d * om, 0];
    const qdr = mulv2(inv2(jac2(cfg.link_lengths, cfg.q_hold)), v);   // reference q-dot: J(q_hold)^-1 [v_x, 0]
    const g = this.arm.gravity(obs.q);
    const tau = new Array(n);
    for (let i = 0; i < n; i++) tau[i] = clip(this.kd[i] * (qdr[i] - obs.qd[i]) + g[i], this.tauMax);   // position anchor weight 0 (as Python default)
    return tau;
  }
}

class PoleCtcLqr {
  constructor(arm, cfg, tauMax) {
    Object.assign(this, { arm, cfg, tauMax, q: cfg.lqr_q.slice(), r: cfg.lqr_r });
    this.K = cfg.lqr_K.slice();   // exported from scipy for the default weights (the gain at load time)
    this.reset();
  }
  reset() {}
  setParam(id, v) {
    if (id === "q_theta") this.q[2] = v;
    else if (id === "q_x") this.q[0] = v;
    else if (id === "r") this.r = v;
    this.K = poleGain(this.cfg.pole_length, this.cfg.gravity, this.q, this.r);
  }
  act(obs, extra) {
    const { cfg } = this, L = cfg.link_lengths;
    const J = jac2(L, obs.q), x = fk2(L, obs.q), v = mulv2(J, obs.qd);
    const s = [x[0] - cfg.hold[0], v[0], extra.theta, extra.omega];
    const ax = -(this.K[0] * s[0] + this.K[1] * s[1] + this.K[2] * s[2] + this.K[3] * s[3]);
    const az = -cfg.kz[0] * (x[1] - cfg.hold[1]) - cfg.kz[1] * v[1];
    const jd = jdotqd2(L, obs.q, obs.qd);
    const qdd = mulv2(inv2(J), [ax - jd[0], az - jd[1]]);
    return this.arm.inverse(obs.q, obs.qd, qdd).map((t) => clip(t, this.tauMax));
  }
}

export class PoleSession {
  constructor(mj, assets) {
    this.mj = mj;
    this.xml = assets.xml;
    this.cfg = assets.cfg;
    const c = this.cfg;
    Object.assign(this, { dt: c.dt, substeps: c.substeps, tauMax: c.tau_max, duration: c.duration, nSteps: Math.round(c.duration / c.dt) });
    this.arm = new ArmModel(mj, assets.nominal);
    this.delay = new DelayLine();
    this.payload = c.payload_mass;
    this.model = this.data = null;
    this._build(this.payload);
    this.controllers = { pid: new PolePID(this.arm, c, this.tauMax, this.dt), ctc: new PoleCtcLqr(this.arm, c, this.tauMax) };
    this.active = "pid";
    this.reset(1000);
  }
  _build(payload) {
    const old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel) } : null;
    this.data?.delete();
    this.model?.delete();
    this.model = loadXml(this.mj, withPayload(this.xml, payload), "/plant.xml");
    this.data = new this.mj.MjData(this.model);
    if (old) {
      old.q.forEach((x, i) => (this.data.qpos[i] = x));
      old.v.forEach((x, i) => (this.data.qvel[i] = x));
      this.mj.mj_forward(this.model, this.data);
    }
    const names = (type) => (n) => this.mj.mj_name2id(this.model, type.value, n);
    this.tipSite = names(this.mj.mjtObj.mjOBJ_SITE)("tip");
    this.bobGeom = this.model.ngeom - 1;   // last geom = the pole bob
  }
  // ---- plugin interface ----
  setController(id) { this.active = id; }
  setParam(ctl, id, v) { this.controllers[ctl].setParam(id, v); }
  setWeakness(id, v) {
    if (id === "payload") { this.payload = this.cfg.payload_mass * (1 + v / 100); this._build(this.payload); }
    else if (id === "delay") this.delay.steps = v | 0;
  }
  setOption() {}
  controllerInfo(id) {
    if (id !== "ctc") return "";
    const K = this.controllers.ctc.K;
    return `LQR gain on [x, xdot, theta, omega] = [${K.map((k) => k.toFixed(1)).join(", ")}]`;
  }
  /** initial state: from the seed (JS RNG, same distribution as Python) or explicitly {q:[q1,q2,hinge], qd:[...]} (validation) */
  reset(seed, init = null) {
    const c = this.cfg, n = 2;
    this.seed = seed;
    this.mj.mj_resetData(this.model, this.data);
    let q, qd;
    if (init) { q = init.q; qd = init.qd; } else {
      const r = rng(seed), u = (a) => (2 * r() - 1) * a;
      const q0 = c.q_hold.map((x) => x + u(0.02));
      const th0 = u(c.theta0), om0 = u(c.omega0);
      q = [q0[0], q0[1], th0 - q0[0] - q0[1]];
      qd = [0, 0, om0];
    }
    for (let i = 0; i <= n; i++) { this.data.qpos[i] = q[i]; this.data.qvel[i] = qd[i]; }
    this.data.xfrc_applied.fill(0);
    this.mj.mj_forward(this.model, this.data);
    this.k = 0;
    this.done = false;
    this.outcome = null;
    this.message = "";
    this.survival = 0;
    this.maxTheta = 0;
    this.recordNow = true;
    this.delay.clear();
    Object.values(this.controllers).forEach((x) => x.reset());
    this.lastTau = null;
    this._observe();
  }
  _observe() {
    const d = this.data;
    this.obs = { k: this.k, q: [d.qpos[0], d.qpos[1]], qd: [d.qvel[0], d.qvel[1]] };
    this.extra = { theta: d.qpos[0] + d.qpos[1] + d.qpos[2], omega: d.qvel[0] + d.qvel[1] + d.qvel[2] };
  }
  get ctrl() { return this.controllers[this.active]; }
  tipPos() {
    const s = this.data.site_xpos;
    return [s[3 * this.tipSite], s[3 * this.tipSite + 2]];
  }
  bobPos() {
    const x = this.data.geom_xpos, g = this.bobGeom;
    return [x[3 * g], x[3 * g + 2]];
  }
  /** one control period (env.step order); `hook(session)` runs before every physics substep (mouse force) */
  step(hook) {
    const cmd = this.ctrl.act(this.obs, this.extra);
    const tau = this.delay.push(cmd).map((x) => clip(x, this.tauMax));
    const d = this.data;
    d.ctrl[0] = tau[0];
    d.ctrl[1] = tau[1];
    for (let s = 0; s < this.substeps; s++) {
      if (hook) hook(this);
      this.mj.mj_step(this.model, d);
    }
    this.k += 1;
    this._observe();
    this.lastTau = tau;
    this.recordNow = !this.done;
    if (!this.done) {
      this.maxTheta = Math.max(this.maxTheta, Math.abs(this.extra.theta));
      if (Math.abs(this.extra.theta) > this.cfg.fall_angle) {
        this.done = true; this.outcome = "failure"; this.survival = this.k * this.dt;
        this.message = `Fell after ${this.survival.toFixed(2)} s`;
      } else if (this.k >= this.nSteps) {
        this.done = true; this.outcome = "success"; this.survival = this.k * this.dt;
        this.message = `Balanced for ${this.duration.toFixed(0)} s`;
      }
    }
    return tau;
  }
  // ---- display ----
  chartValue() { return this.extra.theta; }
  refPath() { return null; }
  target() { return this.cfg.hold; }
  trailPoint() { return this.bobPos(); }
  metrics() {
    const tip = this.tipPos(), h = this.cfg.hold, deg = 180 / Math.PI;
    return [
      ["time", `${(this.k * this.dt).toFixed(2)} s`],
      ["pole angle", `${(this.extra.theta * deg).toFixed(1)} deg`],
      ["tip drift", `${(Math.hypot(tip[0] - h[0], tip[1] - h[1]) * 100).toFixed(1)} cm`],
      ["worst angle", `${(this.maxTheta * deg).toFixed(1)} deg`],
      ["torque", this.lastTau ? `${this.lastTau.map((x) => x.toFixed(0)).join(", ")} Nm` : "-"],
      ["result", this.done ? (this.outcome === "success" ? "balanced" : "FELL") : "running"],
    ];
  }
  geomColor(i) { return i === this.bobGeom ? 0x7c3aed : i === this.bobGeom - 1 ? 0x374151 : null; }
  /** segments the mouse can grab: [{body, a:[x,z], b:[x,z], fmax?, k?, damp?}] */
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]];
    return [
      { body: 1, a: p(1), b: p(2) },
      { body: 2, a: p(2), b: p(3) },
      { body: 4, a: p(3), b: this.bobPos(), fmax: 1.5, k: 12, damp: 0.25 },   // the pole is light (0.05 kg): gentle pushes only
    ];
  }
  dispose() { this.data?.delete(); this.model?.delete(); this.arm.data.delete(); this.arm.model.delete(); }
}

const fmt1 = (v) => v.toFixed(1);

export default {
  id: "pole_balance",
  title: "Pole balance",
  summary:
    "An inverted pendulum stands on a free hinge at the arm's tip; the arm must keep it upright while staying near a hold point. " +
    "Compare a joint-space PID driven by the pole angle (Level 0) with computed torque plus an LQR on the pole (Levels 1-2). " +
    "Tune the gains, add payload error or latency, and poke the pole with the mouse.",
  assets: { xml: "assets/pole_balance.xml", nominal: "assets/pole_balance_nominal.xml", cfg: "assets/pole_balance.json" },
  camera: { target: [0.36, 0.2], distance: 2.0, fov: 32 },
  chart: { label: "pole angle (deg)", scale: 180 / Math.PI, signed: true, fixedMax: 35, guides: [34.4], guideLabel: "falls beyond 34 deg" },
  legendNote: "line colour = controller active at that moment; red dot = hold point, purple = pole bob, black = bob path",
  interaction: { fmax: 120, k: 220, damp: 18, hitRadius: 0.14 },
  controllers: [
    {
      id: "pid", label: "PID", color: "#2563eb", name: "PID on pole angle + gravity comp.",
      note: "Joint servo with gravity compensation, driven by a PID on the pole angle that commands a tip velocity (v = c_i * integral(theta) + c_p * theta + c_d * omega). No inverse dynamics.",
      params: [
        { id: "wn", label: "joint bandwidth", min: 10, max: 60, step: 1, value: 30, fmt: (v) => v.toFixed(0) + " rad/s" },
        { id: "c_i", label: "angle integral c_i", min: 0, max: 200, step: 5, value: 70, fmt: (v) => v.toFixed(0) },
        { id: "c_p", label: "angle gain c_p", min: 0, max: 20, step: 0.5, value: 5, fmt: fmt1 },
        { id: "c_d", label: "rate gain c_d", min: 0, max: 1, step: 0.05, value: 0.2, fmt: (v) => v.toFixed(2) },
      ],
    },
    {
      id: "ctc", label: "CTC + LQR", color: "#dc2626", name: "CTC + LQR",
      note: "An LQR on [tip x error, tip velocity, pole angle, pole rate] chooses the tip acceleration; computed torque (nominal inverse dynamics) turns it into joint torques.",
      params: [
        { id: "q_theta", label: "LQR weight Q_theta", min: 1, max: 4, step: 0.1, value: Math.log10(200), scale: "log10", fmt: (v) => v.toExponential(1) },
        { id: "q_x", label: "LQR weight Q_x", min: -1, max: 3, step: 0.1, value: Math.log10(20), scale: "log10", fmt: (v) => v.toExponential(1) },
        { id: "r", label: "control cost R", min: -4, max: 0, step: 0.1, value: -2, scale: "log10", fmt: (v) => v.toExponential(1) },
      ],
      info: true,
    },
  ],
  weaknesses: [
    { id: "payload", label: "true payload", min: -50, max: 100, step: 5, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
    { id: "delay", label: "command delay", min: 0, max: 8, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
  ],
  weaknessNote: "The controllers always assume the nominal 0.5 kg tip payload; \"true payload\" changes the plant only. Delay is actuator latency in whole 5 ms control periods. Drag the pole (purple) to push it; drag a link to shove the arm.",
  options: [],
  create: (mj, assets) => new PoleSession(mj, assets),
};
