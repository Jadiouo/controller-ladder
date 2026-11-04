// Task plugin: racket juggling (3-link arm + flat racket keeps one ball bouncing).
// JS port of spike03b/env.py: plant MJCF (exported to assets/juggling.xml), alpha-beta ball estimator, mirror-law teacher (Buhler/Koditschek style) + nominal computed torque.
import { ArmModel } from "../sim.js";
import { clip, rng, loadXml } from "./common.js";

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

/** np.interp on the exported restitution -> contact damping ratio map (clamped at the ends) */
export function zetaFor(map, e) {
  const idx = map.e.map((_, i) => i).sort((a, b) => map.e[a] - map.e[b]);
  const es = idx.map((i) => map.e[i]), zs = idx.map((i) => map.zeta[i]);
  if (e <= es[0]) return zs[0];
  if (e >= es[es.length - 1]) return zs[zs.length - 1];
  let j = 1;
  while (es[j] < e) j++;
  return zs[j - 1] + ((zs[j] - zs[j - 1]) * (e - es[j - 1])) / (es[j] - es[j - 1]);
}

/** planar 3-link kinematics of env.py: q = [q1, q2, q3], wrist at (x, z), racket angle phi = q1 + q2 + q3 */
export function ik(L, x, z, phi) {
  const r2 = x * x + z * z, c2 = clamp((r2 - L[0] * L[0] - L[1] * L[1]) / (2 * L[0] * L[1]), -1, 1), q2 = -Math.acos(c2);
  const q1 = Math.atan2(z, x) - Math.atan2(L[1] * Math.sin(q2), L[0] + L[1] * Math.cos(q2));
  return [q1, q2, phi - q1 - q2];
}
function jacInvApply(L, q, pd) {
  const p1 = q[0], p12 = q[0] + q[1], s1 = Math.sin(p1), c1 = Math.cos(p1), s12 = Math.sin(p12), c12 = Math.cos(p12);
  const A = [[-L[0] * s1 - L[1] * s12, -L[1] * s12, 0, pd[0]], [L[0] * c1 + L[1] * c12, L[1] * c12, 0, pd[1]], [1, 1, 1, pd[2]]];
  for (let c = 0; c < 3; c++) {   // Gauss-Jordan, partial pivoting
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 3; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j < 4; j++) A[r][j] -= f * A[c][j]; }
  }
  return [A[0][3] / A[0][0], A[1][3] / A[1][1], A[2][3] / A[2][2]];
}

/** standard normal samples from the mulberry32 stream (Box-Muller) */
function gaussian(seed) {
  const u = rng(seed);
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    const r = Math.sqrt(-2 * Math.log(1 - u())), a = 2 * Math.PI * u();
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
}

export class JugglingSession {
  constructor(mj, assets) {
    this.mj = mj;
    this.xml = assets.xml;
    const c = (this.cfg = assets.cfg);
    Object.assign(this, { dt: c.dt, substeps: c.substeps, tauMax: c.tau_max, duration: c.nmax * c.dt, nmax: c.nmax });
    this.arm = new ArmModel(mj, assets.nominal);
    this.OFF = c.r_ball + c.half_t;
    this.q0 = ik(c.L, c.x_star, c.wrist_z0, 0);
    // controller parameters (defaults = spike03b: the teacher tuned once on the nominal model)
    this.p = { kE: c.tp.kE, Kx: c.tp.Kx, kappa_max: c.kappa_max, e_model: c.e_nom, h_star: c.h_star, pred: c.tp.pred };
    this.trueE = c.e_nom;
    this.mscale = 1;
    this.sigma = c.sigma;
    this.delay = 0;
    this.noiseOverride = null;
    this.model = this.data = null;
    this._build();
    this.active = "mirror";
    this.reset(1000);
  }
  _build() {
    const c = this.cfg, old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel) } : null;
    this.data?.delete();
    this.model?.delete();
    const [m1, m2, m3] = c.masses.map((m) => m * this.mscale);
    const xml = this.xml.replaceAll("@M1@", m1).replaceAll("@M2@", m2).replaceAll("@M3@", m3).replaceAll("@ZETA@", zetaFor(c.zeta_map, this.trueE));
    this.model = loadXml(this.mj, xml, "/juggle.xml");
    this.data = new this.mj.MjData(this.model);
    if (old) {
      old.q.forEach((x, i) => (this.data.qpos[i] = x));
      old.v.forEach((x, i) => (this.data.qvel[i] = x));
      this.mj.mj_forward(this.model, this.data);
    }
  }
  // ---- plugin interface ----
  setController(id) { this.active = id; }
  setParam(ctl, id, v) { this.p[id] = id === "pred" ? Math.round(v) : v; }
  setWeakness(id, v) {
    if (id === "restitution") { this.trueE = v; this._build(); }
    else if (id === "payload") { this.mscale = 1 + v / 100; this._build(); }
    else if (id === "delay") {
      this.delay = v | 0;
      const want = this.delay + 1;   // measurement buffer: newest last
      while (this.buf.length < want) this.buf.unshift(this.buf[0]);
      while (this.buf.length > want) this.buf.shift();
    }
  }
  setOption() {}
  controllerInfo() { return ""; }
  /** init = {x, h, vx} (validation) else drawn from the seed; noise = flat [nx, nz, ...] sensor samples (validation) else Gaussian * sigma */
  reset(seed, init = null, noise = null) {
    const c = this.cfg, d = this.data, I = c.init;
    this.seed = seed;
    if (!init) {
      const r = rng(seed), u = (a, b) => a + (b - a) * r();
      init = { x: c.x_star + u(-I.dx, I.dx), h: u(I.h[0], I.h[1]), vx: u(-I.vx, I.vx) };
    }
    this.gauss = gaussian(seed + 7000000);
    this.noiseOverride = noise;
    this.mj.mj_resetData(this.model, d);
    for (let i = 0; i < 3; i++) d.qpos[i] = this.q0[i];
    d.qpos[3] = init.x;
    d.qpos[4] = c.wrist_z0 + this.OFF + init.h;
    d.qvel[3] = init.vx;
    d.xfrc_applied.fill(0);
    this.mj.mj_forward(this.model, d);
    this.buf = Array.from({ length: this.delay + 1 }, () => [init.x, d.qpos[4]]);
    this.est = [init.x, d.qpos[4] - this.OFF - c.wrist_z0, init.vx, 0];
    this.k = 0; this.hits = 0; this.dead = 0;
    this.done = false; this.outcome = null; this.message = "";
    this.recordNow = true;
    this.lastTau = null; this.lastKappa = 0;
    this.maxApex = 0; this.lastApex = null; this.apexErrs = []; this.hitX = [];
    this.fail = "";
  }
  zb() { return this.data.qpos[4] - this.OFF - this.cfg.wrist_z0; }
  /** sensor: noisy + delayed ball position -> alpha-beta filter (Python JugEnv.measure_update) */
  _measure() {
    const c = this.cfg, d = this.data, a = c.tp.a, b = c.tp.b, dt = this.dt, G = c.gravity, e = this.est;
    this.buf.push([d.qpos[3], d.qpos[4]]);
    const m = this.buf.shift();
    let nx, nz;
    if (this.noiseOverride) { nx = this.noiseOverride[2 * this.k] ?? 0; nz = this.noiseOverride[2 * this.k + 1] ?? 0; }
    else { nx = this.gauss() * this.sigma; nz = this.gauss() * this.sigma; }
    const zx = m[0] + nx, zz = m[1] - this.OFF - c.wrist_z0 + nz;
    const px = e[0] + e[2] * dt, pz = e[1] + e[3] * dt, pvz = e[3] - G * dt;
    const rx = zx - px, rz = zz - pz;
    e[0] = px + a * rx; e[2] = e[2] + (b / dt) * rx; e[1] = pz + a * rz; e[3] = pvz + (b / dt) * rz;
  }
  /** mirror law on the estimated ball state + nominal-model computed torque (Python JugEnv.teacher) */
  _teacher() {
    const c = this.cfg, tp = c.tp, p = this.p, G = c.gravity, dt = this.dt, L = c.L, d = this.data;
    const q = [d.qpos[0], d.qpos[1], d.qpos[2]], qd = [d.qvel[0], d.qvel[1], d.qvel[2]];
    let [xh, zh, vx, vz] = this.est;
    if (p.pred) { const t = p.pred * dt; xh = xh + vx * t; zh = zh + vz * t - 0.5 * G * t * t; vz = vz - G * t; }
    const E = G * Math.max(zh, 0) + 0.5 * vz * vz;
    const kap = clamp((1 - p.e_model) / (1 + p.e_model) + p.kE * (G * p.h_star - E) / G, 0, p.kappa_max);
    const zr = c.wrist_z0 - kap * zh, zrd = -kap * vz;
    const xr = clamp(xh, c.x_star - 0.35, c.x_star + 0.35), xrd = xr === xh ? vx : 0;
    let vimp = Math.sqrt(vz * vz + 2 * G * Math.max(zh, 0));
    vimp = Math.max(vimp, 0.5);
    const vxd = clamp(p.Kx * (c.x_star - xh), -tp.vxmax, tp.vxmax), dvx = vxd - vx;
    const s2 = clamp((2 * dvx) / ((1 + p.e_model) * (1 + kap) * vimp), -0.95, 0.95);
    const phi = clamp(-0.5 * Math.asin(s2), -tp.phimax, tp.phimax);
    const qr = ik(L, xr, zr, phi), qdr = jacInvApply(L, qr, [xrd, zrd, 0]);
    const acc = [0, 1, 2].map((i) => tp.kp * (qr[i] - q[i]) + tp.kd * (qdr[i] - qd[i]));
    this.lastKappa = kap;
    return this.arm.inverse(q, qd, acc).map((t) => clip(t, this.tauMax));
  }
  /** one control period (env.step order); `hook(session)` runs before every physics substep (mouse force) */
  step(hook) {
    const c = this.cfg, d = this.data;
    this._measure();
    const tau = this._teacher();
    d.ctrl[0] = tau[0]; d.ctrl[1] = tau[1]; d.ctrl[2] = tau[2];
    const vz0 = d.qvel[4];
    for (let s = 0; s < this.substeps; s++) {
      if (hook) hook(this);
      this.mj.mj_step(this.model, d);
    }
    this.k += 1;
    this.lastTau = tau;
    this.recordNow = !this.done;
    if (this.done) return tau;
    const zb = this.zb(), vz = d.qvel[4];
    if (vz0 < -0.1 && vz > 0 && zb < 0.12) { this.hits += 1; this.hitX.push(Math.abs(d.qpos[3] - c.x_star)); }
    if (vz0 > 0 && vz <= 0 && this.hits >= 1) {
      this.lastApex = zb; this.maxApex = Math.max(this.maxApex, zb);
      if (this.hits >= 3) this.apexErrs.push(Math.abs(zb - this.p.h_star));
    }
    this.dead = zb < 0.04 && Math.abs(vz) < 0.5 ? this.dead + 1 : 0;
    let f = "";
    if (this.dead > 60) f = "dead";
    else if (zb < -0.3) f = "dropped";
    else if (Math.abs(d.qpos[3] - c.x_star) > 0.5) f = "lateral";
    else if (!Number.isFinite(d.qpos[0]) || !Number.isFinite(d.qpos[4])) f = "nan";
    else if (this.k >= this.nmax) f = "timeout";
    if (f) this._finish(f, "failure");
    else if (this.hits >= c.hits_cap) this._finish("cap", "success");
    return tau;
  }
  _finish(f, outcome) {
    this.fail = f; this.done = true; this.outcome = outcome;
    const t = (this.k * this.dt).toFixed(1);
    this.message = f === "cap" ? `${this.hits} hits in a row (capped), ${t} s`
      : f === "timeout" ? `${this.hits} hits, ran the full ${t} s`
        : `Ball lost after ${this.hits} hits (${t} s)`;
  }
  // ---- display ----
  chartValue() { return this.hits; }
  refPath() { return null; }
  target() { return [this.cfg.x_star, this.cfg.wrist_z0 + this.OFF + this.p.h_star]; }
  trailPoint() { return [this.data.qpos[3], this.data.qpos[4]]; }
  metrics() {
    const cm = (v) => (v === null ? "-" : `${(v * 100).toFixed(0)} cm`);
    return [
      ["time", `${(this.k * this.dt).toFixed(1)} s`],
      ["consecutive hits", `${this.hits}`],
      ["last apex", cm(this.lastApex)],
      ["highest apex", this.hits ? cm(this.maxApex) : "-"],
      ["racket kappa", this.lastKappa.toFixed(2)],
      ["result", this.done ? (this.outcome === "success" ? "100 hits" : "ball lost") : "juggling"],
    ];
  }
  geomColor(i) { return i === 3 ? 0xea580c : i === 2 ? 0x111827 : null; }
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]];
    const g = this.data.geom_xpos, R = this.data.geom_xmat, w = 0.12;
    const rk = [g[6], g[8]], ux = R[18], uz = R[24];   // racket geom (index 2): centre and its local x axis in the x-z plane
    const ball = [this.data.qpos[3], this.data.qpos[4]];
    return [
      { body: 4, a: [ball[0] - 0.001, ball[1]], b: [ball[0] + 0.001, ball[1]], fmax: 0.9, k: 25, damp: 0.12 },   // a gentle side push on the 50 g ball
      { body: 3, a: [rk[0] - w * ux, rk[1] - w * uz], b: [rk[0] + w * ux, rk[1] + w * uz] },
      { body: 1, a: p(1), b: p(2) },
      { body: 2, a: p(2), b: p(3) },
    ];
  }
  dispose() { this.data?.delete(); this.model?.delete(); this.arm.data.delete(); this.arm.model.delete(); }
}

const fmt2 = (v) => v.toFixed(2);

export default {
  id: "juggling",
  title: "Juggling",
  summary:
    "A three-joint arm holds a flat racket and keeps a ball bouncing. The classical teacher is a mirror law: it moves the racket like a mirror " +
    "so the ball is sent back up to a target height, tracked by computed torque. It assumes a bouncy ball, so a deader ball, sensor latency or heavier links break it; " +
    "retune its model of the ball and push the ball with the mouse to see how much it can take.",
  assets: { xml: "assets/juggling.xml", nominal: "assets/juggling_nominal.xml", cfg: "assets/juggling.json" },
  camera: { target: [0.45, 0.4], distance: 2.6, fov: 32 },
  chart: { label: "consecutive hits", scale: 1, signed: false, fixedMax: 100 },
  legendNote: "line = consecutive racket hits over the episode; red dot = target apex of the ball; orange = ball, black = racket, black trail = ball path",
  interaction: { fmax: 60, k: 150, damp: 14, hitRadius: 0.1 },
  controllers: [
    {
      id: "mirror", label: "Mirror law + CTC", color: "#ea580c", name: "Mirror law + computed torque",
      note: "Racket height z_r = -kappa * z_ball and velocity -kappa * v_ball, with kappa set from the ball's energy error (the mirror gain), racket tilt to steer it back to x = 0.6 m. The controller's model of the bounce (model restitution) and its kappa limit are what you retune.",
      params: [
        { id: "kE", label: "mirror gain (energy)", min: 0, max: 3, step: 0.1, value: 1.0, fmt: fmt2 },
        { id: "h_star", label: "target apex height", min: 0.25, max: 0.75, step: 0.01, value: 0.5, fmt: (v) => (v * 100).toFixed(0) + " cm" },
        { id: "e_model", label: "model restitution", min: 0.3, max: 0.95, step: 0.01, value: 0.8, fmt: fmt2 },
        { id: "kappa_max", label: "kappa limit", min: 0.2, max: 1.6, step: 0.05, value: 0.4, fmt: fmt2 },
        { id: "Kx", label: "lateral gain", min: 0, max: 2.5, step: 0.1, value: 1.1, fmt: fmt2 },
        { id: "pred", label: "latency compensation", min: 0, max: 40, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
      ],
    },
  ],
  weaknesses: [
    { id: "restitution", label: "true restitution", min: 0.4, max: 0.8, step: 0.02, value: 0.8, fmt: fmt2 },
    { id: "delay", label: "ball sensing delay", min: 0, max: 30, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
    { id: "payload", label: "true link mass", min: 0, max: 400, step: 25, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
  ],
  weaknessNote: "The mirror law assumes restitution 0.8, nominal link masses and a clean, instant ball position. \"True\" values change the plant only: with restitution 0.4 the untuned teacher usually loses the ball within a few dozen hits; lower \"model restitution\" and raise the kappa limit to retune it. Drag the ball sideways with the mouse to disturb it (gentle force), the Restart button throws a new ball.",
  options: [],
  create: (mj, assets) => new JugglingSession(mj, assets),
};
