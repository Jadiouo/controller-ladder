// Task plugin: whip (3-link arm flings a 10-link passive chain so that the CHAIN TIP hits a target).
// JS port of arena/tasks/whip.py (plant, success test, metrics) and of the EXECUTION half of arena/controllers/whip_teacher.py
// (clamped cubic spline through 4 joint keyframes + computed torque with the arm-only nominal model).
// The PLANNING half (CMA-ES keyframe search on the nominal arm+chain model) is NOT ported: it takes seconds to minutes in Python
// (batched mujoco.rollout), too slow for a browser. Plans are PRECOMPUTED OFFLINE by the Python teacher for a grid of targets and
// assumed chain lengths (assets/whip.json "plans", made by validation/tools/export_whip.py) and looked up here.
import { ArmModel, lqrDoubleIntegrator } from "../sim.js";
import { clip, rng, loadXml, withPayload, DelayLine } from "./common.js";

const deg = 180 / Math.PI;

/** clamped cubic spline through (t_i, y_i) with zero end slopes (scipy CubicSpline bc_type=((1,0),(1,0))): returns {y, yd, ydd}(t) */
function makeSpline(T, Y) {
  const n = T.length, h = T.slice(1).map((t, i) => t - T[i]);
  const A = Array.from({ length: n }, () => new Array(n + 1).fill(0));
  A[0][0] = 2 * h[0]; A[0][1] = h[0]; A[0][n] = 6 * ((Y[1] - Y[0]) / h[0]);
  for (let i = 1; i < n - 1; i++) {
    A[i][i - 1] = h[i - 1]; A[i][i] = 2 * (h[i - 1] + h[i]); A[i][i + 1] = h[i];
    A[i][n] = 6 * ((Y[i + 1] - Y[i]) / h[i] - (Y[i] - Y[i - 1]) / h[i - 1]);
  }
  A[n - 1][n - 2] = h[n - 2]; A[n - 1][n - 1] = 2 * h[n - 2]; A[n - 1][n] = 6 * (-(Y[n - 1] - Y[n - 2]) / h[n - 2]);
  for (let c = 0; c < n; c++) {          // Gauss-Jordan (tridiagonal, diagonally dominant: no pivoting needed)
    const d = A[c][c];
    for (let j = c; j <= n; j++) A[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c && A[r][c]) { const f = A[r][c]; for (let j = c; j <= n; j++) A[r][j] -= f * A[c][j]; }
  }
  const M = A.map((r) => r[n]);
  return (t) => {
    if (t >= T[n - 1]) return [Y[n - 1], 0, 0];   // held after the last keyframe (Python: tc = min(t, T_last), derivatives 0 when t > T_last)
    let i = 0;
    while (i < n - 2 && t >= T[i + 1]) i++;
    const hh = h[i], a = (T[i + 1] - t) / hh, b = (t - T[i]) / hh;
    return [
      a * Y[i] + b * Y[i + 1] + (((a ** 3 - a) * M[i] + (b ** 3 - b) * M[i + 1]) * hh * hh) / 6,
      (Y[i + 1] - Y[i]) / hh - ((3 * a * a - 1) / 6) * hh * M[i] + ((3 * b * b - 1) / 6) * hh * M[i + 1],
      a * M[i] + b * M[i + 1],
    ];
  };
}

/** joint trajectory object for start posture q0 and keyframes (4 x n) */
export function makeTraj(knotT, q0, knots) {
  const n = q0.length;
  const sp = Array.from({ length: n }, (_, j) => makeSpline(knotT, [q0[j], ...knots.map((k) => k[j])]));
  return (t) => {
    const q = [], qd = [], qdd = [];
    for (let j = 0; j < n; j++) { const [a, b, c] = sp[j](t); q.push(a); qd.push(b); qdd.push(c); }
    return { q, qd, qdd };
  };
}

/** MJCF of the true plant from the exported template (arm from Python build_mjcf; chain + target from whip.py tip_xml) */
export function buildWhipXml(template, cfg, { len, mass, damp }) {
  const n = cfg.n_links;
  let xml = "";
  for (let i = n - 1; i >= 0; i--) {
    const L = len[i], m = mass[i], c = damp[i];
    const pos = i === 0 ? "0 0 0" : `${len[i - 1]} 0 0`;
    const site = i === n - 1 ? `<site name="whip_tip" pos="${L} 0 0" size="0.012" rgba="1 0.2 0.2 1"/>` : "";
    xml = `<body name="c${i + 1}" pos="${pos}"><joint name="cj${i + 1}" type="hinge" axis="0 1 0" damping="${c}"/>` +
      `<geom type="capsule" fromto="0 0 0 ${L} 0 0" size="0.008" mass="${m}" rgba="0.9 0.75 0.2 1" contype="0" conaffinity="0"/>${site}${xml}</body>`;
  }
  const world = `<geom name="target" type="sphere" size="${cfg.r_hit}" pos="1.4 0.06 0" rgba="0.2 0.8 0.3 0.35" contype="0" conaffinity="0"/>` +
    `<geom name="pedestal" type="box" size="0.06 0.06 0.455" pos="0 0 -0.455" rgba="0.40 0.42 0.46 1" contype="0" conaffinity="0"/>`;
  return template.replace("@@CHAIN@@", xml).replace("@@WORLD@@", world);
}

class WhipPlanned {
  /** classical teacher: look up the offline plan, then spline + computed torque */
  constructor(arm, cfg, tauMax) {
    Object.assign(this, { arm, cfg, tauMax, qp: cfg.qp, qv: cfg.qv });
    this._gains();
  }
  _gains() { [this.Kp, this.Kd] = lqrDoubleIntegrator(this.qp, this.qv); }
  setParam(id, v) { this[id] = v; this._gains(); }
  reset(session) { this.traj = makeTraj(this.cfg.knot_t, session.q0, session.knots()); }
  act(obs) {
    const { q, qd, qdd } = this.traj(obs.k * this.cfg.dt);
    const v = qdd.map((a, j) => a + this.Kp * (q[j] - obs.q[j]) + this.Kd * (qd[j] - obs.qd[j]));
    return this.arm.inverse(obs.q, obs.qd, v).map((t) => clip(t, this.tauMax));
  }
}

class WhipRandom extends WhipPlanned {
  /** baseline: same tracking, keyframes = one random draw around the start posture (no search) */
  reset(session) {
    const r = rng(session.seed * 7919 + 13), g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    const { q_lo, q_hi } = this.cfg;
    this.knots = Array.from({ length: 4 }, () => session.q0.map((x) => Math.max(q_lo, Math.min(q_hi, x + 0.8 * g()))));
    this.traj = makeTraj(this.cfg.knot_t, session.q0, this.knots);
  }
}

export class WhipSession {
  constructor(mj, assets) {
    this.mj = mj;
    this.template = assets.xml;
    this.cfg = assets.cfg;
    const c = this.cfg;
    Object.assign(this, { dt: c.dt, substeps: c.substeps, tauMax: c.tau_max, duration: c.duration, nSteps: Math.round(c.duration / c.dt) });
    this.arm = new ArmModel(mj, assets.nominal);
    this.delay = new DelayLine();
    this.nc = c.n_links;
    this.weak = { chainLen: 0, chainMass: 0, chainDamp: 0, payload: 0 };    // % change of the TRUE plant
    this.opt = { target: "seed", assumed: "100%" };
    this.planOverride = null;
    this.model = this.data = null;
    this._build();
    this.controllers = { teacher: new WhipPlanned(this.arm, c, this.tauMax), random: new WhipRandom(this.arm, c, this.tauMax) };
    this.active = "teacher";
    this.reset(1000);
  }
  /** (re)create the true plant from the weakness settings, keeping the current joint state */
  _build() {
    const c = this.cfg, w = this.weak, f = (p) => 1 + p / 100;
    const old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel) } : null;
    this.data?.delete();
    this.model?.delete();
    const fill = (x) => new Array(this.nc).fill(x);
    this.trueLen = c.chain_len * f(w.chainLen);
    let xml = buildWhipXml(this.template, c, { len: fill(this.trueLen), mass: fill(c.chain_mass * f(w.chainMass)), damp: fill(c.chain_damp * f(w.chainDamp)) });
    xml = withPayload(xml, c.payload_mass * f(w.payload));
    this.model = loadXml(this.mj, xml, "/whip_plant.xml");
    this.data = new this.mj.MjData(this.model);
    if (old) {
      old.q.forEach((x, i) => (this.data.qpos[i] = x));
      old.v.forEach((x, i) => (this.data.qvel[i] = x));
    }
    this.tipSite = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_SITE.value, "whip_tip");
    this.targetGeom = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_GEOM.value, "target");
    this.pedestalGeom = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_GEOM.value, "pedestal");
    if (this.tgt) this._placeTarget();
    this.mj.mj_forward(this.model, this.data);
  }
  _placeTarget() {
    const g = this.targetGeom;
    // y = +0.06 (behind the chain as seen by the camera) instead of -0.05 in Python: purely visual, the geom has no contact
    this.model.geom_pos[3 * g] = this.tgt[0]; this.model.geom_pos[3 * g + 1] = 0.06; this.model.geom_pos[3 * g + 2] = this.tgt[1];
  }
  // ---- plugin interface ----
  setController(id) { this.active = id; }
  setParam(ctl, id, v) { this.controllers[ctl].setParam(id, v); }
  setWeakness(id, v) {
    if (id === "delay") { this.delay.steps = v | 0; return; }
    this.weak[id] = v;
    this._build();
  }
  setOption(id, v) { this.opt[id] = v; }
  controllerInfo() { return ""; }
  /** key of the plan to look up for this episode's target and the planner's assumed chain length */
  planKey() { return `${this.targetCell.id}|${this.assumedScale().toFixed(1)}`; }
  assumedScale() { return { "90%": 0.9, "100%": 1.0, "110%": 1.1 }[this.opt.assumed] ?? 1.0; }
  /** keyframes (4 x n) of the active plan */
  knots() { return this.planOverride ?? this.cfg.plans[this.planKey()].knots; }
  plannedMinDist() { return this.planOverride ? NaN : this.cfg.plans[this.planKey()].planned_min_dist; }
  /** validation hook: execute these keyframes instead of the table (the Python teacher's per-seed plan) */
  setPlan(knots) { this.planOverride = knots; }
  /** seed -> grid target + start posture jitter (same distribution as Python for the posture); `init` = {q0, target:[x,z]} for validation */
  reset(seed, init = null) {
    const c = this.cfg, n = c.n;
    this.seed = seed;
    const r = rng(seed + 2000);       // salt chosen so that the first demo episodes hit (nominal hit rate over 100 seeds: 91 %)
    const cells = c.targets;
    const pick = this.opt.target === "seed" ? cells[Math.min(cells.length - 1, Math.floor(r() * cells.length))] : cells.find((t) => t.id === this.opt.target);
    this.targetCell = init?.cell ?? pick;
    this.tgt = init?.target ?? [this.targetCell.x, this.targetCell.z];
    this.q0 = init?.q0 ?? c.q_start.map((x) => x + (2 * r() - 1) * 0.02);
    this.mj.mj_resetData(this.model, this.data);
    for (let i = 0; i < n; i++) this.data.qpos[i] = this.q0[i];
    this.data.qpos[n] = Math.PI / 2 - this.q0.reduce((a, b) => a + b, 0);   // first chain link points straight down, the rest hangs along it
    this.data.xfrc_applied.fill(0);
    this._placeTarget();
    this.mj.mj_forward(this.model, this.data);
    this.k = 0;
    this.done = false;
    this.outcome = null;
    this.message = "";
    this.recordNow = true;
    this.delay.clear();
    this.lastTau = null;
    this.hitSpeed = NaN;
    this.minDist = this.tipDist();
    this.prevTip = this.tipPos();
    this.controllers.teacher.reset(this);
    this.controllers.random.reset(this);
    this._observe();
  }
  _observe() {
    const d = this.data, n = this.cfg.n;
    this.obs = { k: this.k, q: Array.from(d.qpos.slice(0, n)), qd: Array.from(d.qvel.slice(0, n)) };
  }
  get ctrl() { return this.controllers[this.active]; }
  tipPos() {
    const s = this.data.site_xpos;
    return [s[3 * this.tipSite], s[3 * this.tipSite + 2]];
  }
  tipDist() { const p = this.tipPos(); return Math.hypot(p[0] - this.tgt[0], p[1] - this.tgt[1]); }
  step(hook) {
    const cmd = this.ctrl.act(this.obs);
    const tau = this.delay.push(cmd).map((x) => clip(x, this.tauMax));
    const d = this.data, n = this.cfg.n;
    for (let i = 0; i < n; i++) d.ctrl[i] = tau[i];
    for (let s = 0; s < this.substeps; s++) {
      if (hook) hook(this);
      this.mj.mj_step(this.model, d);
    }
    this.k += 1;
    this._observe();
    this.lastTau = tau;
    this.recordNow = !this.done;
    if (!this.done) {
      const dist = this.tipDist(), tip = this.tipPos();
      this.minDist = Math.min(this.minDist, dist);
      if (dist < this.cfg.r_hit) {
        this.hitSpeed = Math.hypot(tip[0] - this.prevTip[0], tip[1] - this.prevTip[1]) / this.dt;
        this.done = true; this.outcome = "success";
        this.message = `Hit at ${(this.k * this.dt).toFixed(2)} s, chain tip ${this.hitSpeed.toFixed(1)} m/s`;
      } else if (this.k >= this.nSteps) {
        this.done = true; this.outcome = "failure";
        this.message = `Missed: closest approach ${(this.minDist * 100).toFixed(1)} cm (hit radius ${(this.cfg.r_hit * 100).toFixed(0)} cm)`;
      }
      this.prevTip = tip;
    }
    return tau;
  }
  // ---- display ----
  chartValue() { return this.tipDist(); }
  refPath() { return null; }
  target() { return this.tgt; }
  trailPoint() { return this.tipPos(); }
  metrics() {
    const planned = this.plannedMinDist(), cm = (x) => `${(x * 100).toFixed(1)} cm`;
    return [
      ["time", `${(this.k * this.dt).toFixed(2)} s`],
      ["tip to target", cm(this.tipDist())],
      ["closest so far", cm(this.minDist)],
      ["planned closest", this.active === "teacher" && Number.isFinite(planned) ? cm(planned) : "-"],
      ["chain true/assumed", `${(this.trueLen * this.nc).toFixed(2)} / ${(this.cfg.chain_len * this.nc * this.assumedScale()).toFixed(2)} m`],
      ["result", this.done ? (this.outcome === "success" ? "HIT" : "missed") : "running"],
    ];
  }
  geomColor(i) {
    const n = this.cfg.n, body = this.model.geom_bodyid[i];
    if (i === this.targetGeom) return 0x4ade80;
    if (i === this.pedestalGeom) return 0x9ca3af;
    if (body === n + 1) return 0x374151;                // tip payload
    if (body >= n + 2) return 0xd4a017;                 // chain
    return null;
  }
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]], n = this.cfg.n, tip = this.tipPos();
    const segs = [];
    for (let b = 1; b <= n; b++) segs.push({ body: b, a: p(b), b: p(b + 1) });                       // arm links (bodies 1..n; body n+1 = tip)
    for (let i = 0; i < this.nc; i++) {                                                            // chain links (bodies n+2 ...): light, gentle pulls
      const b = n + 2 + i;
      segs.push({ body: b, a: p(b), b: i + 1 < this.nc ? p(b + 1) : tip, fmax: 0.6, k: 8, damp: 0.12 });
    }
    return segs;
  }
  dispose() { this.data?.delete(); this.model?.delete(); this.arm.data.delete(); this.arm.model.delete(); }
}

// grid of the offline plan table (distance d from the shoulder in m, angle a in rad); ids as in assets/whip.json
const TARGET_IDS = [1.2, 1.35, 1.5, 1.65].flatMap((d) => [-0.35, 0, 0.35].map((a) => `d${d.toFixed(2)}_a${a < 0 ? "-" : "+"}${Math.abs(a).toFixed(2)}`));
const fmtPct = (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %";

export default {
  id: "whip",
  title: "Whip",
  summary:
    "A 3-link arm has to fling a 10-link passive chain (1 m, no motors) so that the CHAIN TIP touches a green target that lies beyond the arm's own reach. " +
    "The classical teacher plans a swing offline (CMA-ES search over joint keyframes on a model of arm + chain) and tracks it with computed torque. " +
    "The plans here are precomputed by the Python teacher for a grid of targets and assumed chain lengths; the browser runs the swing on a chain whose true length, mass and damping you can change.",
  assets: { xml: "assets/whip.xml", nominal: "assets/whip_nominal.xml", cfg: "assets/whip.json" },
  camera: { target: [0.85, -0.68], distance: 4.7, fov: 32 },
  chart: { label: "chain tip to target (cm)", scale: 100, signed: false, guides: [8], guideLabel: "hit radius 8 cm", minRange: 0.4, maxCap: 2.4 },
  legendNote: "red dot / green sphere = target, gold = chain, black line = chain-tip path; success = tip within 8 cm of the target before 2.5 s",
  interaction: { fmax: 80, k: 160, damp: 14, hitRadius: 0.12 },
  controllers: [
    {
      id: "teacher", label: "Teacher", color: "#2563eb", name: "Offline-planned swing + computed torque",
      note: "Plans precomputed offline by the Python teacher (not planned in your browser). A clamped cubic spline through 4 joint keyframes per joint is tracked by computed torque (nominal arm model, LQR gains). The chain is never measured: the swing runs open loop with respect to it.",
      params: [
        { id: "qp", label: "LQR weight Q_p", min: 2, max: 7, step: 0.1, value: Math.log10(1.6e5), scale: "log10", fmt: (v) => v.toExponential(1) },
        { id: "qv", label: "LQR weight Q_v", min: 1, max: 5, step: 0.1, value: Math.log10(800), scale: "log10", fmt: (v) => v.toExponential(1) },
      ],
    },
    {
      id: "random", label: "Random swing", color: "#dc2626", name: "Random keyframes (no search)",
      note: "Baseline: same tracking controller, but the keyframes are a random draw around the start posture. It shows how much of the hit rate comes from the offline search.",
      params: [],
    },
  ],
  weaknesses: [
    { id: "chainLen", label: "true chain length", min: -25, max: 25, step: 5, value: 0, fmt: fmtPct },
    { id: "chainMass", label: "true chain mass", min: -50, max: 100, step: 10, value: 0, fmt: fmtPct },
    { id: "chainDamp", label: "true chain damping", min: -100, max: 400, step: 50, value: 0, fmt: fmtPct },
    { id: "payload", label: "true tip payload", min: -50, max: 100, step: 10, value: 0, fmt: fmtPct },
    { id: "delay", label: "command delay", min: 0, max: 8, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
  ],
  weaknessNote:
    "The planner's belief is set by \"planner assumes chain\" (plans precomputed offline by the Python teacher for 90 / 100 / 110 % of the 1 m chain); the sliders change the TRUE plant only, so a mismatch such as an assumed 100 % chain on a true 90 % chain makes the swing miss. " +
    "Plan table (Python, 12 targets): 12/12 hits when the assumed chain equals the true one, 6-8/12 when they differ by 10 %. " +
    "In the browser the nominal hit rate is about 91 % (100 seeds) because the table plans start from the mean posture and each episode jitters it by 0.02 rad; the full Python teacher plans from the actual posture and hits 20/20 (9/20 on a 10 % shorter chain). " +
    "Targets are limited to 12 grid cells because plans exist only for those.",
  options: [
    { id: "assumed", label: "planner assumes chain", choices: ["90%", "100%", "110%"], value: "100%" },
    { id: "target", label: "target", choices: ["seed", ...TARGET_IDS], value: "seed" },     // "seed": a random grid cell per episode
  ],
  create: (mj, assets) => new WhipSession(mj, assets),
};
