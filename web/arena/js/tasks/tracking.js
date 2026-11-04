// Task plugin: trajectory tracking (the original arena task). Plant and controllers live in ../sim.js (ArmSim, PID, CtcLqr);
// this file only wraps them in the task-plugin interface (see README.md). Behaviour is unchanged.
import { ArmSim, GLYPHS, lqrDoubleIntegrator } from "../sim.js";

export class TrackingSession extends ArmSim {
  constructor(mj, assets) {
    super(mj, assets.xml, assets.refs);
    this.duration = assets.refs.duration;
    Object.assign(this.controllers.ctc, { qp: 1.6e5, qv: 800 });   // CtcLqr keeps only the derived gains; remember the weights for the sliders
  }
  // ---- plugin interface ----
  setController(id) { this.select(id); }
  setParam(ctl, id, v) {
    const c = this.controllers[ctl];
    if (ctl === "pid") {
      c[id] = v;
      if (id === "wn") c.kp = null;       // gains are recomputed from the nominal inertia on the next act()
    } else if (ctl === "ctc") {
      c[id] = v;                            // qp / qv weights
      [c.Kp, c.Kd] = lqrDoubleIntegrator(c.qp, c.qv);
    }
  }
  setWeakness(id, v) {
    if (id === "payload") this.setPayload(this.nominalPayload * (1 + v / 100));
    else if (id === "delay") this.setDelay(v);
  }
  setOption(id, v) { if (id === "glyph") this.glyph = v; }
  controllerInfo(id) {
    if (id !== "ctc") return "";
    const c = this.controllers.ctc;
    return `Kp = ${c.Kp.toFixed(0)}, Kd = ${c.Kd.toFixed(1)}`;
  }
  reset(seed) {
    super.reset();
    this.sumSq = 0;
    this.nErr = 0;
    this.recordNow = true;
    this.maxTip = 0;
  }
  step(hook) {
    const tau = super.step(hook);
    this.recordNow = this.k <= this.nSteps;
    if (this.recordNow) {
      const e = this.errors().tip;
      this.sumSq += e * e; this.nErr++;
      this.maxTip = Math.max(this.maxTip, e);
    }
    return tau;
  }
  get done() { return this.k >= this.nSteps; }
  get outcome() { return this.done ? "complete" : null; }
  get message() { return this.done ? `Figure complete: RMS tip error ${(this.rms() * 1000).toFixed(1)} mm` : ""; }
  rms() { return this.nErr ? Math.sqrt(this.sumSq / this.nErr) : 0; }
  chartValue() { return this.errors().tip; }
  refPath() { return this.ref.EE; }
  target() { return this.referenceAt(this.k).ee; }
  trailPoint() { return this.tipPos(); }
  metrics() {
    return [
      ["time", `${(this.k * this.dt).toFixed(2)} s`],
      ["tip error", `${(this.errors().tip * 1000).toFixed(1)} mm`],
      ["episode RMS", `${(this.rms() * 1000).toFixed(1)} mm`],
      ["worst error", `${(this.maxTip * 1000).toFixed(1)} mm`],
      ["torque", this.lastTau ? `${this.lastTau.map((x) => x.toFixed(0)).join(", ")} Nm` : "-"],
    ];
  }
  geomColor() { return null; }
  grabSegments() {
    const x = this.data.xpos, p = (b) => [x[3 * b], x[3 * b + 2]];   // bodies: world, l1, l2, tip
    return [{ body: 1, a: p(1), b: p(2) }, { body: 2, a: p(2), b: p(3) }];
  }
  dispose() { this.data?.delete(); this.model?.delete(); this.arm.data.delete(); this.arm.model.delete(); }
}

export default {
  id: "tracking",
  title: "Trace a figure",
  summary:
    "The arm's tip must trace a figure (S, 8, M, Z, heart, circle) as accurately as it can. Compare a plain PID with gravity compensation " +
    "against computed torque with LQR-chosen gains, tune them, add model error or latency, and grab the arm to disturb it.",
  assets: { xml: "assets/arm2.xml", refs: "assets/tracking_refs.json" },
  camera: { target: [0.38, -0.02], distance: 2.0, fov: 32 },
  chart: { label: "tip error (mm)", scale: 1000, signed: false, skip: 0.3, maxCap: 1.0, minRange: 0.02 },
  legendNote: "line colour = controller active at that moment; red dot = target, grey line = figure, black = actual tip path",
  interaction: { fmax: 120, k: 220, damp: 18, hitRadius: 0.14 },
  controllers: [
    {
      id: "pid", label: "PID", color: "#2563eb", name: "PID + gravity comp.",
      note: "Per-joint Kp = w_n² M_ii, Kd = 2 w_n M_ii (critically damped on the nominal inertia). No inverse dynamics.",
      params: [
        { id: "wn", label: "bandwidth w_n", min: 4, max: 60, step: 1, value: 20, fmt: (v) => v.toFixed(0) + " rad/s" },
        { id: "ki", label: "integral Ki", min: 0, max: 400, step: 10, value: 0, fmt: (v) => v.toFixed(0) },
        { id: "gravity", label: "gravity comp.", type: "checkbox", value: true },
      ],
    },
    {
      id: "ctc", label: "CTC + LQR", color: "#dc2626", name: "CTC + LQR",
      note: "Computed torque: tau = ID(q, qd, qdd_ref + Kp e + Kd ed) with the nominal model; [Kp, Kd] is the LQR gain of the per-joint double integrator.",
      params: [
        { id: "qp", label: "LQR weight Q_p", min: 2, max: 7, step: 0.1, value: Math.log10(1.6e5), scale: "log10", fmt: (v) => v.toExponential(1) },
        { id: "qv", label: "LQR weight Q_v", min: 1, max: 5, step: 0.1, value: Math.log10(800), scale: "log10", fmt: (v) => v.toExponential(1) },
      ],
      info: true,
    },
  ],
  weaknesses: [
    { id: "payload", label: "true payload", min: -50, max: 100, step: 5, value: 0, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(0) + " %" },
    { id: "delay", label: "command delay", min: 0, max: 8, step: 1, value: 0, fmt: (v) => `${v} steps (${(v * 5).toFixed(0)} ms)` },
  ],
  weaknessNote: "The controllers always assume the nominal 1.5 kg payload; \"true payload\" changes the plant only. Delay is an actuator latency of whole 5 ms control periods.",
  options: [{ id: "glyph", label: "figure", choices: GLYPHS, value: "S" }],
  create: (mj, assets) => new TrackingSession(mj, assets),
};
