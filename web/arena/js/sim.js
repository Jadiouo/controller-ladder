// Pure simulation + control logic (no DOM): runs in the browser and in node (used by validate.mjs).
// JS port of arena/env.py (step order), arena/controllers/pid.py and arena/controllers/ctc_lqr.py (tracking task only).

export const GLYPHS = ["S", "8", "M", "Z", "heart", "circle"];

export async function loadMujocoModule(loader) {
  return await loader();
}

function writeModel(mj, xml, path) {
  mj.FS.writeFile(path, xml);
  return mj.MjModel.mj_loadXML(path);
}

/** Nominal arm model used by the controllers (arena/controllers/_arm.py ArmModel). */
export class ArmModel {
  constructor(mj, xml) {
    this.mj = mj;
    this.model = writeModel(mj, xml, "/nominal.xml");
    this.data = new mj.MjData(this.model);
    this.n = this.model.nv;
  }
  inverse(q, qd, qdd) {
    const { mj, data, model } = this;
    for (let i = 0; i < this.n; i++) {
      data.qpos[i] = q[i];
      data.qvel[i] = qd[i];
      data.qacc[i] = qdd[i];
      data.qfrc_applied[i] = 0;
      data.ctrl[i] = 0;
    }
    mj.mj_inverse(model, data);
    return Array.from(data.qfrc_inverse.slice(0, this.n));
  }
  gravity(q) {
    const z = new Array(this.n).fill(0);
    return this.inverse(q, z, z);
  }
  /** diagonal of the mass matrix: M e_i = inverse(q, 0, e_i) - inverse(q, 0, 0) */
  massDiag(q) {
    const z = new Array(this.n).fill(0);
    const g = this.inverse(q, z, z);
    const out = [];
    for (let i = 0; i < this.n; i++) {
      const e = z.slice();
      e[i] = 1;
      out.push(this.inverse(q, z, e)[i] - g[i]);
    }
    return out;
  }
}

/** LQR gain [Kp, Kd] of the double integrator e'' = u with Q = diag(qp, qv), R = 1 (closed-form Riccati solution). */
export function lqrDoubleIntegrator(qp, qv) {
  const p12 = Math.sqrt(qp);
  return [p12, Math.sqrt(qv + 2 * p12)];
}

const clip = (x, lim) => Math.max(-lim, Math.min(lim, x));

export class PID {
  constructor(arm, tauLimit, dt, { wn = 20, ki = 0, gravity = true, iClip = 0.5 } = {}) {
    Object.assign(this, { arm, tauLimit, dt, wn, ki, gravity, iClip });
    this.reset();
  }
  reset() {
    this.z = new Array(this.arm.n).fill(0);
    this.kp = this.kd = null;
  }
  act(obs) {
    const n = this.arm.n;
    if (this.kp === null) {
      const m = this.arm.massDiag(obs.ref.q);
      this.kp = m.map((x) => this.wn * this.wn * x);
      this.kd = m.map((x) => 2 * this.wn * x);
    }
    const tau = new Array(n);
    const g = this.gravity ? this.arm.gravity(obs.q) : new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const e = obs.ref.q[i] - obs.q[i];
      this.z[i] = clip(this.z[i] + e * this.dt, this.iClip);
      tau[i] = clip(this.kp[i] * e + this.kd[i] * (obs.ref.qd[i] - obs.qd[i]) + this.ki * this.z[i] + g[i], this.tauLimit);
    }
    return tau;
  }
}

export class CtcLqr {
  constructor(arm, tauLimit, { qp = 1.6e5, qv = 800 } = {}) {
    Object.assign(this, { arm, tauLimit });
    [this.Kp, this.Kd] = lqrDoubleIntegrator(qp, qv);
  }
  reset() {}
  act(obs) {
    const n = this.arm.n;
    const v = new Array(n);
    for (let i = 0; i < n; i++)
      v[i] = obs.ref.qdd[i] + this.Kp * (obs.ref.q[i] - obs.q[i]) + this.Kd * (obs.ref.qd[i] - obs.qd[i]);
    return this.arm.inverse(obs.q, obs.qd, v).map((x) => clip(x, this.tauLimit));
  }
}

/** The plant + task loop. `payload` and `delay` are the weaknesses (PayloadError, Delay of arena/perturb/effects.py). */
export class ArmSim {
  constructor(mj, xml, refs) {
    this.mj = mj;
    this.xml = xml;
    this.refs = refs;
    this.dt = refs.dt;
    this.substeps = refs.substeps;
    this.tauMax = refs.tau_max;
    this.nominalPayload = refs.payload_mass;
    this.arm = new ArmModel(mj, xml);
    this.payload = refs.payload_mass;
    this.delaySteps = 0;
    this.glyph = "S";
    this.model = null;
    this.data = null;
    this._buildPlant(this.payload);
    this.controllers = {
      pid: new PID(this.arm, this.tauMax, this.dt),
      ctc: new CtcLqr(this.arm, this.tauMax),
    };
    this.active = "pid";
    this.reset();
  }

  _plantXml(payload) {
    return this.xml.replace(/(<geom type="sphere"[^>]*mass=")[^"]*(")/, `$1${payload}$2`);
  }

  _buildPlant(payload) {
    const old = this.data ? { q: Array.from(this.data.qpos), v: Array.from(this.data.qvel) } : null;
    if (this.data) this.data.delete();
    if (this.model) this.model.delete();
    this.model = writeModel(this.mj, this._plantXml(payload), "/plant.xml");
    this.data = new this.mj.MjData(this.model);
    if (old) {
      for (let i = 0; i < old.q.length; i++) {
        this.data.qpos[i] = old.q[i];
        this.data.qvel[i] = old.v[i];
      }
      this.mj.mj_forward(this.model, this.data);
    }
    this.tipSite = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_SITE.value, "tip");
  }

  setPayload(mass) {
    this.payload = mass;
    this._buildPlant(mass);
  }
  setDelay(steps) {
    this.delaySteps = steps | 0;
  }
  setGlyph(g) {
    this.glyph = g;
    this.reset();
  }
  select(name) {
    this.active = name;
  }
  get ctrl() {
    return this.controllers[this.active];
  }

  reset() {
    const ref = this.refs.glyphs[this.glyph];
    this.ref = ref;
    this.nSteps = ref.Q.length;
    this.k = 0;
    this._hist = [];
    this.mj.mj_resetData(this.model, this.data);
    for (let i = 0; i < 2; i++) {
      this.data.qpos[i] = ref.q0[i];
      this.data.qvel[i] = ref.qd0[i];
      this.data.xfrc_applied.fill(0);
    }
    this.mj.mj_forward(this.model, this.data);
    for (const c of Object.values(this.controllers)) c.reset();
    this.obs = this._observe();
  }

  referenceAt(k) {
    const j = Math.min(k, this.nSteps - 1);
    const r = this.ref;
    return { q: r.Q[j], qd: r.Qd[j], qdd: r.Qdd[j], ee: r.EE[j] };
  }

  _observe() {
    return { k: this.k, q: Array.from(this.data.qpos.slice(0, 2)), qd: Array.from(this.data.qvel.slice(0, 2)), ref: this.referenceAt(this.k) };
  }

  tipPos() {
    const s = this.data.site_xpos;
    return [s[3 * this.tipSite], s[3 * this.tipSite + 2]];
  }

  /** one control period; `extra(sim)` is called before every physics substep (mouse force etc.) */
  step(extra) {
    const cmd = this.ctrl.act(this.obs);
    // Delay.on_command: arena/perturb/effects.py _pick
    this._hist.push(cmd.slice());
    const out = this._hist[Math.max(0, this._hist.length - 1 - this.delaySteps)];
    if (this._hist.length > this.delaySteps + 2) this._hist.shift();
    const tau = out.map((x) => clip(x, this.tauMax));
    const d = this.data;
    d.ctrl[0] = tau[0];
    d.ctrl[1] = tau[1];
    for (let s = 0; s < this.substeps; s++) {
      if (extra) extra(this);
      this.mj.mj_step(this.model, d);
    }
    this.k += 1;
    this.obs = this._observe();
    this.lastTau = tau;
    return tau;
  }

  /** tip error to the reference figure (m) and joint error norm (rad) at the current state */
  errors() {
    const r = this.referenceAt(this.k);
    const p = this.tipPos();
    return {
      tip: Math.hypot(p[0] - r.ee[0], p[1] - r.ee[1]),
      joint: Math.hypot(this.data.qpos[0] - r.q[0], this.data.qpos[1] - r.q[1]),
    };
  }
}
