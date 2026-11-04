// Pole balance: JS vs the Python arena (validation/python_pole.json, written by validation/tools/export_pole.py).
// 1) LQR gain computed in JS vs scipy;  2) closed-loop trajectory of seed 1000 (same initial state, nominal plant);
// 3) outcome of 20 seeds x {PID, CTC+LQR} x {clean, hard = payload +30 % and 2-step delay} from the Python initial states.
// Usage: node validate_pole.mjs   (writes validation/js_vs_python_pole.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import pole, { poleGain } from "./js/tasks/pole_balance.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const cfg = JSON.parse(rd("./assets/pole_balance.json"));
const py = JSON.parse(rd("./validation/python_pole.json"));
const assets = { xml: rd("./assets/pole_balance.xml"), nominal: rd("./assets/pole_balance_nominal.xml"), cfg };
const out = { mujoco_wasm: "mujoco@3.1.16", gain: {}, traj: {}, outcomes: {} };

const K = poleGain(cfg.pole_length, cfg.gravity, cfg.lqr_q, cfg.lqr_r);
out.gain = { js: K, python: cfg.lqr_K, max_abs_diff: Math.max(...K.map((k, i) => Math.abs(k - cfg.lqr_K[i]))), max_rel_diff: Math.max(...K.map((k, i) => Math.abs(k - cfg.lqr_K[i]) / Math.abs(cfg.lqr_K[i]))) };
console.log("LQR gain", out.gain);

const session = (ctl, hard) => {
  const s = pole.create(mj, assets);
  if (hard) { s.setWeakness("payload", 30); s.setWeakness("delay", 2); }
  s.setController(ctl === "pid" ? "pid" : "ctc");
  return s;
};
const initOf = (seed) => py.init[String(seed)];
const ctlName = (k) => (k.endsWith("pid") ? "pid" : "ctc");

for (const [key, ref] of Object.entries(py.traj)) {          // trajectories: clean, seed 1000, all steps Python ran
  const s = session(ctlName(key), false);
  s.reset(1000, initOf(1000));
  const steps = ref.tau.length;
  let maxQ = 0, maxTau = 0, range = 0, firstBig = null;
  for (let k = 0; k < steps; k++) {
    const tau = s.step();
    for (let j = 0; j < 2; j++) maxTau = Math.max(maxTau, Math.abs(tau[j] - ref.tau[k][j]));
    for (let j = 0; j < 3; j++) { const e = Math.abs(s.data.qpos[j] - ref.q[k + 1][j]); maxQ = Math.max(maxQ, e); range = Math.max(range, Math.abs(ref.q[k + 1][j])); if (e > 1e-6 && firstBig === null) firstBig = k; }
  }
  s.dispose();
  out.traj[key] = { steps, max_abs_q_err_rad: maxQ, max_abs_tau_err_Nm: maxTau, q_range_rad: range, first_step_err_over_1e_6: firstBig };
  console.log(key, out.traj[key]);
}

for (const [key, rows] of Object.entries(py.cases)) {         // outcomes over seeds
  const [cname, cn] = key.split("/");
  let agree = 0, balancedJs = 0, balancedPy = 0, maxStepDiff = 0;
  const detail = [];
  for (const row of rows) {
    const s = session(ctlName(cn), cname === "hard");
    s.reset(row.seed, initOf(row.seed));
    while (!s.done) s.step();
    const ok = s.outcome === "success", okPy = row.outcome === "timeout";
    balancedJs += ok; balancedPy += okPy; agree += ok === okPy;
    maxStepDiff = Math.max(maxStepDiff, Math.abs(s.k - row.steps));
    s.dispose();
    detail.push({ seed: row.seed, py: row.outcome, js: s.outcome, py_steps: row.steps, js_steps: s.k });
  }
  out.outcomes[key] = { balanced_py: balancedPy, balanced_js: balancedJs, seeds: rows.length, outcome_agree: agree, max_abs_step_diff: maxStepDiff, detail };
  console.log(key, { balanced_py: balancedPy, balanced_js: balancedJs, agree: `${agree}/${rows.length}`, maxStepDiff });
}
fs.writeFileSync(new URL("./validation/js_vs_python_pole.json", import.meta.url), JSON.stringify(out, null, 1));
