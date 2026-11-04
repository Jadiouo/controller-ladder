// Toss: JS vs the Python arena (validation/python_toss.json, written by validation/tools/export_catch_toss.py).
// 1) closed-loop trajectory of seed 1000 (same initial state and basket) for the clean case and the 40 ms late-release case;
// 2) outcomes of the 20 calibration seeds 1000-1019 x {clean, release +40 ms, ball mass +30 %} from the Python initial states.
// Usage: node validate_toss.mjs   (writes validation/js_vs_python_toss.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import task from "./js/tasks/toss.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const assets = { xml: rd("./assets/toss.xml"), hold: rd("./assets/toss_nominal_hold.xml"), free: rd("./assets/toss_nominal_free.xml"), cfg: JSON.parse(rd("./assets/toss.json")) };
const py = JSON.parse(rd("./validation/python_toss.json"));
const out = { mujoco_wasm: "mujoco@3.1.16", traj: {}, outcomes: {} };
const weak = { clean: {}, release_delay_40ms: { release: 40 }, ball_mass_p30: { mass: 30 } };
const make = (cname) => { const s = task.create(mj, assets); for (const [k, v] of Object.entries(weak[cname])) s.setWeakness(k, v); return s; };
const initOf = (seed) => py.init[String(seed)];

for (const [cname, ref] of Object.entries(py.traj)) {
  const s = make(cname);
  s.reset(1000, initOf(1000));
  const steps = ref.tau.length;
  let maxQ = 0, maxObj = 0, maxTau = 0, range = 0, firstBig = null;
  for (let k = 0; k < steps; k++) {
    const tau = s.step();
    for (let j = 0; j < 3; j++) maxTau = Math.max(maxTau, Math.abs(tau[j] - ref.tau[k][j]));
    for (let j = 0; j < 5; j++) {
      const e = Math.abs(s.data.qpos[j] - ref.q[k + 1][j]);
      if (j < 3) maxQ = Math.max(maxQ, e); else maxObj = Math.max(maxObj, e);
      range = Math.max(range, Math.abs(ref.q[k + 1][j]));
      if (e > 1e-6 && firstBig === null) firstBig = k;
    }
  }
  out.traj[cname] = { steps, js_done_at_step: s.done ? s.k : null, max_abs_q_err_rad: maxQ, max_abs_obj_err_m: maxObj, max_abs_tau_err_Nm: maxTau, q_range: range, first_step_err_over_1e_6: firstBig };
  s.dispose();
  console.log(cname, out.traj[cname]);
}
for (const [cname, rows] of Object.entries(py.cases)) {
  let nJs = 0, nPy = 0, agree = 0, maxStepDiff = 0;
  const detail = [];
  for (const row of rows) {
    const s = make(cname);
    s.reset(row.seed, initOf(row.seed));
    while (!s.done && s.k < s.nSteps) s.step();
    const ok = s.outcome === "success", okPy = row.outcome === "success";
    nJs += ok; nPy += okPy; agree += ok === okPy;
    maxStepDiff = Math.max(maxStepDiff, Math.abs(s.k - row.steps));
    detail.push({ seed: row.seed, py: row.outcome, js: s.outcome ?? "timeout", py_steps: row.steps, js_steps: s.k });
    s.dispose();
  }
  out.outcomes[cname] = { success_py: nPy, success_js: nJs, seeds: rows.length, outcome_agree: agree, max_abs_step_diff: maxStepDiff, detail };
  console.log(cname, { success_py: nPy, success_js: nJs, agree: `${agree}/${rows.length}`, maxStepDiff });
}
fs.writeFileSync(new URL("./validation/js_vs_python_toss.json", import.meta.url), JSON.stringify(out, null, 1));
