// Catch: JS vs the Python arena (validation/python_catch.json, written by validation/tools/export_catch_toss.py).
// 1) closed-loop trajectory of seed 1000 (same initial state) for the clean case and the 10 ms ball-observation-delay case;
// 2) outcomes of the 20 calibration seeds 1000-1019 x {clean, obs delay 2 steps, ball mass +30 %} from the Python initial states.
// Usage: node validate_catch.mjs   (writes validation/js_vs_python_catch.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import task from "./js/tasks/catch.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const assets = { xml: rd("./assets/catch.xml"), nominal: rd("./assets/catch_nominal.xml"), cfg: JSON.parse(rd("./assets/catch.json")) };
const py = JSON.parse(rd("./validation/python_catch.json"));
const out = { mujoco_wasm: "mujoco@3.1.16", traj: {}, outcomes: {} };
const weak = { clean: {}, obs_delay_2steps_10ms: { delay: 2 }, ball_mass_p30: { mass: 30 } };
const make = (cname) => { const s = task.create(mj, assets); for (const [k, v] of Object.entries(weak[cname])) s.setWeakness(k, v); return s; };
const initOf = (seed) => py.init[String(seed)];

for (const [cname, ref] of Object.entries(py.traj)) {
  const s = make(cname);
  s.reset(1000, initOf(1000));
  const steps = ref.tau.length;
  let maxQ = 0, maxBall = 0, maxTau = 0, range = 0, firstBig = null, jsSteps = 0;
  for (let k = 0; k < steps; k++) {
    const tau = s.step();
    jsSteps++;
    for (let j = 0; j < 3; j++) maxTau = Math.max(maxTau, Math.abs(tau[j] - ref.tau[k][j]));
    for (let j = 0; j < 5; j++) {
      const e = Math.abs(s.data.qpos[j] - ref.q[k + 1][j]);
      if (j < 3) maxQ = Math.max(maxQ, e); else maxBall = Math.max(maxBall, e);
      range = Math.max(range, Math.abs(ref.q[k + 1][j]));
      if (e > 1e-6 && firstBig === null) firstBig = k;
    }
    if (s.done && k + 1 < steps) { /* Python also stops at the terminal step; lengths must agree */ }
  }
  out.traj[cname] = { steps, js_done_at_step: s.done ? s.k : null, max_abs_q_err_rad: maxQ, max_abs_ball_err_m: maxBall, max_abs_tau_err_Nm: maxTau, q_range: range, first_step_err_over_1e_6: firstBig };
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
fs.writeFileSync(new URL("./validation/js_vs_python_catch.json", import.meta.url), JSON.stringify(out, null, 1));
