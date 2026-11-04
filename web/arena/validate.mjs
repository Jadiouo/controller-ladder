// Numerical comparison of the JS controllers against the Python arena (validation/python_traj.json).
// Usage: node validate.mjs   (writes validation/js_vs_python.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import tracking from "./js/tasks/tracking.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const refs = JSON.parse(rd("./assets/tracking_refs.json"));
const py = JSON.parse(rd("./validation/python_traj.json"));
const assets = { xml: rd("./assets/arm2.xml"), refs };   // through the task-plugin interface (js/tasks/tracking.js)

const result = { mujoco_wasm: "mujoco@3.1.16", glyph: py.glyph, steps: py.steps, cases: {} };
for (const [key, ref] of Object.entries(py.cases)) {
  const [cname, ctl] = key.split("/");
  const sim = tracking.create(mj, assets);
  sim.setOption("glyph", py.glyph);
  if (cname === "hard") { sim.setWeakness("payload", 30); sim.setWeakness("delay", 2); }
  sim.setController(ctl === "pid" ? "pid" : "ctc");
  sim.reset(1000);
  const Q = [Array.from(sim.data.qpos.slice(0, 2))];
  const TAU = [];
  for (let k = 0; k < py.steps; k++) { TAU.push(sim.step()); Q.push(Array.from(sim.data.qpos.slice(0, 2))); }
  let maxQ = 0, sumQ = 0, maxTau = 0, rangeQ = 0;
  Q.forEach((q, i) => { for (let j = 0; j < 2; j++) { const e = Math.abs(q[j] - ref.q[i][j]); maxQ = Math.max(maxQ, e); sumQ += e * e; rangeQ = Math.max(rangeQ, Math.abs(ref.q[i][j])); } });
  TAU.forEach((t, i) => { for (let j = 0; j < 2; j++) maxTau = Math.max(maxTau, Math.abs(t[j] - ref.tau[i][j])); });
  result.cases[key] = { max_abs_q_err_rad: maxQ, rms_q_err_rad: Math.sqrt(sumQ / (Q.length * 2)), max_abs_tau_err_Nm: maxTau, joint_range_rad: rangeQ };
  console.log(key, result.cases[key]);
}
fs.writeFileSync(new URL("./validation/js_vs_python.json", import.meta.url), JSON.stringify(result, null, 1));
