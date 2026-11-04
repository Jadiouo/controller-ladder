// Push-T: JS vs the Python arena (validation/python_push_t.json, written by validation/tools/export_push_t.py).
// 1) geometry/planner unit checks (coverage, first plan of seed 1000 vs the Python plan log);
// 2) closed-loop trajectory of seed 1000 (nominal) from the exact Python initial state: first step where JS and Python diverge, error growth;
// 3) outcomes of 20 calibration seeds (1000-1019) from the Python initial states: nominal and sensor delay 30 ms (6 steps), plus
//    friction +50 % / block mass +100 % / COM offset 20 mm (JS only; Python numbers for these come from the paper's runs, not re-run here).
// Usage: node validate_push_t.mjs   (writes validation/js_vs_python_push_t.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import pushT from "./js/tasks/push_t.js";
import * as G from "./js/tasks/push_t_plan.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const cfg = JSON.parse(rd("./assets/push_t.json"));
const py = JSON.parse(rd("./validation/python_push_t.json"));
const assets = { xml: rd("./assets/push_t.xml"), nominal: rd("./assets/push_t_nominal.xml"), cfg };
const out = { mujoco_wasm: "mujoco@3.1.16", geom: {}, traj: {}, outcomes: {} };

out.geom = { rho2_diff: Math.abs(G.RHO2 - cfg.geom.RHO2), area_diff: Math.abs(G.AREA - cfg.geom.AREA), coverage_goal_vs_goal: G.coverage(cfg.goal, cfg.goal), n_primitives: new G.PushPlanner(cfg.mu, G.RHO2, cfg.pusher_radius, cfg.goal).M };
console.log("geometry", out.geom);

const session = (weak = {}, ctl = "teacher") => {
  const s = pushT.create(mj, assets);
  for (const [k, v] of Object.entries(weak)) s.setWeakness(k, v);
  s.setController(ctl);
  return s;
};
const initOf = (row) => ({ q: [row.init.q[0], row.init.q[1], row.init.q[2], row.init.q[3], row.init.q[4]] });

{ // trajectory seed 1000, nominal
  const ref = py.traj, row = py.cases.nominal.find((r) => r.seed === 1000);
  const s = session();
  s.reset(1000, initOf(row));
  const steps = ref.tau.length;
  let maxQ = 0, maxTau = 0, first = null, firstTau = null;
  const errAt = {};
  const t0 = performance.now();
  for (let k = 0; k < steps; k++) {
    const tau = s.step();
    for (let j = 0; j < 2; j++) { const e = Math.abs(tau[j] - ref.tau[k][j]); maxTau = Math.max(maxTau, e); if (e > 1e-3 && firstTau === null) firstTau = k; }
    let e = 0;
    for (let j = 0; j < 5; j++) e = Math.max(e, Math.abs(s.data.qpos[j] - ref.q[k + 1][j]));
    maxQ = Math.max(maxQ, e);
    if (e > 1e-6 && first === null) first = k;
    if ((k + 1) % 250 === 0) errAt[k + 1] = maxQ;
    if (s.done && k + 1 >= steps) break;
  }
  const secs = (performance.now() - t0) / 1000;
  const logJs = s.controllers.teacher.planLog.map((p) => ({ k: p.k, prim: p.prim, cost: p.cost }));
  const same = logJs.length === row.plans.length && logJs.every((p, i) => p.k === row.plans[i].k && p.prim === row.plans[i].prim);
  out.traj = { steps, max_abs_q_err: maxQ, first_step_q_err_over_1e_6: first, first_step_tau_err_over_1e_3: firstTau, running_max_q_err_at_step: errAt,
    plans_py: row.plans.length, plans_js: logJs.length, plan_log_identical: same, js_outcome: s.outcome, py_outcome: row.outcome, js_steps: s.k, py_steps: row.steps, js_seconds: secs };
  console.log("trajectory seed 1000", out.traj);
  s.dispose();
}

const cases = [["nominal", {}, "nominal"], ["obs_delay_30ms", { delay: 6 }, "obs_delay_30ms"]];
for (const [name, weak, pykey] of cases) {
  const rows = py.cases[pykey];
  let agree = 0, okJs = 0, okPy = 0, maxStep = 0, nSame = 0;
  const detail = [];
  for (const row of rows) {
    const s = session(weak);
    s.reset(row.seed, initOf(row));
    while (!s.done && s.k < s.nSteps) s.step();
    const ok = s.outcome === "success", okp = row.outcome === "success";
    agree += ok === okp; okJs += ok; okPy += okp;
    if (ok && okp) { maxStep = Math.max(maxStep, Math.abs(s.k - row.steps)); nSame += s.k === row.steps; }
    detail.push({ seed: row.seed, py: row.outcome, js: s.outcome, py_steps: row.steps, js_steps: s.k, py_cov: row.coverage, js_cov: s.cov, plans_py: row.plans.length, plans_js: s.controllers.teacher.planLog.length });
    s.dispose();
  }
  out.outcomes[name] = { success_py: okPy, success_js: okJs, seeds: rows.length, outcome_agree: agree, success_both_same_step_count: nSame, success_both_max_step_diff: maxStep, detail };
  console.log(name, { success_py: okPy, success_js: okJs, agree: `${agree}/${rows.length}`, nSame, maxStep });
}
// further perturbations of the paper's push_t sweep (Python re-run here with the same 20 seeds; JS pose noise uses its own RNG, so only the statistics are comparable)
for (const [name, weak] of [["friction+50%", { friction: 50 }], ["mass+100%", { mass: 100 }], ["com(20,0)mm", { comx: 20 }], ["noise 5mm", { noise: 5 }]]) {
  let okJs = 0, okPy = 0;
  for (const row of py.cases[name]) {
    const s = session(weak);
    s.reset(row.seed, initOf(row));
    while (!s.done && s.k < s.nSteps) s.step();
    okJs += s.outcome === "success"; okPy += row.outcome === "success";
    s.dispose();
  }
  out.outcomes[name] = { success_py: okPy, success_js: okJs, seeds: py.cases[name].length };
  console.log(name, { success_py: okPy, success_js: okJs });
}
fs.writeFileSync(new URL("./validation/js_vs_python_push_t.json", import.meta.url), JSON.stringify(out, null, 1));
