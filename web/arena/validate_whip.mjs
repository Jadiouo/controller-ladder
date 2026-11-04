// Whip: JS vs the Python arena (validation/python_whip.json, written by validation/tools/export_whip.py).
// 1) closed-loop trajectories (13 DoF = 3 arm + 10 chain) with the same start posture, target and keyframes, nominal and +-10 % true chain;
// 2) outcomes + closest approach of the 108-case plan-table grid (12 targets x 3 assumed x 3 true chain lengths);
// 3) 20 calibration seeds 1000-1019, the Python teacher's own per-seed plan executed in JS: nominal and true chain -10 %.
// Usage: node validate_whip.mjs   (writes validation/js_vs_python_whip.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import whip from "./js/tasks/whip.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const cfg = JSON.parse(rd("./assets/whip.json"));
const py = JSON.parse(rd("./validation/python_whip.json"));
const assets = { xml: rd("./assets/whip.xml"), nominal: rd("./assets/whip_nominal.xml"), cfg };
const out = { mujoco_wasm: "mujoco@3.1.16", traj: {}, grid: {}, cal: {} };

const session = (trueScale) => {
  const s = whip.create(mj, assets);
  s.setWeakness("chainLen", (trueScale - 1) * 100);
  return s;
};
const cellOf = (id) => cfg.targets.find((t) => t.id === id);
const runEpisode = (s, init, knots) => {
  s.setPlan(knots);
  s.reset(init.seed ?? 1000, init);
  while (!s.done) s.step();
  return s;
};

// 1) trajectories
for (const [key, ref] of Object.entries(py.traj)) {
  const s = session(ref.true);
  s.setPlan(cfg.plans[`${key.split("|")[0]}|${ref.assumed.toFixed(1)}`].knots);
  s.reset(1000, { q0: ref.q0, target: ref.target, cell: cellOf(key.split("|")[0]) });
  const steps = ref.tau.length;
  let maxQ = 0, maxTau = 0, maxQarm = 0, range = 0, firstBig = null, errAt = {};
  for (let k = 0; k < steps; k++) {
    const tau = s.step();
    for (let j = 0; j < 3; j++) maxTau = Math.max(maxTau, Math.abs(tau[j] - ref.tau[k][j]));
    for (let j = 0; j < 13; j++) {
      const e = Math.abs(s.data.qpos[j] - ref.q[k + 1][j]);
      maxQ = Math.max(maxQ, e); if (j < 3) maxQarm = Math.max(maxQarm, e);
      range = Math.max(range, Math.abs(ref.q[k + 1][j]));
      if (e > 1e-6 && firstBig === null) firstBig = k;
    }
    if ([100, 200, 300, 400].includes(k + 1)) errAt[k + 1] = maxQ;
  }
  s.dispose();
  out.traj[key] = { steps, max_abs_q_err_rad: maxQ, max_abs_arm_q_err_rad: maxQarm, max_abs_tau_err_Nm: maxTau, q_range_rad: range, first_step_err_over_1e_6: firstBig, running_max_err_at_step: errAt };
  console.log(key, out.traj[key]);
}

// 2) plan-table grid
{
  const acc = {};
  let agree = 0, maxD = 0, maxDsucc = 0;
  for (const g of py.grid) {
    const s = session(g.true);
    runEpisode(s, { seed: g.seed, q0: g.q0, target: [cellOf(g.target).x, cellOf(g.target).z], cell: cellOf(g.target) }, cfg.plans[`${g.target}|${g.assumed.toFixed(1)}`].knots);
    const ok = s.outcome === "success", okPy = g.outcome === "success";
    agree += ok === okPy;
    maxD = Math.max(maxD, Math.abs(s.minDist - g.min_dist));
    const key = `assumed ${g.assumed} / true ${g.true}`;
    acc[key] ??= { py: 0, js: 0, n: 0 };
    acc[key].py += okPy; acc[key].js += ok; acc[key].n += 1;
    s.dispose();
  }
  out.grid = { n: py.grid.length, outcome_agree: agree, py_success: py.grid.filter((g) => g.outcome === "success").length, max_abs_min_dist_diff_m: maxD, by_condition: acc };
  console.log("grid", { ...out.grid, by_condition: undefined }, acc);
}

// 3) calibration seeds with the Python per-seed plans
for (const [cname, rows] of Object.entries(py.cal)) {
  const trueScale = cname === "nominal" ? 1.0 : 0.9;
  let agree = 0, jsOk = 0, pyOk = 0, maxD = 0;
  const detail = [];
  for (const r of rows) {
    const s = session(trueScale);
    runEpisode(s, { seed: r.seed, q0: r.q0, target: r.target }, r.knots);
    const ok = s.outcome === "success", okPy = r.outcome === "success";
    agree += ok === okPy; jsOk += ok; pyOk += okPy;
    maxD = Math.max(maxD, Math.abs(s.minDist - r.min_dist));
    detail.push({ seed: r.seed, py: r.outcome, js: s.outcome, py_steps: r.steps, js_steps: s.k, py_min: r.min_dist, js_min: s.minDist });
    s.dispose();
  }
  out.cal[cname] = { success_py: pyOk, success_js: jsOk, seeds: rows.length, outcome_agree: agree, max_abs_min_dist_diff_m: maxD, detail };
  console.log(cname, { success_py: pyOk, success_js: jsOk, agree: `${agree}/${rows.length}`, maxD });
}
fs.writeFileSync(new URL("./validation/js_vs_python_whip.json", import.meta.url), JSON.stringify(out, null, 1));
