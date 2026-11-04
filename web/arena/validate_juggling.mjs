// Racket juggling: JS vs the Python spike03b env (validation/python_juggling.json + python_juggling_noise.bin, written by validation/tools/export_juggling.py).
// 1) seed-1000 nominal closed-loop trajectory, same initial state and the exact sensor-noise samples;  2) consecutive hits of 20 CAL seeds (1000-1019),
// nominal and restitution 0.4, untuned teacher: (a) JS driven by the exact Python noise, (b) JS with its own Gaussian noise (different RNG).
// Usage: node validate_juggling.mjs   (writes validation/js_vs_python_juggling.json)
import fs from "node:fs";
import loadMujoco from "./vendor/mujoco.js";
import task from "./js/tasks/juggling.js";

const rd = (p) => fs.readFileSync(new URL(p, import.meta.url), "utf8");
const mj = await loadMujoco();
const cfg = JSON.parse(rd("./assets/juggling.json"));
const py = JSON.parse(rd("./validation/python_juggling.json"));
const bin = fs.readFileSync(new URL("./validation/python_juggling_noise.bin", import.meta.url));
const allNoise = new Float64Array(bin.buffer, bin.byteOffset, bin.byteLength / 8);
const noiseOf = (seed) => allNoise.subarray(2 * py.noise_offset[seed], 2 * (py.noise_offset[seed] + py.noise_len[seed]));
const assets = { xml: rd("./assets/juggling.xml"), nominal: rd("./assets/juggling_nominal.xml"), cfg };
const mk = (e) => { const s = task.create(mj, assets); if (e !== 0.8) s.setWeakness("restitution", e); return s; };
const out = { mujoco_wasm: "mujoco@3.1.16", traj: {}, hits: {} };

// ---- 1) trajectory ----
{
  const s = mk(0.8), ref = py.traj, ic = py.init[py.seeds.indexOf(1000)];
  s.reset(1000, ic, noiseOf(1000));
  let hitsSeen = 0, firstHit = null;
  const errAt = [];   // max |state error| per step (arm q, qd, ball x z)
  let maxTau = 0;
  for (let k = 0; k < ref.steps; k++) {
    let e = 0;
    for (let j = 0; j < 5; j++) e = Math.max(e, Math.abs(s.data.qpos[j] - ref.q_qd[k][j]));
    for (let j = 0; j < 5; j++) e = Math.max(e, Math.abs(s.data.qvel[j] - ref.q_qd[k][5 + j]));
    errAt.push(e);
    const tau = s.step();
    for (let j = 0; j < 3; j++) maxTau = Math.max(maxTau, Math.abs(tau[j] - ref.tau[k][j]));
    if (s.hits > hitsSeen && firstHit === null) firstHit = k;
    hitsSeen = s.hits;
  }
  const idx = (th) => errAt.findIndex((e) => e > th);
  const upTo = (n) => Math.max(...errAt.slice(0, n));
  out.traj = {
    steps: ref.steps, first_hit_step: firstHit, max_state_err_before_first_hit: firstHit ? upTo(firstHit) : null, max_state_err_all: upTo(ref.steps),
    first_step_err_over_1e_9: idx(1e-9), first_step_err_over_1e_6: idx(1e-6), first_step_err_over_1e_3: idx(1e-3),
    err_at_steps: Object.fromEntries([100, 200, 400, 600, 800, 1000, 1200, 1499].map((k) => [k, errAt[k]])), max_tau_err_before_first_hit: null, js_hits: s.hits,
  };
  // torque error before the first hit
  const s2 = mk(0.8); s2.reset(1000, ic, noiseOf(1000));
  let mt = 0; for (let k = 0; k < (firstHit ?? 0); k++) { const t = s2.step(); for (let j = 0; j < 3; j++) mt = Math.max(mt, Math.abs(t[j] - ref.tau[k][j])); }
  out.traj.max_tau_err_before_first_hit = mt; s2.dispose(); s.dispose();
  console.log("trajectory", out.traj);
}

// ---- 2) hit counts ----
const run = (e, seed, exact) => {
  const s = mk(e), i = py.seeds.indexOf(seed);
  s.reset(seed, py.init[i], exact ? noiseOf(seed) : null);
  while (!s.done) s.step();
  const r = { hits: s.hits, fail: s.fail, steps: s.k };
  s.dispose();
  return r;
};
const stats = (a) => { const b = [...a].sort((x, y) => x - y), n = b.length, m = a.reduce((x, y) => x + y, 0) / n; return { median: (b[(n - 1) >> 1] + b[n >> 1]) / 2, mean: m, min: b[0], max: b[n - 1] }; };
for (const [name, e] of [["nominal", 0.8], ["e0.4", 0.4]]) {
  const rows = py.cases[name], exact = [], own = [];
  for (const row of rows) { exact.push(run(e, row.seed, true)); own.push(run(e, row.seed, false)); }
  const pyH = rows.map((r) => r.hits), exH = exact.map((r) => r.hits), owH = own.map((r) => r.hits);
  const mad = (a, b) => a.reduce((s, x, i) => s + Math.abs(x - b[i]), 0) / a.length;
  out.hits[name] = {
    python: stats(pyH), js_exact_noise: stats(exH), js_own_noise: stats(owH),
    exact_same_count: exH.filter((h, i) => h === pyH[i]).length, exact_mean_abs_diff: mad(exH, pyH), exact_max_abs_diff: Math.max(...exH.map((h, i) => Math.abs(h - pyH[i]))),
    own_mean_abs_diff: mad(owH, pyH), py_fail: rows.map((r) => r.fail), js_exact_fail: exact.map((r) => r.fail), js_own_fail: own.map((r) => r.fail),
    detail: rows.map((r, i) => ({ seed: r.seed, py: r.hits, js_exact: exH[i], js_own: owH[i] })),
  };
  const o = out.hits[name];
  console.log(name, "python", o.python, "js exact", o.js_exact_noise, "js own", o.js_own_noise, "same", o.exact_same_count, "mean|d|", o.exact_mean_abs_diff.toFixed(2), "max|d|", o.exact_max_abs_diff);
}
fs.writeFileSync(new URL("./validation/js_vs_python_juggling.json", import.meta.url), JSON.stringify(out, null, 1));
