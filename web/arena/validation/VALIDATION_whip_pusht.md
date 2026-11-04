# JS vs Python: whip and Push-T

Same questions as `../VALIDATION.md` (not modified): does the browser plugin (`js/tasks/whip.js`, `js/tasks/push_t.js`, `js/tasks/push_t_plan.js`) reproduce the Python arena? Calibration seeds 1000-1019 only (no TEST seeds). Raw numbers: `js_vs_python_whip.json`, `js_vs_python_push_t.json`. Python references: `python_whip.json`, `python_push_t.json`.

Physics: Python `mujoco` 3.17.0 vs the vendored WASM (npm `mujoco` 3.1.16; its `mj_versionString()` reports 3.5.1). Plants are the Python-exported MJCF (`assets/whip.xml`, `push_t.xml` are templates; the chain / T block / pusher XML is rebuilt in JS from the same formulas so the weakness sliders can change it).

## Whip (`node validate_whip.mjs`; Python side `tools/export_whip.py`)

**What is and is not ported.** The execution half of the teacher (clamped cubic spline through 4 joint keyframes + computed torque on the arm-only nominal model, LQR gains) is ported. The planner (CMA-ES search on the nominal arm+chain model with batched `mujoco.rollout`) is NOT: **plans are precomputed offline by the Python teacher** (seed 0, from the nominal start posture) for 12 targets (d = 1.20/1.35/1.50/1.65 m x angle -0.35/0/+0.35 rad) x 3 assumed chain lengths (90/100/110 % of 1.0 m) = 36 plans in `assets/whip.json`, and looked up in the browser. The UI says so. Targets in the browser are limited to those 12 cells.

| check | result |
|---|---|
| trajectory (13 DoF: 3 arm + 10 chain), same q0, target, keyframes; 2 hits and 2 misses of the plan table, torque too | hits (154 and 289 steps until the hit): max abs q error 1.7e-11 and 2.4e-11 rad, torque 5e-10 Nm. Misses (full 500 steps): identical to 4.4e-11 rad until step ~320, then the chain's chaotic late motion diverges: max 4.2e-3 rad (torque 3.0 Nm) by step 500 |
| plan-table grid: 12 targets x 3 assumed x 3 true chain length (108 episodes, start jitter from seeds 1000-1011) | same outcome in 106/108; success Python 72, JS 72 in total. Per condition (assumed/true): 0.9/0.9 11 vs 11, 0.9/1.0 6 vs 6, 0.9/1.1 3 vs 3, 1.0/0.9 8 vs 8, **1.0/1.0 12 vs 11**, 1.0/1.1 7 vs 8, 1.1/0.9 6 vs 6, 1.1/1.0 8 vs 8, 1.1/1.1 11 vs 11 (Python vs JS). The two disagreements are near-threshold cases whose closest approach differs by up to 5.9 cm (the median difference is 1e-11 m) |
| 20 seeds 1000-1019, the Python teacher's own per-seed plan executed in JS, nominal | Python 20/20, JS 20/20, outcome identical 20/20 |
| same, weakness: true chain length -10 % | Python 9/20, JS 10/20, outcome identical 19/20 (seed 1006 differs) |

Reading: the failure mode the arena is meant to show is reproduced, in the plan table as well: a plan made for a 1.0 m chain hits 12/12 on a 1.0 m chain and 8/12 (0.9 m) or 7/12 (1.1 m) on a +-10 % chain; the same mismatch effect appears when the planner assumes 0.9 or 1.1 m (6-8/12 on the nominal chain). Differences between the builds appear only after chaotic chain motion (misses, near-threshold cases).

Not validated / limits: the table plans start from the mean posture while each episode jitters the posture by +-0.02 rad (as in Python), so the browser's nominal hit rate is 91/100 seeds (JS only) instead of the 20/20 of per-seed planning. The browser's random start jitter and target choice use a JS RNG (not reproducible from Python seeds); the "random swing" baseline likewise. The command-delay and tip-payload sliders are not part of the Python whip experiments and were not compared. Mouse perturbations not validated.

## Push-T (`node validate_push_t.mjs`; Python side `tools/export_push_t.py`)

**Everything is ported**, including the planner (beam search over 168 push primitives with the quasi-static ellipsoidal-limit-surface model, visibility-graph approach paths, hold/re-plan logic, operational-space computed torque). Planning is fast enough for the browser (6 plans + 2230 simulated steps of seed 1000 run in 0.29 s in node), so nothing is precomputed. Tie-breaking inside the beam search differs from numpy only for exactly equal costs.

| check | result |
|---|---|
| geometry constants (rho^2, area) vs Python | identical (difference 0) |
| trajectory seed 1000, nominal, 2230 steps, same Python initial state | q and torque agree to <= 3e-11 rad / 5e-9 Nm until the pusher first touches the block (step 158); from first contact the two MuJoCo builds differ (2e-7 rad at step 159, growing to 1.1e-3 rad max over the episode). **The sequence of 6 planned pushes is identical** (step and primitive index) |
| outcomes, 20 seeds, nominal | Python 20/20, JS 20/20; steps to success identical in 3/20 seeds, differ by up to 667 steps in the others |
| outcomes, sensor delay 30 ms (6 steps on q, qd and block pose) | Python 0/20 (14 block lost, 6 timeouts), JS 0/20, outcome identical 20/20 (the failure class and step counts differ per seed) |
| friction +50 % / block mass +100 % / noise 5 mm + 0.0125 rad | Python 20/20 each, JS 20/20 each (JS noise uses its own RNG: statistics only) |
| block centre of mass offset 20 mm in x | Python 20/20, JS 19/20 |

Reading: success rates agree (20/20 nominal, 0/20 with a 30 ms sensor delay, no effect of friction / mass / noise, COM within one seed); individual episodes differ in timing because contact dynamics differ between the two MuJoCo builds and the planner re-plans on a slightly different block pose after each push. Exact trajectories are not reproduced after the first contact.

Not validated / limits: random initial states and pose noise (JS RNG); mouse perturbations; "restart from the dropped T" is a browser-only feature (the pose is clamped into the sampling area, the pusher start is re-sampled from the seed). The pusher is drawn by a massless extra sphere (`pusher_vis`) because the viewer does not draw cylinders; the physics geom is the Python cylinder. After an episode ends the arm is brought to rest by damping (Python stops the episode).

## Re-running

```
.venv/bin/python web/arena/validation/tools/export_whip.py       # ~2 min, writes assets/whip*.{xml,json}, validation/python_whip.json
.venv/bin/python web/arena/validation/tools/export_push_t.py     # ~2 min, writes assets/push_t*.{xml,json}, validation/python_push_t.json
cd web/arena && node validate_whip.mjs && node validate_push_t.mjs
```
(Python heavy runs: `flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 ...`.)

Interface note: `js/app.js` got one change, unsigned charts draw their guide line once (`for (const sgn of spec.signed ? [1, -1] : [1])`); before, a guide on an unsigned chart was also drawn mirrored below the axis.
