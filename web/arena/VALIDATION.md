# JS controllers vs the Python arena

Question: does the browser arena (JavaScript controllers + MuJoCo WASM) reproduce the Python arena's closed-loop joint trajectories?

## Setup

| | Python reference | JavaScript |
|---|---|---|
| physics | `mujoco` 3.17.0 (Python wheel), `arena.env.ArenaEnv` | `mujoco@3.1.16` WASM (npm, vendored in `vendor/`), `js/sim.js` |
| plant | `build_mjcf(RobotSpec.two_link())`, RK4, dt_sim 2.5 ms, 2 substeps (200 Hz control) | the same MJCF string, exported as `assets/arm2.xml` |
| task | `Scenario("tracking", {"ref": "glyph", "glyph": "S"})`, seed 1000 (calibration seed) | reference and initial state exported from Python (`assets/tracking_refs.json`, rounded to 1e-9) |
| controllers | `arena.controllers.pid.PID` (defaults: w_n = 20 rad/s, Ki = 0, gravity comp.), `ctc_lqr.CtcLqr` (Qp = 1.6e5, Qv = 800 -> Kp = 400, Kd = 40) | `PID`, `CtcLqr` in `js/sim.js`; nominal inverse dynamics from `mj_inverse` on a nominal model, as in Python |
| duration | first 2 s = 400 control steps | 400 control steps |
| cases | `clean`; `hard` = `PayloadError(rel=0.3)` + `Delay(2)` (command delay) | payload 1.5 x 1.3 kg in the plant only, delay 2 steps |

No sensor noise and no torque pulses (they are random in Python and not part of the JS arena). The mouse perturbation of the arena is not part of this comparison.

## Result (`node validate.mjs`, raw numbers in `validation/js_vs_python.json`)

Errors are JS minus Python over all 401 stored states (q) and 400 applied torques (tau, after delay and clipping).

| case | controller | max abs q error (rad) | RMS q error (rad) | max abs tau error (Nm) | max abs q in the run (rad) |
|---|---|---|---|---|---|
| clean | PID | 1.8e-10 | 4.7e-11 | 2.6e-7 | 2.17 |
| clean | CTC+LQR | 1.8e-10 | 4.5e-11 | 2.4e-7 | 2.17 |
| hard | PID | 1.9e-10 | 5.3e-11 | 3.0e-7 | 2.18 |
| hard | CTC+LQR | 1.8e-10 | 4.9e-11 | 2.5e-7 | 2.17 |

Largest torque in the 2 s window: 28.5 Nm (PID), 51.1 Nm (CTC+LQR) of the 80 Nm limit.

## Reading

- The worst joint-angle difference over 2 s is 1.9e-10 rad (relative 1e-10 of the joint range), and the worst torque difference is 3.0e-7 Nm. This is consistent with the 1e-9 rounding of the exported reference and initial state; no systematic drift is visible, so the Riccati closed form, the inverse-dynamics path, the gravity term, the integrator clip, and the delay buffer match the Python implementation.
- The two MuJoCo builds differ in version (3.17.0 vs 3.1.16) and in compiler/libm, which could introduce differences at the 1e-12 level in this smooth, collision-free problem. They are not distinguishable from the reference rounding here.
- The table above covers tracking only; pole balancing is in its own section below. Learned controllers are not in the browser arena.

### Regression after the task-plugin refactor

`validate.mjs` now drives tracking through the plugin interface (`js/tasks/tracking.js`: `create`, `setOption`, `setWeakness`, `setController`, `reset`, `step`). Re-run: max abs q error 1.79e-10 / 1.79e-10 (clean PID / CTC+LQR) and 1.93e-10 / 1.83e-10 (hard PID / CTC+LQR) rad; max abs torque error 2.4e-7 to 3.0e-7 Nm. Unchanged at the 1e-10 level.

## Pole balance (`node validate_pole.mjs`, raw numbers in `validation/js_vs_python_pole.json`)

Setup: Python `Scenario("pole_balance")` (2-link arm, 0.5 kg tip payload, hinge pole L = 0.35 m, bob 0.05 kg, 1000 control steps = 5 s), controllers `pid` (w_n 30, c_i 70, c_p 5, c_d 0.2) and `ctc_lqr` (Q = diag(20, 2, 200, 20), R = 0.01, kz = 400/40). JS starts from the exact Python initial state of each seed (exported by `validation/tools/export_pole.py`; the browser itself draws random initial states from the same distribution with its own RNG). The JS LQR gain is computed in JS (matrix-sign-function Riccati solver), not copied from scipy.

| check | result |
|---|---|
| LQR gain, JS vs scipy | max abs diff 7.6e-12, max relative 5.0e-14 |
| trajectory, seed 1000, clean, 1000 steps, PID | max abs q error 1.1e-12 rad (3 joints incl. hinge), torque 8.7e-11 Nm |
| trajectory, seed 1000, clean, 1000 steps, CTC+LQR | max abs q error 1.3e-12 rad, torque 5.4e-11 Nm |
| outcomes, 20 seeds (1000-1019), clean, PID | Python 19 balanced, JS 19; same outcome on 20/20 seeds, identical step counts |
| outcomes, clean, CTC+LQR | Python 20, JS 20; 20/20 agree |
| outcomes, hard (payload +30 %, delay 2), PID | Python 13, JS 13; 20/20 agree, identical fall steps |
| outcomes, hard, CTC+LQR | Python 20, JS 20; 20/20 agree |

Not validated: mouse perturbations and the browser's random initial states (not reproducible from Python seeds). The two MuJoCo builds differ (3.17.0 vs WASM 3.1.16); no deviation above 1e-6 appears in any run.

## Re-running

```
cd web
npm install                      # only needed to refresh vendor/ copies (mujoco@3.1.16, three@0.160.0)
node arena/validate.mjs
node arena/validate_pole.mjs
.venv/bin/python web/arena/validation/tools/export_pole.py   # pole assets + Python reference (repo root)
.venv/bin/python web/arena/validation/tools/python_trajectories.py   # regenerate the Python side (repo root)
```

## Other tasks

Each added task has its own JS-vs-Python check (CAL seeds 1000-1019; same initial states exported from Python):

- Catch and toss: [validation/VALIDATION_catch_toss.md](validation/VALIDATION_catch_toss.md)
- Juggling: [validation/VALIDATION_juggling.md](validation/VALIDATION_juggling.md)
- Whip and Push-T: [validation/VALIDATION_whip_pusht.md](validation/VALIDATION_whip_pusht.md)

Contact-rich tasks (juggling, toss, Push-T, missed whip swings) do not match bit for bit after the first contact or release. The browser build reports MuJoCo 3.5.1, while the Python runs used MuJoCo 3.17, and small differences grow through contact. Success counts per seed are compared instead, and the files above list every mismatch. Whip plans are precomputed offline by the Python teacher on a grid of 12 targets times 3 assumed chain lengths. The browser only executes them, on a chain whose true parameters can be changed.
