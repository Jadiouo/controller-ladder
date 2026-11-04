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
- Only the tracking task and the 2-link arm are covered. Pole balancing, and any learned controller, are not in the browser arena.

## Re-running

```
cd web
npm install                      # only needed to refresh vendor/ copies (mujoco@3.1.16, three@0.160.0)
node arena/validate.mjs
.venv/bin/python web/arena/validation/tools/python_trajectories.py   # regenerate the Python side (repo root)
```
