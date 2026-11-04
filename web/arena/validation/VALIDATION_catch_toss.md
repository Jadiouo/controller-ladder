# Catch and Toss: JS vs the Python arena

Question: do the browser versions of `catch` and `toss` (`js/tasks/catch.js`, `js/tasks/toss.js`, MuJoCo WASM) reproduce the Python arena (`arena/tasks/{catch,toss}.py`, teachers `catch_teacher`, `toss_teacher`)?

## Setup

| | Python reference | JavaScript |
|---|---|---|
| physics | `mujoco` 3.17.0, `ArenaEnv`, RK4, dt_sim 2.5 ms, 2 substeps (200 Hz) | `mujoco@3.1.16` WASM, same MJCF |
| plant | `build_mjcf(three_link(tool=net/claw), task.tip_xml)` | exported to `assets/catch.xml`, `assets/toss.xml` (basket moved per seed with `model.geom_pos`, as in `Toss.reset`) |
| nominal models of the controllers | `ArmModel(spec)` (toss: `payload_mass + ball_mass` while holding) | `assets/catch_nominal.xml`, `assets/toss_nominal_{hold,free}.xml`, `mj_inverse` |
| controllers | `catch_teacher`, `toss_teacher` (default parameters) | line-by-line ports: least-squares drag-ballistic fit, intercept search + quintic/min-jerk path + level IK + computed torque (catch); ballistic inversion (scipy `brentq` ported from the C source) + joint quintic + computed torque (toss) |
| seeds | 1000-1019 (calibration seeds only; no test seeds used) | the exact Python initial states (`validation/tools/export_catch_toss.py` -> `python_{catch,toss}.json`) |
| cases | `clean`; catch: `BallObsDelay(2)` (10 ms), `ObjectMass(+0.3)`; toss: `ReleaseDelay(40)` (40 ms late release), `ObjectMass(+0.3)` | `setWeakness("delay"|"release"|"mass", ...)` (plant/sensor only) |

The browser itself draws launches / baskets from its own RNG with the same distributions as Python (it cannot reproduce numpy's generator); validation therefore starts both sides from the Python initial state.

## Trajectory (seed 1000, whole episode; `node validate_catch.mjs`, `node validate_toss.mjs`)

| task / case | steps | max abs arm q error (rad) | max abs ball error (m) | max abs torque error (Nm) |
|---|---|---|---|---|
| catch clean | 327 (caught) | 2.9e-11 | 7.7e-11 | 4.8e-9 |
| catch 10 ms ball-observation delay | 249 (missed) | 1.1e-11 | 1.5e-11 | 5.0e-9 |
| toss clean | 246 (scored) | 3.6e-5 | 7.8e-4 | 2.5e-3 |
| toss 40 ms late release | 270 (missed) | 4.1e-5 | 5.2e-4 | 2.5e-3 |

Catch matches to 1e-10, including the contacts of the ball with the basket and the episode end step.

Toss does NOT match bit for bit. The plans are identical (seed 1000: release speed 2.6215593004832938 m/s in Python vs 2.6215593004840803 in JS, difference 8e-13, same release point, angle, posture; the brentq port is faithful). The difference appears from step 0 as a smooth growth (3e-7 rad at step 12, 3e-5 rad at step 160) while the ball is still held by the soft `connect` equality constraint (solref 0.005), and the ball then follows the arm release state, so the 3.6e-5 rad arm error becomes 0.8 mm of ball position error at the end. The most likely cause is the constraint solver of MuJoCo 3.1.16 (WASM) vs 3.17.0 (Python): the tracking, pole and catch tasks have no equality constraint and agree to 1e-10 to 1e-12. I did not find another mismatch and did not investigate the MuJoCo changes further.

## Outcomes, 20 calibration seeds (1000-1019)

| task / case | Python successes | JS successes | seeds with the same outcome | max abs episode-length difference (steps) |
|---|---|---|---|---|
| catch clean | 14 / 20 | 14 / 20 | 20 / 20 | 0 |
| catch 10 ms ball-observation delay | 5 / 20 | 5 / 20 | 20 / 20 | 0 |
| catch ball mass +30 % | 16 / 20 | 16 / 20 | 20 / 20 | 0 |
| toss clean | 20 / 20 | 20 / 20 | 20 / 20 | 0 |
| toss 40 ms late release | 16 / 20 | 16 / 20 | 20 / 20 | 1 (seeds 1008, 1018) |
| toss ball mass +30 % | 18 / 20 | 18 / 20 | 20 / 20 | 0 |

Raw numbers per seed: `js_vs_python_catch.json`, `js_vs_python_toss.json`. The catch clean rate of 14 / 20 on these seeds is the Python value as well (checked with `arena.eval.evaluate(Scenario("catch"), "catch_teacher", range(1000, 1020))`); the 10 ms delay drops the catch rate from 14 to 5 in both implementations.

## Not validated

- The browser's random launches and baskets (own RNG), the mouse shove on the ball or links, the options `throw` / `basket`, the "release-lag compensation" slider of toss (default 0 reproduces Python; non-zero is a browser-only extension), drag error and the other slider values of the controllers (only defaults are compared).
- Air-drag weakness and the restitution weakness: the plant edit is the same code path as the mass edit (`withBall`), but no Python comparison was run for it.
- Python and JS use different MuJoCo versions (3.17.0 vs 3.1.16).

## Re-running

```
# repo root: ~20 s, writes web/arena/assets/* and web/arena/validation/python_{catch,toss}.json
.venv/bin/python web/arena/validation/tools/export_catch_toss.py
cd web/arena
node validate_catch.mjs && node validate_toss.mjs
```
