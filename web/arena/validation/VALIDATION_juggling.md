# Racket juggling: JS vs the Python spike03b env

Question: does the browser port (`js/tasks/juggling.js`, MuJoCo WASM 3.1.16) reproduce the Python mirror-law teacher on the Python plant (`spike03b/env.py`, `mujoco` Python wheel)?
Calibration seeds only (1000-1019); no TEST seeds. Re-run: `python validation/tools/export_juggling.py` (heavy: use the flock/systemd-run wrapper), then `node validate_juggling.mjs` (raw numbers: `validation/js_vs_python_juggling.json`).

## Setup

| | Python | JavaScript |
|---|---|---|
| plant | `E.build(zeta_for(e), mscale)`: 3 hinges, flat racket box, ball on two slide joints, implicitfast, dt_sim 1 ms, 5 substeps (200 Hz) | the same MJCF, exported as a template (`assets/juggling.xml`, `@M1@..@ZETA@` filled in JS) |
| controller | `JugEnv.measure_update` (alpha-beta filter, delay buffer, noise) + `teacher` (mirror law, nominal CTC by `mj_inverse`) | line-by-line port (`_measure`, `_teacher`), `ArmModel.inverse` for CTC |
| restitution | `zeta_for(e)` (np.interp of `zeta_map.json`) | same map, same interpolation |
| episode | ends at 100 hits (cap), ball lost (dropped / dead / lateral), or 100 s | same rules |
| inputs | initial state from `init_cond(seed)`, sensor noise sigma 0.2 mm from `default_rng(seed + 7e6)` | the Python initial state and (case a) the exact Python noise samples, (case b) its own Gaussian noise |

## 1. Fixed-step trajectory (seed 1000, nominal, first 1500 steps = 7.5 s, exact noise)

Max abs error over arm q, arm qd, ball x, ball z, ball vx, ball vz (JS minus Python), measured on the state before each control step.

| step | 100 | 200 | 400 | 600 | 800 | 1000 | 1200 | 1499 |
|---|---|---|---|---|---|---|---|---|
| max state error | 3e-16 | 8e-14 | 4e-12 | 1e-8 | 3e-5 | 5e-4 | 0.11 | 0.19 |

- The first racket hit is at step 56. Up to it (free flight plus CTC tracking) the state differs by at most 1.1e-15 and the torque by 5e-14 Nm: ball estimator, mirror law, kinematics and computed torque match.
- After the hits the error grows by roughly 10x per 100-200 steps (above 1e-9 at step 424, 1e-6 at 663, 1e-3 at 907). This is contact dynamics: soft contact plus a bouncing ball amplify any round-off difference (MuJoCo 3.17 vs 3.1.16, compiler, libm) exponentially. The runs are not bit-identical after the first few hits, and should not be expected to be.
- By step 1200 the trajectories are visibly different (0.11 m); the JS run had 12 hits in these 1500 steps.

## 2. Consecutive hits, 20 CAL seeds (untuned teacher: model restitution 0.8)

Python: `E.run_policy(JugEnv(20, Cond(e)), None, ...)`; hits capped at 100.

| true restitution | Python median / mean (min-max) | JS, exact Python noise | JS, own noise | exact-noise seeds with identical count | mean / max abs diff per seed |
|---|---|---|---|---|---|
| 0.8 (nominal) | 100 / 100 (100-100) | 100 / 100 (100-100) | 100 / 100 (100-100) | 20 / 20 | 0 / 0 |
| 0.4 | 27 / 26.85 (20-33) | 26.5 / 26.85 (22-32) | 25.5 / 25.65 (21-32) | 3 / 20 | 2.0 / 5 |

- Nominal: every seed reaches the 100-hit cap in both implementations (20/20 agree).
- Restitution 0.4: the untuned teacher loses the ball after 20-33 hits in Python (median 27, consistent with the roughly 26 quoted for it) and after 22-32 in JS with identical inputs. Per-seed counts differ by 2.0 hits on average (max 5) because the divergence of section 1 decides which hit the ball is lost on; the distributions match (mean 26.85 vs 26.85, ranges overlap). With JS's own noise the mean is 25.65, inside the seed-to-seed spread.
- All failures are "ball lost"; none is a numerical blow-up.

## Not covered

Sensing delay, link-mass scaling and the slider-tuned teacher parameters use the same code path as the Python `Cond` fields but were not separately compared against Python. The mouse push exists only in the browser. Browser sensor noise uses mulberry32 + Box-Muller, not NumPy's generator, so a browser episode with seed s is not the Python episode with seed s (the validation above feeds the Python noise samples to remove this difference).
