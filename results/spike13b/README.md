# spike13b: headroom decomposition + fragility sweep (whip, realistic conditions)

## 0. Pre-registration (written BEFORE any holdout / new-seed result was looked at; not edited afterwards)

Setting: whip task, conditions abc at x1 (sensor noise, per-link mass + Coulomb friction, actuator noise), plant integration step /4 (SUBF=4), exactly as spike10b. Plant of each seed is the seed-derived one (rep=None). CPU only.
Seeds: main = holdout 1100-1599 (n=500). Replication = new seeds 700600-700799 (n=200). Fragility sweep = 1100-1299 (n=200). CAL 1000-1049 is used only for smoke tests (no numbers reported from it). TEST (0-49) and TEST-B (20000-20499) are never touched (`spike13b/c13.py` guard asserts); `results/.test_access.log` must stay at 109 lines.

### 0.1 The ladder (each rung changes exactly one thing relative to the rung above; all rungs run on the same seeds, so differences are paired)

| rung | method | what changes vs the rung above |
|---|---|---|
| R0 | T-adapt3-full (spike10b best-classical, frozen config) | baseline: probe 0.6 s, fit 13-parameter plant, replay to reconstruct chain state, plan, execute |
| R1 | same pipeline, estimator returns the TRUE plant theta | removes parameter-estimation error (probe, replay-state reconstruction and planner unchanged) |
| R2 | R1 + the swing planner starts from the TRUE full state (arm + chain qpos/qvel read from the simulator at the end of the probe) instead of the replay-reconstructed one | removes chain-state reconstruction error (replay on a noise-free model vs the real noisy plant) |
| R3 | T-oracle that waits: true theta, true REST state, no probe motion; arm holds q0 for 0.6 s, then plans/executes a swing of duration 1.9 s (identical clock and horizon as R2) | removes the probe excitation itself (the probe leaves arm and chain in a moving state at swing start; R3 starts from rest). Same time budget. |
| R4 | T-oracle (spike10b OracleFull): true theta, true rest state, swing planned from t=0 with the full 2.5 s horizon | gives back the 0.6 s probe time (time-budget cost) |
| R5 | canonical T-oracle (spike11a teacher: continuation warm-start CMA-ES, `canon=True`) | planner-search variance: same information as R4, only the planner's search is made deterministic/continuous. Not an information gap. |

Why R3 keeps the clock (waits 0.6 s) rather than starting at t=0: it isolates "start state" from "time"; the time effect is R3 -> R4. This is the fair split I chose before seeing results; the sum R2 -> R4 does not depend on this choice.
L-est-full (spike10b models, 3 training seeds, `lest_full#0..2`, run on every seed; per-seed value = mean over the 3 models, so one number per seed) is placed on the same table. It is a pipeline "R0 with a learned estimator", so it is compared against R0 (what the learned estimator gains/loses over the fitted one) and against R1 (how much of the parameter-error block it recovers).

Checks of the ladder ends (reported, not tuned): R0 should be about 66-70 % on holdout (spike10b: 69.7 % on 1100-1399, 65.6 % on TEST-B); R4/R5 about 88-96 %. If R0 is outside 62-72 % or R4 outside 85-97 % I report it as a mismatch and do not proceed to interpretation.

### 0.2 Block definitions and judgments

* Block k = paired mean difference of adjacent rungs (R1-R0 = parameter error, R2-R1 = chain-state error, R3-R2 = probe-excitation/start state, R4-R3 = probe time, R5-R4 = planner search). 95 % CI by paired bootstrap over seeds (20000 resamples). A block is called "real" if its CI excludes 0; otherwise "not distinguishable from 0" (never "absent").
* Total headroom = R5 - R0 and R4 - R0 (with CI). Blocks are not required to add to the total exactly (they do, telescoping, by construction; the report shows both).
* "L-est eats block X": L-est-full minus R0 has a CI excluding 0 on the positive side AND L-est-full is within the CI of R1 (i.e. recovers the parameter block); otherwise "does not".
* The replication on new seeds 700600-700799 is a consistency check: a block counts as "replicated" if the sign of the point estimate agrees and the new-seed CI overlaps the holdout CI.
* Fragility: factors swept separately, sensor noise (obs.q, obs.qd of the arm, x1 = 1e-3 rad / 1e-2 rad/s) and actuator noise (x1 = U(0.01, 0.02) Nm), multipliers {0.5, 1, 2, 4, 8}, the other factors stay at their realistic x1 value. Methods: T-nom, T-robust, T-adapt3-full (frozen, NOT re-tuned), L-est-full (frozen, NOT retrained; 3 training seeds assigned cyclically over the seeds, so one pooled n=200 number), T-oracle (spike10b OracleFull, plain; canonical oracle is not swept). Seeds 1100-1299.
* "Drop-10 multiplier" of a method = smallest multiplier at which success falls to <= (success at x1 - 10 pp), linear interpolation in log2(multiplier) between grid points; ">8" if never; "<=0.5 (already below)" would be reported if x0.5 is already 10 pp above x1 is not counted (we only look at multipliers >= 1).
* "A collapses before B" only if their drop-10 multipliers differ by at least one grid step AND at the first multiplier where A has dropped >= 10 pp, the paired drop difference (A's drop - B's drop) has a CI excluding 0. Otherwise "no distinguishable order".
* Noting in advance: T-adapt3-full's sq, sv (measurement-noise scales) were tuned at x1; L-est was trained at x1. Neither is adapted at higher noise; the sweep measures robustness of the frozen methods, not best achievable at each noise level.
* Optional privileged-observation probe: only if time allows; labelled exploratory.

## 1. Results

## 0.3 Addendum (written after the holdout ladder R0-R5 was seen, before the sub-decomposition below was run; therefore post hoc / exploratory)
The ladder showed that R0 -> R1 (parameter error) is the largest block. To say WHICH parameters matter, the R0 -> R1 block is split on the same holdout seeds 1100-1599 by group (ls = length scale, m = 10 per-link masses, ds = viscous damping, fr = Coulomb friction), using the cached T-adapt3-full fit of each episode:
* `H_in_<g>`: fitted theta with group g replaced by the TRUE value (how much does fixing only g buy?).
* `H_out_<g>`: TRUE theta with group g replaced by the FIT (how much does only g being wrong cost?).
Effects need not be additive. Judgment rule: a group "matters" if its H_in minus R0 or R1 minus H_out has a CI excluding 0.
Also exploratory (not part of any judgment): ridge probe `spike13b/probe_state.py` (does the arm probe history linearly contain the chain state / theta?).
All numbers: `table.md` (generated by `spike13b/report13.py`), raw per-seed results in `ladder_hold.json`, `ladder_new.json`, `sweep.json`, `hybrid.json`, `probe_state.json`. Statements marked (inference) are interpretation.

### 1.1 Ladder (holdout 1100-1599, n=500; new seeds 700600-700799, n=200; step = paired difference to the rung above, pp [95% CI])
| rung | holdout % | step | new seeds % | step |
|---|---|---|---|---|
| R0 T-adapt3-full | 69.2 [65.2, 73.2] | - | 69.0 [62.5, 75.5] | - |
| R1 true theta | 86.0 [83.0, 89.0] | +16.8 [+12.0, +21.6] | 82.0 [76.5, 87.0] | +13.0 [+4.5, +21.5] |
| R2 + true chain state | 89.0 [86.2, 91.6] | +3.0 [-1.0, +7.0] | 89.5 [85.0, 93.5] | +7.5 [+1.0, +14.0] |
| R3 oracle at rest, waits 0.6 s | 88.2 [85.4, 91.0] | -0.8 [-4.6, +3.2] | 85.5 [80.5, 90.5] | -4.0 [-10.5, +2.5] |
| R4 T-oracle (t=0, 2.5 s) | 90.8 [88.2, 93.2] | +2.6 [-1.2, +6.2] | 87.5 [82.5, 92.0] | +2.0 [-4.5, +8.5] |
| R5 canonical T-oracle | 92.0 [89.6, 94.2] | +1.2 [-2.2, +4.6] | 92.5 [88.5, 96.0] | +5.0 [-0.5, +10.5] |
| L-est-full (mean of 3 models) | 64.7 [62.1, 67.3] (vs R0 -4.5 [-9.5, +0.4]) | | 66.7 [62.5, 70.8] (vs R0 -2.3 [-9.7, +5.2]) | |

Ends check: R0 69.2 / 69.0 (spike10b 69.7 on 1100-1399) and R4 90.8 / R5 92.0 (spike10b T-oracle 90.0 holdout, spike11a canonical 92.5 on fresh seeds): both ends match the prior numbers.

Findings (measured):
* The parameter-estimation block R0 -> R1 is the largest: +16.8 pp [+12.0, +21.6] on holdout, +13.0 [+4.5, +21.5] on new seeds (replicated, same sign, overlapping CIs). R0 -> R2 (everything estimation-side) is +19.8 [+15.2, +24.4] of the total R0 -> R4 +21.6 [+17.0, +26.2] / R0 -> R5 +22.8 [+18.2, +27.4].
* Chain-state reconstruction (R1 -> R2): +3.0 [-1.0, +7.0] on holdout (not distinguishable from 0), +7.5 [+1.0, +14.0] on new seeds. Pooled sign is positive and small (single-digit pp); not replicated as "real" on the main set.
* Probe excitation (R2 -> R3): -0.8 / -4.0, CIs contain 0. Probe time (R3 -> R4): +2.6 / +2.0, CIs contain 0. Planner canonicalisation (R4 -> R5): +1.2 / +5.0, CIs contain 0 (new-seed upper end 10.5). Together R2 -> R4 (probe cost) is +1.8 [-1.6, +5.4] on holdout, -2.0 [-8.0, +4.0] on new seeds: no probe cost distinguishable from 0, with a CI of about +-5 pp.
* L-est-full: 4.5 pp BELOW R0 on holdout (CI touches 0), 21.3 pp below R1 [-25.3, -17.3]. It recovers none of the parameter block. Per-model results 62.8-65.6 % (holdout).
* Post-hoc split of the parameter block (holdout, n=500): fixing only one group of the fit does not help (H_in - R0: length -6.0 [-11.8, -0.2], masses -3.8, damping -4.2, friction +0.6); but making only one group wrong while all others are true costs: damping 18.4 [13.6, 23.4], per-link masses 16.2 [11.6, 20.8], length 7.4 [2.8, 12.0], friction 4.6 [0.4, 8.8]. **(inference)** the errors of the fit are jointly harmful and not additive: the planner needs several parameters (damping, per-link masses, length) right at once; fixing one of several wrong ones does not help (compensation between parameters in the fit).
* Exploratory ridge probe (3000 train episodes, 200 holdout seeds, linear): the probe features explain 82 % of the variance of theta but only 12 % of the chain state at swing start (27 % of positions, 12 % of velocities); even with the true theta added 15 %. Linear readout only: it says nothing about nonlinear readouts. The privileged-observation residual RL (optional item) was NOT run.

### 1.2 Fragility (seeds 1100-1299, n=200, frozen methods, other factors at x1; full tables with CIs in `table.md`)
Drop-10 multiplier (point estimates; success x1 -> worst):
| method | sensor noise | actuator noise |
|---|---|---|
| T-nom (57.0 at x1) | >8 (flat) | >8 (flat) |
| T-robust (53.0) | >8 (flat) | >8 (flat) |
| T-adapt3-full (69.5) | 2.0 (x2: -10.0 [-19.5, -1.0]; x4 -9.0; x8 -5.0, non-monotone) | 3.5 (x4: -12.5 [-21.5, -3.5]; x8 -14.5 [-24.0, -5.0]) |
| L-est-full (59.0) | >8 (x8 -5.5 [-15.0, +3.5]) | >8 (x8 -0.5 [-10.0, +9.0]) |
| T-oracle (89.5) | >8 | >8 |
* Actuator noise: T-adapt3-full collapses before L-est-full under the pre-registered rule: drop-10 multipliers 3.5 vs >8, and at x4 the paired drop difference is -16.5 pp [-28.5, -5.0]. Sensor noise: multipliers differ (2.0 vs >8), but at x2 the paired drop difference is -10.5 [-23.0, +2.0], CI contains 0, and T-adapt3's curve rebounds at x8 -> by the rule "no distinguishable order" for sensor noise.
* L-est-full never reaches the level of T-adapt3-full at x1 in this sweep (59.0 vs 69.5 at x1), so "L-est is more robust" means "falls from a lower level, not by less than a few pp"; at x4/x8 actuator noise L-est (63.0 / 58.5) is above T-adapt3 (57.0 / 55.0), at x4 sensor equal. L-est x0.5 sensor (72.5) vs x1 (59.0) shows its x1 baseline in this pooled-cyclic design is noisy (CI +-6.5 pp); drops relative to it are uncertain.
* T-nom and T-robust are insensitive (<= 3 pp) but at 53-57 %. T-oracle is flat (86-93.5 %), so noise per se does not remove headroom (it sits in the estimator pipeline).

## 2. Deviations, limits, and what is not claimed
* Added post hoc: the group split of the parameter block (0.3), the ridge probe, CAL reproduction check. Optional residual-RL-with-privileged-observation probe not run (time spent elsewhere, linear probe only).
* The ladder's R1 -> R2 step is read from the simulator state (qpos/qvel after 0.6 s); the planner then starts from the true noise-free state but the arm reading used for q0 is the noisy obs, as in the other rungs.
* R3 uses a waiting design (same clock), R4 returns the time; the R2 -> R4 total does not depend on that split.
* Consistency check against spike10b's CAL ceiling diagnostic (CAL n=50: true-theta pipeline 74 %, T-adapt3-full 78 %, T-oracle 90 %): reproduced exactly (R1 74.0, R0 78.0, oracle 90.0 on CAL 1000-1049). On holdout n=500, R1 is 86.0 %. The spike10b inference "the gap is largely probe time and unknown chain state, not parameter error" is NOT supported at n=500 / new seeds. Possible reasons (inference): n=50 CAL CIs are +-12 pp, and T-adapt3-full was tuned on CAL (winner's curse: 78 on CAL vs 69 on holdout).
* T-adapt3-full and L-est-full were not retuned/retrained for the noise sweep. Sweep n=200 means CIs of +-6.5 pp per cell and +-9 pp for paired drops; only large drops are resolved. The T-adapt3 sensor curve is non-monotone (x2 down, x8 back up), which I cannot explain (not diagnosed).
* L-est in the sweep: one training seed per episode (cyclic), so its cell values mix three models; ladder uses the mean of the three.
* The sweep T-oracle is the plain spike10b oracle, not the canonical one.
* One plant family, simulation only.
* `results/.test_access.log` is still 109 lines; no seed < 1000 or in 20000-20499 was used (guard in `spike13b/c13.py`).
