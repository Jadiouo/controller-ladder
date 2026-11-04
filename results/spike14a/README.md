# spike14a: estimator duel (whip, realistic conditions) -- learned vs strengthened classical parameter estimation

## 0. Pre-registration (written BEFORE any CAL / holdout / new-seed result of any spike14a method was looked at; NOT edited afterwards -- everything below section 0 is appended later)

Interface (fixed): the SAME probe data (0.6 s probe, amp 0.6, identical for all estimators because the probe is open-loop and seed-deterministic; verified by exact key match of the stored probe data in the pipeline) -> estimator -> 13 parameters theta_hat = [length scale, 10 per-link mass scales, damping scale, Coulomb friction] -> the replay + planner + execution pipeline of spike13b R0/R1 (`Probe3`, unchanged). Probe design is NOT changed (optional probe-excitation optimisation is outside the main duel and would be reported separately).
Setting: whip, conditions abc x1, plant step /4, as spike10b / spike13b. Plant of each seed = seed-derived (rep=None).

Seeds: CAL = 1000-1049 + 2000-2149 (n=200; hyper-parameters, checkpoints, and the choice of the best method on each side). Holdout = 1100-1599 (n=500, primary). New seeds = 701000-701199 (n=200, confirmation). Training data of learned estimators = seeds 500000-511999 (x1, same seeds as spike10b L-est-full) and 500000-511999 + 540000-575999 (x4), rep 0. TEST (0-49) and TEST-B (20000-20499) are never touched (guard `guard14` in `spike14a/c14.py`); `results/.test_access.log` must stay at 109 lines.

Methods.
* Learned (each >= 3 training seeds; per-episode value of a variant = mean over its seeds): L-mlp1 = spike10b L-est-full (frozen, baseline); L-gru1 / L-gru4 = GRU on the raw probe time series (18 channels x 120 steps), data x1 / x4; L-dec1 (and L-dec4 if time) = same GRU, loss = (1-a) z-MSE + a * Gauss-Newton tip-trajectory error of the oracle swing (per-episode sensitivity J_i of the planner's swing model, computed on training episodes only); L-mlp4 = the L-est-full MLP recipe on x4 data (control for data vs architecture). Optional: seed-ensemble mean of the best variant.
* Classical (all use the known generative parameter distribution as prior, which is the classical counterpart of the information the learner gets from simulated labels): C0 = T-adapt3-full (frozen, baseline = R0); C-map* = C0 + moment-matched Gaussian MAP prior on all 13 parameters; with multi-start (random draws from the prior); C-ms* = K-step-ahead (multiple-shooting-style) residual: the arm state is re-initialised to the measured (q, qd) every K control steps; combinations.
* References on the same seeds: R0 = C0 pipeline, R1 = pipeline with the TRUE theta (spike13b definition).

Selection (CAL only): every candidate is screened by CAL label error (z-space, label std of the training set) and then run through the pipeline on CAL (n=200). Best learned = learned variant with the highest mean CAL pipeline success; best classical likewise; ties -> lower CAL label error. The choice is made before looking at holdout / new-seed pipeline numbers. Hyper-parameter sweeps are done on CAL only (label error, plus pipeline success for the finalists).

Primary judgement (written now, fixed): paired difference (best learned - best classical) in pipeline success on the HOLDOUT (n=500), 95% CI by paired bootstrap over episodes (20000 resamples).
* "Learned wins": difference >= +10 pp AND CI excludes 0; "confirmed" if on the new seeds the point difference has the same sign (positive). 
* "Classical wins": difference <= -10 pp AND CI excludes 0 (same confirmation rule, negative).
* Otherwise: "no side wins at the 10 pp bar" (the CI is reported so that power is visible; a difference that is real but <10 pp is stated as such, not as a win).
Also reported (not part of the judgement): recovered fraction of the R0 -> R1 gap by each method = (method - R0) / (R1 - R0) on holdout and new seeds (paired bootstrap CI of the numerator); parameter errors per group (length, per-link mass, damping, friction) with RMS on the holdout and new seeds; every candidate's holdout success (exploratory, no multiplicity correction); dose curve (learning data x1 vs x4).

Budget (three axes per method, section "Budget"): episodes (probe episodes / full-episode equivalents x 0.6/2.5 / CAL tuning episodes), wall-clock (cpu-seconds and wall), label source (simulator-privileged theta labels vs none). Learned-side data generation, sensitivity labelling (oracle planning of training episodes), training and tuning all count as learned budget; the classical side counts its CAL tuning fits. If the total budgets of the two selected methods differ by more than 2x it is flagged in the report.

Limits stated in advance: one plant family, simulation only; learned estimators train on simulator labels from the same distribution as the evaluation (in-distribution), classical uses the same distribution only as a prior; CAL n=200 selection is noisy (+-6.5 pp per cell) so the CAL-selected "best" can be a winner's-curse pick; deviations are listed in the section "Deviations" below.

---
## 1. Results (everything below section 0 was written after the runs; numbers are generated into `table.md`, `posthoc.md`, `summary.json`, `budget.json` by `spike14a/report14.py`, `posthoc14.py`, `budget14.py`)

### 1.1 Verdict (pre-registered rule, section 0)
Selection on CAL (made at 23:10 on 2 Nov, before any holdout / new-seed pipeline number of the selected learned method existed; `selection.json`): best learned = **L-dec4** (GRU, decision-weighted loss, a=0.9, data x4; CAL 73.2 % as mean of its 3 training seeds; the other learned variants: seed-0 model only, 65.5-70.5 %); best classical = **C0 = T-adapt3-full** (CAL 72.0 %; the strengthened variants C-map1-ms 65.0 %, C-ms10 60.0 %).

| set | best learned (L-dec4) | best classical (C0 = R0) | paired difference | R1 (true theta) | verdict |
|---|---|---|---|---|---|
| holdout n=500 | 70.3 [67.9, 72.8] | 69.2 [65.2, 73.2] | **+1.1 pp [-3.3, +5.7]** | 86.0 | **no side wins at the 10 pp bar** |
| new seeds n=200 | 68.0 [63.7, 72.2] | 67.5 [61.0, 74.0] | +0.5 pp [-6.7, +7.8] | 80.0 | same sign, nowhere near 10 pp |

Recovered fraction of the R0 -> R1 gap (+16.8 pp holdout, +12.5 pp new): L-dec4 **0.07** (holdout) / **0.04** (new); classical C0 = 0 by definition; the best classical *variants* recover a negative amount (HYB -0.33, MAP+multi-start negative on CAL). The learned side narrows the 17 pp gap by about one-tenth at the pre-registered point; the CI upper end on holdout is +5.7 pp, which excludes the 10 pp bar for this pair. (measured)

Not part of the rule, reported because the user asked for ensembles/posteriors as an option: the **seed-ensemble mean** of the three L-dec4 models (a single theta_hat per episode, no extra data) gets 74.0 [70.0, 77.8] on holdout = +4.8 pp [-0.8, +10.4] over R0 (recovered 0.29) and 71.0 on new seeds (+3.5 [-5.0, +12.0], recovered 0.28). It was not the pre-registered pick, it was evaluated after the verdict candidates were fixed, and its CI still contains 0; it is a hint, not a result.

### 1.2 What the tables show (measured unless marked inference)
* **Parameter error is much lower on the learned side; success is not.** Holdout RMS (ls / per-link log mass / log damping / friction in 1e-3 Nm): C0 0.034 / 0.132 / 0.693 / 0.63; L-mlp1 (spike10b) 0.027 / 0.121 / 0.474 / 0.56; L-dec4 0.016 / 0.094 / 0.323 / 0.50; classical MAP + multi-start (CAL) 0.034 / 0.133 / 0.531 / 0.60. L-dec4 halves the damping error and the length error of C0 and still recovers 7 % of the success gap. Friction is essentially unidentified by every method (prior sd 0.58, errors 0.50-0.65).
* **Precision is the bottleneck, not the direction of the error (post hoc diagnostic, `posthoc.md`).** True theta plus *independent* Gaussian noise of the size of L-dec4's holdout error gives 65.0 % (noise x1) and 74.0 % (noise x0.5) against 86.0 % for true theta. So errors of the size the probe data allow, if they were unstructured, would cost more than the whole R0 -> R1 gap; L-dec4's actual errors are better than iid noise of the same size (70.3 %), i.e. structured (shrinkage towards the mean). Recovering most of the 17 pp would need errors well below ~0.5 x the current learned error (inference: this is an information limit of the 0.6 s probe, which is why probe design is the untested lever).
* **The gain/loss pattern is a lottery at episode level (post hoc).** On the holdout, L-dec4 (seed 0) solves 89 of the 125 episodes that R1 solves and R0 does not (71 %), but fails 94 of the 346 episodes R0 solves (`posthoc.md`); net +10 episodes. The same happens for every estimator (L-mlp1: 77 converted, 108 lost). Success of the pipeline is chaotic in theta_hat: re-running the *same* spike10b L-est-full models on the *same* seeds with theta_hat differing only by float rounding (float32 features vs float64) moved holdout success from 65.6 / 65.6 / 62.8 % (spike13b) to 67.0 / 69.2 / 63.4 % (this spike): replicate noise of 1-3.5 pp per model run that no bootstrap over episodes captures. Differences below about 3-4 pp between single model runs are therefore not interpretable. (measured; the chaos explanation is inference)
* **Data dose.** x1 -> x4 training data lowers label error by 6-19 % (damping RMS 0.426 -> 0.355 for the plain GRU, 0.398 -> 0.323 for the decision-weighted GRU, 0.474 -> 0.443 for the MLP) but success does not follow: L-dec1 (seed 0 only) 70.2 % holdout / 73.0 % new vs L-dec4 (3 seeds) 70.3 / 68.0; GRU plain x1 (seed 0) 69.2 / 70.0 vs x4 (seed 0) 72.8 / 65.0; MLP x1 66.5 / 64.0 (3 seeds) vs x4 (seed 0) 67.0 / 65.0. Single-seed cells vary by +-3-4 pp from the replicate noise above, so the dose curve in success is flat within noise; the dose curve in parameter error is monotone. (measured)
* **Decision-weighted loss vs plain MSE.** Decision weighting lowers *plain* parameter error too (damping 0.323 vs 0.355 at x4, 0.398 vs 0.426 at x1) and was the best on CAL (73.2 vs 65.5-66.0 for the plain variants, seed-0). On holdout the plain x4 GRU seed 0 (72.8) is as good as L-dec4 seed 0 (71.2): the advantage seen on CAL did not reproduce (winner's curse of a CAL-selected pick). (measured)
* **Classical strengthening did not help the pipeline.** MAP prior + multi-start lowers damping error (0.698 -> 0.531 on CAL, 0.693 -> 0.638 on the holdout) but success does not improve: CAL 72.0 -> 65.0, holdout 67.2 [63.0, 71.2] = -2.0 pp [-7.6, +3.6] vs R0 (new seeds 68.0, +0.5); K=10 step-ahead resetting is worse on every parameter group and on success (CAL 60.0 %, new seeds 63.5 % = -4.0 pp [-12.5, +4.5]). MAP + multi-start on the new seeds: 68.0 % (+0.5 [-8.5, +9.5]). The hybrid (dec4-ensemble init, then the same MAP polish) converges to the same MAP solution as multi-start MAP (CAL errors essentially identical, 0.531 vs 0.531 log damping; ~10x cheaper: 5.8 vs 66.8 cpu-s per fit) and gets 63.6 % holdout / 67.5 % new. (inference: the fit-the-probe-data objective is not the planning-relevant one; sq/sv in the fit were inflated for model mismatch in spike10b, so the posterior mode is not the posterior mean, and the learned regressors, which estimate a posterior mean, behave differently.) C0 was tuned on this CAL in spike10b, which favours C0 in the CAL selection (CAL 72.0 vs holdout 69.2).

### 1.3 Budget (three axes; `budget.json`)
Full-episode equivalents: a probe-only episode counts 0.6/2.5 = 0.24 (spike10b convention), an oracle plan counts 1.
| method | episodes (full-episode equivalents) | wall-clock | labels |
|---|---|---|---|
| L-mlp1 (spike10b, frozen) | 12000 probe = 2880 | 645 s (whole spike10b estimator pipeline) | simulator theta13 x 12000 |
| L-gru1 / L-mlp4 / L-gru4 | 12000 / 48000 / 48000 probe = 2880 / 11520 / 11520 | data 166 s (+807 s for the extra 36000), training 147 s / 62 s / 198 s (3 seeds, GPU) | simulator theta13 |
| L-dec1 / **L-dec4** | 12000 / 48000 probe + 1600 oracle plans = 4480 / **13120** | data as above, sensitivities 1585 s (8 workers), training 147 s / 367 s | simulator theta13 + true plant for 1600 oracle plans |
| C0 (T-adapt3-full, spike10b) | 750 tuning episodes | 3007 s tuning, 7.5 cpu-s per fit unloaded | none (tuned on CAL success) |
| C-ms10, C-map1-ms, HYB (this spike) | 200 CAL fits each (48 equivalents) + 200 CAL pipeline episodes each | fits 1622 s / 1724 s / 151 s wall on a heavily loaded machine (inference cost per episode 12-67 s wall per worker vs milliseconds for any learned estimator) | CAL labels for screening only |
Learned (L-dec4) : classical (C0 + its screening variants, about 1500 equivalents) is roughly **9-17x**: far above the 2x flag. The ratio is flagged, not equalised; classical has no cheap way to turn more budget into accuracy (its tuning knobs were explored on CAL with 3 settings), learned scales with data (x1 -> x4 gave -6 to -19 % damping error). Inference cost points the other way: every classical fit costs seconds of simulator time per episode, a learned estimate is a forward pass.

### 1.4 Learned-side hyper-parameters actually tried (CAL only; all counted in the budget)
GRU 2 layers, hidden {96, 192} x lr {2e-3, 1e-3} at x1 (CAL z-MSE 0.097-0.115 -> H=192, lr 2e-3); epochs 60 (x1), 20/38 (x4), 150 (x1 check: best epoch 35-49, no gain); decision loss a in {0.5, 0.9} (identical label error, a=0.9 used); MLP x4 (spike10b recipe, H=256). Checkpoint = epoch with the lowest CAL loss. Sensitivity weights: oracle plan of 1600 training episodes, central finite differences of the planner's swing-model tip trajectory (100 values) w.r.t. the 13 parameters; 20 % flagged unstable were replaced by the mean weight matrix, the other 46400+ training episodes use the mean matrix as well (so the "per-episode" part of the weighting is for 1600 episodes only).

## 2. Deviations, limits, and what is not claimed
* **CPU contention changed the plan.** The machine ran at load average 25-60 from other sessions throughout (classical fits ran 2-3x slower than in spike10b). Consequences: (a) only three strengthened classical settings were tried (pw=1 MAP + multi-start with 8 random starts; K=10 step-ahead; hybrid), each at **one** hyper-parameter setting, not tuned on CAL; the single-knob variants (MAP alone, other pw, K=30) and probe-excitation optimisation (optional, would change the probe) were dropped. The classical side therefore got far fewer tuning rounds than the learned side (which swept H, lr, epochs, loss weight on CAL). A classical tuning effort that matched the learned one could change the classical numbers; I did not do it. (b) CAL pipeline success of the learned variants was measured on the seed-0 model for gru1, gru4, mlp4, dec1 and for all 3 seeds for L-mlp1 and L-dec4 (dec4 seeds 1, 2 were run after the selection and confirmed it: 70.5, 73.0 vs seed 0 76.0). The pre-registration said "mean over seeds"; the selection of L-dec4 rests on seed 0 (76.0 vs <=70.5) and was confirmed by the 3-seed mean (73.2). (c) Holdout/new evaluation of non-selected variants is partial (one seed each for gru1, gru4, mlp4, dec1; C-ms10 on CAL and new seeds only (not on the holdout); C-map1-ms and HYB on all three sets).
* The GPU queue was blocked by another project for ~55 minutes (20:12-21:06); I waited and did not use the CPU fallback. Neural-network training used `a GPU job queue` with the memory cap, 2 CPU threads.
* The spike13b R0/R1 estimates on the holdout were reused from caches for C0 and R1 (exact reproduction checked: 69.2 and 86.0). C-mlp1 on the holdout: this spike's re-run is the primary row, the spike13b run is listed beside it.
* Seeds: the new seeds 701000-701199, CAL 1000-1049 + 2000-2149, holdout 1100-1599; training seeds 500000-511999 (x1) and 540000-575999 (extra for x4); sensitivities from 500000-501599. `results/.test_access.log` has 109 lines (unchanged); no seed in 0-49 or 20000-20499 was used (`guard14`).
* Post hoc (not pre-registered): the seed-ensemble, the R1+noise diagnostic, the overlap analysis, the HYB and dec1/gru/mlp comparisons beyond the CAL selection.
* One plant family, simulation only; the learned estimators are in-distribution (train and evaluation draw parameters from the same distribution, sim labels), while classical uses the distribution only as a prior. Whether a learned estimator trained on a wrong prior holds up (spike10b OOD finding) was not re-tested here.
* "No side wins" is a statement at the 10 pp bar with the CIs above; a real +1 to +5 pp difference (and the ensemble's +4.8) cannot be excluded.

## 3. Files
`spike14a/` code (c14.py shared, gen_data.py, sens.py, nn_train.py, classical.py, run_classical.py, evaluate.py, export_preds.py, report14.py, posthoc14.py, budget14.py); `results/spike14a/` table.md, posthoc.md, summary.json, budget.json, selection.json, pipe_{cal,hold,new}.json (per-episode successes), classical_cost.json, nn_sweep_*.json. Large artefacts (raw probe data, predictions, models) are in `.scratch/s14/`.

## 4. Full generated tables (copy of table.md at the time of writing)

### spike14a tables (generated by spike14a/report14.py)

## CAL (1000-1049 + 2000-2149, n=200): selection set

| method | success % [95% CI] | vs R0 (pp [CI]) | recovered fraction of R0->R1 | unstable ep % | param RMS: ls / mass(link) / log-damp / fric(1e-3Nm) |
|---|---|---|---|---|---|
| L-mlp1 (spike10b L-est-full) | 67.3 [63.2, 71.3] | -4.7 [-11.5, +2.2] | -0.78 | 0.0 | 0.026 / 0.124 / 0.458 / 0.52 |
| L-dec4 a0.9 | 73.2 [69.2, 77.0] | +1.2 [-5.7, +8.2] | 0.19 | 0.0 | 0.014 / 0.097 / 0.309 / 0.48 |
| L-ens(dec4) | 72.5 [66.0, 78.5] | +0.5 [-8.0, +9.0] | 0.08 | 0.0 | 0.014 / 0.095 / 0.304 / 0.48 |
| C0 T-adapt3-full (=R0) | 72.0 [66.0, 78.0] | - |  | 0.0 | 0.034 / 0.132 / 0.698 / 0.65 |
| C-ms10 (K=10 step-ahead) | 60.0 [53.0, 66.5] | -12.0 [-21.0, -3.0] | -2.00 | 0.0 | 0.053 / 0.147 / 0.831 / 0.57 |
| C-map1-ms (MAP + multi-start) | 65.0 [58.5, 71.5] | -7.0 [-15.5, +1.5] | -1.17 | 0.0 | 0.034 / 0.133 / 0.531 / 0.60 |
| R1 true theta | 78.0 [72.0, 83.5] | +6.0 [-2.5, +14.5] |  | 0.0 | 0.000 / 0.000 / 0.000 / 0.00 |

## Holdout 1100-1599 (n=500)

| method | success % [95% CI] | vs R0 (pp [CI]) | recovered fraction of R0->R1 | unstable ep % | param RMS: ls / mass(link) / log-damp / fric(1e-3Nm) |
|---|---|---|---|---|---|
| L-mlp1 (spike10b L-est-full) | 66.5 [63.9, 69.1] | -2.7 [-7.4, +2.1] | -0.16 | 0.0 | 0.027 / 0.121 / 0.474 / 0.56 |
| L-mlp1 spike13b original run (holdout only) | 64.7 [62.1, 67.3] | -4.5 [-9.5, +0.5] | -0.27 | 0.0 |  |
| L-dec4 a0.9 | 70.3 [67.9, 72.8] | +1.1 [-3.3, +5.7] | 0.07 | 0.0 | 0.016 / 0.094 / 0.323 / 0.50 |
| L-ens(dec4) | 74.0 [70.0, 77.8] | +4.8 [-0.8, +10.4] | 0.29 | 0.0 | 0.015 / 0.093 / 0.318 / 0.50 |
| C0 T-adapt3-full (=R0) | 69.2 [65.2, 73.2] | - |  | 0.0 | 0.034 / 0.132 / 0.693 / 0.63 |
| C-map1-ms (MAP + multi-start) | 67.2 [63.0, 71.2] | -2.0 [-7.6, +3.6] | -0.12 | 0.0 | 0.036 / 0.134 / 0.638 / 0.59 |
| HYB (dec4-ensemble init + MAP polish) | 63.6 [59.4, 67.8] | -5.6 [-11.0, -0.2] | -0.33 | 0.0 | 0.036 / 0.134 / 0.639 / 0.59 |
| R1 true theta | 86.0 [83.0, 89.0] | +16.8 [+12.0, +21.6] |  | 0.0 | 0.000 / 0.000 / 0.000 / 0.00 |

Per training seed (hold):

| model | success % | param RMS ld |
|---|---|---|
| lmlp1_s0 | 67.0 | 0.471 |
| lmlp1_s1 | 69.2 | 0.471 |
| lmlp1_s2 | 63.4 | 0.48 |
| lmlp1_13b_s0 | 65.6 |  |
| lmlp1_13b_s1 | 65.6 |  |
| lmlp1_13b_s2 | 62.8 |  |
| gru1_s0 | 69.2 | 0.422 |
| gru4_s0 | 72.8 | 0.358 |
| mlp4_s0 | 67.0 | 0.442 |
| dec1_a0.9_s0 | 70.2 | 0.402 |
| dec4_a0.9_e38_s0 | 71.2 | 0.32 |
| dec4_a0.9_e38_s1 | 70.4 | 0.321 |
| dec4_a0.9_e38_s2 | 69.4 | 0.327 |

## New seeds 701000-701199 (n=200)

| method | success % [95% CI] | vs R0 (pp [CI]) | recovered fraction of R0->R1 | unstable ep % | param RMS: ls / mass(link) / log-damp / fric(1e-3Nm) |
|---|---|---|---|---|---|
| L-mlp1 (spike10b L-est-full) | 64.0 [59.7, 68.2] | -3.5 [-10.8, +3.8] | -0.28 | 0.0 | 0.026 / 0.121 / 0.427 / 0.54 |
| L-dec4 a0.9 | 68.0 [63.7, 72.2] | +0.5 [-6.7, +7.8] | 0.04 | 0.0 | 0.014 / 0.093 / 0.293 / 0.48 |
| L-ens(dec4) | 71.0 [64.5, 77.0] | +3.5 [-5.0, +12.0] | 0.28 | 0.0 | 0.014 / 0.092 / 0.287 / 0.48 |
| C0 T-adapt3-full (=R0) | 67.5 [61.0, 74.0] | - |  | 0.0 | 0.036 / 0.137 / 0.759 / 0.63 |
| C-ms10 (K=10 step-ahead) | 63.5 [57.0, 70.0] | -4.0 [-12.5, +4.5] | -0.32 | 0.0 | 0.050 / 0.153 / 0.768 / 0.57 |
| C-map1-ms (MAP + multi-start) | 68.0 [61.5, 74.5] | +0.5 [-8.5, +9.5] | 0.04 | 0.0 | 0.037 / 0.131 / 0.601 / 0.55 |
| HYB (dec4-ensemble init + MAP polish) | 67.5 [61.0, 74.0] | +0.0 [-9.5, +9.0] | 0.00 | 0.0 | 0.037 / 0.131 / 0.599 / 0.55 |
| R1 true theta | 80.0 [74.5, 85.5] | +12.5 [+4.5, +20.5] |  | 0.0 | 0.000 / 0.000 / 0.000 / 0.00 |

Per training seed (new):

| model | success % | param RMS ld |
|---|---|---|
| lmlp1_s0 | 63.0 | 0.429 |
| lmlp1_s1 | 61.5 | 0.428 |
| lmlp1_s2 | 67.5 | 0.423 |
| gru1_s0 | 70.0 | 0.364 |
| gru4_s0 | 65.0 | 0.299 |
| mlp4_s0 | 65.0 | 0.375 |
| dec1_a0.9_s0 | 73.0 | 0.35 |
| dec4_a0.9_e38_s0 | 65.5 | 0.301 |
| dec4_a0.9_e38_s1 | 69.5 | 0.284 |
| dec4_a0.9_e38_s2 | 69.0 | 0.293 |

## Dose curve: training data x1 (12000 probe episodes) vs x4 (48000); label error = mean over the 3 training seeds, success = mean over the seeds that were run through the pipeline (n_seeds given)

| model family | data | holdout RMS ls / mass(link) / log-damp / fric | holdout success % (n_seeds) | new-seed success % (n_seeds) | CAL RMS log-damp |
|---|---|---|---|---|---|
| MLP (spike10b recipe) | x1 | 0.027 / 0.121 / 0.474 / 0.56 | 66.5 (3) | 64.0 (3) | 0.458 |
| MLP (spike10b recipe) | x4 | 0.023 / 0.116 / 0.443 / 0.54 | 67.0 (1) | 65.0 (1) | 0.406 |
| GRU, plain MSE | x1 | 0.021 / 0.116 / 0.426 / 0.55 | 69.2 (1) | 70.0 (1) | 0.404 |
| GRU, plain MSE | x4 | 0.018 / 0.101 / 0.355 / 0.51 | 72.8 (1) | 65.0 (1) | 0.354 |
| GRU, decision-weighted loss | x1 | 0.020 / 0.111 / 0.398 / 0.54 | 70.2 (1) | 73.0 (1) | 0.369 |
| GRU, decision-weighted loss | x4 | 0.016 / 0.094 / 0.323 / 0.50 | 70.3 (3) | 68.0 (3) | 0.309 |

## Primary judgement (rule fixed in README section 0)

* hold: best learned [L-dec4 a0.9] - best classical [C0 T-adapt3-full (=R0)] = +1.1 pp [-3.3, +5.7]  (learned 70.3, classical 69.2)
* new: best learned [L-dec4 a0.9] - best classical [C0 T-adapt3-full (=R0)] = +0.5 pp [-6.7, +7.8]  (learned 68.0, classical 67.5)

**Verdict: no side wins at the 10 pp bar**

## 5. Post-hoc tables (copy of posthoc.md)

### hold: episode-level overlap with R0 (C0) and R1 (n=500)

| method | success pct | gained vs R0 (R0 fail -> success) | lost vs R0 (R0 success -> fail) | converted of the R1-only episodes (R1 success and R0 fail, n=125) | success pct when R1 fails (n=70) |
|---|---|---|---|---|---|
| dec4 (mean of 3 seeds, rounded to majority) | 76.2 | 106 | 71 | 87/125 = 70% | 69% |
| dec4 seed 0 | 71.2 | 104 | 94 | 89/125 = 71% | 64% |
| ens_dec4 | 74.0 | 115 | 91 | 93/125 = 74% | 71% |
| lmlp1 s0 | 67.0 | 97 | 108 | 77/125 = 62% | 69% |
| gru4 s0 | 72.8 | 111 | 93 | 90/125 = 72% | 69% |
| C_map1_ms | 67.2 | 101 | 111 | 84/125 = 67% | 66% |
| HYB | 63.6 | 84 | 112 | 70/125 = 56% | 59% |

### new: episode-level overlap with R0 (C0) and R1 (n=200)

| method | success pct | gained vs R0 (R0 fail -> success) | lost vs R0 (R0 success -> fail) | converted of the R1-only episodes (R1 success and R0 fail, n=49) | success pct when R1 fails (n=40) |
|---|---|---|---|---|---|
| dec4 (mean of 3 seeds, rounded to majority) | 72.0 | 41 | 32 | 32/49 = 65% | 57% |
| dec4 seed 0 | 65.5 | 40 | 44 | 28/49 = 57% | 55% |
| ens_dec4 | 71.0 | 42 | 35 | 33/49 = 67% | 50% |
| lmlp1 s0 | 63.0 | 37 | 46 | 30/49 = 61% | 48% |
| gru4 s0 | 65.0 | 39 | 44 | 31/49 = 63% | 60% |
| C_map1_ms | 68.0 | 43 | 42 | 34/49 = 69% | 57% |
| HYB | 67.5 | 45 | 45 | 34/49 = 69% | 65% |

### R1 + noise diagnostic (holdout, n=500; post hoc): TRUE theta plus iid Gaussian noise of the size of the selected learned estimator's holdout error (ls 0.016, per-link log-mass 0.094, log-damping 0.320, friction 0.50e-3 Nm; noise x1 and x0.5)

| plant parameters given to the planner | success % |
|---|---|
| true theta | 86.0 |
| true theta + noise x0.5 | 74.0 |
| true theta + noise x1 | 65.0 |
| R0 fit | 69.2 |
| learned dec4 (seed 0) | 71.2 |
