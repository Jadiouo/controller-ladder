# spike16a: fair-protocol mini tables for catch, toss and Push-T (paper appendix; all 7 tasks get a result)

Statements marked **(inference)** are interpretation, not measurement. Section 0 was written BEFORE any spike16a run produced a number (the only things run before it: a smoke test of the code on CAL seeds 1000-1002, 3 episodes per task at s=1 with default parameters, outcomes seen but not used for any choice; and a 4-episode wall-clock probe of the default teachers on DEMO seeds 5100-5103 to size the budgets: catch 0.14 s/episode, toss 0.42 s, Push-T 2.8 s on the second machine; their outcomes are not used) and is not edited afterwards. Sections 1+ are filled after the runs. Anything that deviates from section 0 is listed in section "Deviations" with the reason.

## 0. Pre-registered design (frozen before any result)

**Claim under test.** "With the same tools and budget, a classical controller ties or beats a learned one; learning has no >= 10 pp win." This spike can support it (ties / classical better on all three) or refute it (a pre-registered WIN on any task). The fairness of the protocol is fixed first; the outcome is not used to change it.

**Where it runs.** All simulation and training on the second machine (`ssh machine-B`, `~/robot_arm_control`, CPU workers <= 6, GPU unused; the PPO nets are 2x256 and run on CPU, so the one-GPU-program rule is not touched). New code lives in `experiments/spike16a/` (uncommitted, copied with scp). Nothing is run locally except the 10-episode local-vs-machine-B comparison of section 0.9 (wrapped in `flock .scratch/heavy.lock systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`).

### 0.1 Seeds and guard
* CAL = 1000-1049 + 2000-2149 (n=200). CAL-A = 1000-1049 (n=50). CAL-S = 1000-1049 + 2000-2049 (n=100).
* HOLD (holdout) = 1100-1599 (n=500). NEW = 705000-705199 (n=200).
* Additions needed by the design and disclosed here: GATE = 706000-706299 (n=300, new; used only for the sanity gate), training scenarios = seeds >= 510000 (never evaluated).
* LOCKED, never touched: 0-49, 20000-20499, 30000-30499, 40000-40499. Every episode function calls `guard(seed)`, which raises on a locked seed and on any seed outside CAL / HOLD / NEW / GATE / TRAIN. `results/.test_access.log` is not written by this spike (the machine-B clone has its own copy and the test-log function is never called).
* Selection of anything (strength, classical parameters, log-std0, checkpoint) uses CAL / CAL-A / CAL-S / GATE only; HOLD and NEW are used once, at the end, for the final comparison.

### 0.2 The three "real conditions" (unknown, different every episode)
Each episode draws its true parameters from `rng([seed, 16016])` (so the same seed is the same condition for every method: paired). Strength `s` multiplies every range; all relative changes are clipped to >= -0.8. The controller always keeps the NOMINAL model and never sees the drawn values.
* **catch**: air drag rel = s*U(-0.5, 2.0); ball mass rel = s*U(-0.3, 1.5); ball-position noise std = s*U(0, 8) mm; ball-position observation delay = floor(s*U(0, 3.5)) control steps (5 ms each).
* **toss**: air drag rel = s*U(-0.5, 2.0); object mass rel = s*U(-0.3, 1.5); release delay = s*U(0, 60) ms; joint friction Coulomb s*U(0, 4) Nm and viscous s*U(0, 0.8); joint sensor noise q std s*U(0, 0.002) rad, qd std s*U(0, 0.05) rad/s.
* **push_t**: friction coefficient rel (pusher cone and table) = s*U(-0.5, 1.0); block mass rel = s*U(-0.3, 1.0); COM offset dx, dz = s*U(-15, 15) mm each; block-pose noise pos s*U(0, 4) mm, angle s*U(0, 0.03) rad; observation delay (q, qd, pose) = floor(s*U(0, 8)) steps.

### 0.3 Strength calibration rule (CAL only)
Grid s in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}. Step 1: evaluate the DEFAULT classical teacher on CAL-A at ascending s; s0 = the first grid value at which its success is < 55 % (if none, 2.0). Step 2: tune the classical (0.4) at s0 on CAL. Accept s0 if the tuned classical's CAL success is in [60 %, 85 %]. If > 85 %: re-do step 2 once at the next larger grid value (ceiling 2.0); if < 60 %: once at the next smaller value (floor 0.25). The result of the (at most one) re-tune is final even if it is outside the band, and is reported as such. The tuned-on-CAL success is optimistic (selection on the same seeds); the HOLD number is the honest one and is reported next to it.

### 0.4 Strongest classical under an equal tuning budget
* catch: `catch_teacher` extended (subclass in `experiments/spike16a/h16.py`, arena code untouched) with two robustness parameters, `delay_hat` (timestamp shift of the measured ball position) and `k_scale` (scale of the assumed drag rate). Tuned vector (12): fit_window, t_first, replan_dt, tmin, tau_s, d_pre, brake_T, torque_frac, log10 qp, log10 qv, delay_hat, k_scale (ranges in `h16.py`, written before any run).
* toss: `toss_teacher` extended with `k_scale`, `t_shift` (the planned release time is t_r + t_shift: delay compensation), `v_scale` (gain on the planned launch speed: compensates friction / mass losses), brake_T, torque_frac, log10 qp, log10 qv.
* push_t: `push_t_teacher` extended with `mu_scale` (assumed friction coefficient of the planner), tuned with v_push, v_approach, kp, kd, hold_s, travel_w.
* Both "robust fixed parameters" and an online-adaptive element exist only where the controller has a signal to adapt on; none of the three teachers gets an online identifier (a limitation, declared; the earlier spikes' T-adapt do not exist for these tasks).
* Optimiser (same for all tasks, deterministic): a (3/3, 10)-ES in the unit cube of the log / linear-scaled parameters, start = the default teacher, sigma0 = 0.25, per-generation recombination of the best 3 and a 1/5-th-rule-like sigma step (x0.85 / x1.15 if the best fitness did / did not improve). Candidates are scored on the whole tuning set with common seeds. Fitness = CAL success + 0.1 * smooth, smooth = catch: -mean(min(min_dist, 1)); toss: -mean(min(landing_err, 1)); push_t: mean(max_cov). The classical reported is the best-fitness vector over all evaluated candidates (the default is candidate 0) = "tuned"; the "strongest classical" is the better of {default, tuned} by CAL success (ties -> tuned).
* Tuning budget (episodes = candidates x tuning-set size; 1 default + generations x 10): catch 81 x 200 = 16 200 (8 generations); toss 51 x 200 = 10 200 (5); push_t 21 x 100 (CAL-S) = 2 100 (2). Sized from the 4-episode wall-clock probe so that the whole spike fits 8 h. Wall clock and label source (success flag of the simulator = the only supervision a classical tuner gets) are reported.

### 0.5 Learned method: residual PPO on top of the strongest classical
* Action: tau = clip(tau_classical + bias + 12 Nm * clip(a, -1, 1)), held for `rep` control steps (catch 4, toss 4, push_t 10); the teacher is called at every control step so its internal state stays correct. Zero action = exactly the classical controller.
* Policy / value: tanh MLP 2x256 (orthogonal init, last actor layer zero), learned log-std (init `log-std0`, floor -5), PPO clip 0.2, GAE 0.95, gamma 0.99 (push_t 0.995), lr 1e-4, 4 epochs, minibatch 2048, grad-clip 0.5, 6 CPU workers x several envs. Training scenarios are drawn from the real-condition distribution at the calibrated strength with seeds >= 510000.
* Observation (no privileged information): joint q, qd, the classical torque, time, and only what the controller also measures (catch: tip-relative last 4 measured ball positions; toss: basket position, time to release; push_t: measured block pose, pusher position, goal-relative pose, pose 10 / 40 steps ago). Reward uses simulator state (catch: ball-tip distance; toss: landing error at the rim; push_t: coverage progress) = privileged label source, same in all tasks.
* Budget: training + CAL-S checkpoint evaluation episodes (7 checkpoints x 100 + the init evaluation x 100 are INCLUDED) = 1.0x the classical tuning episodes: catch 16 200, toss 10 200, push_t 2 100 per run (so the training episodes alone are 15 400 / 9 400 / 1 300). The gate's Stage-A runs use half of it. 1x (not 2x) because of the 8 h wall clock; with 3 training seeds the learner's total is 3x (declared; the classical has one tuning run).
* **Sanity gate (per task, before any real-condition number), same design as spike15b section 0.** In the real-condition environment a constant torque bias b*(+1, -1, +1) Nm (push_t: (+1, -1)) is added on every step after the start. b = the smallest value in {1, 2, 3, 4, 6, 8, 10} Nm for which the zero-residual strongest classical has CAL-A success <= 40 %; if none, 10. Stage A: log-std0 in {-2.0, -1.5}, 1 training seed each, 0.5x of the real-run budget; the one with the higher CAL-S score of its selected checkpoint is chosen (tie -> -2.0). PASS = the chosen run's CAL-S-selected checkpoint beats the zero-residual init on GATE seeds (bias on, paired, n=300) by >= 10 pp with the paired bootstrap 95 % CI lower bound > 0. If Stage A fails: Stage B = also evaluate the other log-std0 on GATE and retrain both at 1x of the real-run budget; the first of {chosen-0.5x, other-0.5x, chosen-1x, other-1x} that passes defines the recipe. If none passes: "gate failed, learner uninterpretable for this task", and the real condition is NOT run for it. If the zero-residual init is already > 90 % on GATE (so +10 pp is nearly impossible), the gate is "not testable"; the real condition is then still run with the chosen log-std0 and labelled "gate not testable" (added before any gate number existed, after the catch calibration showed the classical at the ceiling).
* Real condition (only after the gate passed): the gate-passing recipe unchanged (log-std0, per-run budget of above), 3 training seeds (0, 1, 2). Checkpoint = best of 7 equally spaced on CAL-S (the init is not a candidate, as in spike15b). Evaluated deterministically (mean action) on HOLD and NEW with the SAME checkpoints. Pooled estimate = per-episode mean over the 3 seeds.

### 0.6 Verdict rules (per task; learned minus classical, success rate, paired percentile bootstrap, 10 000 resamples)
* **LEARNED WINS** = pooled diff >= +10 pp AND 95 % CI lower bound > 0 on BOTH HOLD and NEW.
* **TIE** = the 90 % CI of the pooled diff lies inside [-10, +10] pp on BOTH HOLD and NEW (TOST at +-10 pp).
* **UNCERTAIN** = anything else.
* Descriptive only (not a verdict): CLASSICAL BETTER if the 95 % CI upper bound < 0 on both sets. Holm adjustment over the three tasks is applied to the three HOLD win-tests (one-sided bootstrap p); a WIN whose Holm-adjusted p >= 0.05 is called "WIN (not Holm-robust)". The classical side has no "win" rule: the claim is about learning not winning by >= 10 pp.
* Reported next to each verdict: classical and learned success (HOLD, NEW), the diff with CI, residual norm of the learned policy, per-seed diffs.

### 0.7 Budget reported on three axes
For the classical tuner and for every RL run: (1) episodes (and control steps), (2) wall clock, (3) label source (what supervision / privileged information was used). Local machine: no compute.

### 0.8 Optional student
A BC / flow student is NOT pre-registered as a deliverable: there is no off-policy labeller for these stateful teachers (the teachers keep an internal plan), so an honest distillation baseline would be a pure BC of the tuned classical, which cannot exceed it by construction. It is run only if more than 2 h of wall clock remain after everything else; otherwise it is reported as "not done".

### 0.9 Local vs machine-B numerical check
After the machine-B runs, the strongest classical of each task is run on CAL seeds 1000-1009 (real condition) on both machines (machine-B: numpy 2.2.6 / scipy 1.15.3 / mujoco 3.17.0; local: numpy 2.4.6 / scipy 1.17.1 / mujoco 3.17.0). Reported per episode: outcome, end time, final joint state; agreement rate of outcomes and the largest absolute state difference.

### 0.10 Known limitations declared in advance
(a) The classical tuner is a 12-dimensional ES with a small budget; a stronger classical might exist (this favours the learner). (b) RL gets 1x episodes for toss and push_t (this favours the classical). (c) The sanity gate tests representability of a constant correction, not that a real, state-dependent error is learnable. (d) 3 training seeds; each task has one real-condition distribution.

### 0.11 Extension E1 (EXPLORATORY, added during the run; not part of the main verdicts)
Added after the calibration stage had finished (so after seeing that the tuned classical sits at the ceiling for catch, CAL 100 %, and push_t, CAL-S 100 %, at the calibrated strength) and after the first catch gate number existed, but before any real-condition number existed. Purpose: a main-protocol TIE at the ceiling says little about learning (the learner cannot gain +10 pp when the classical is at ~100 %), so E1 repeats the whole pipeline once at the top of the strength grid, s = 2.0, for those tasks whose main-protocol tuned classical had CAL success > 85 % (catch, push_t), in directories `results/spike16a/<task>_x/`. Everything else is identical (same ES budget, same gate rule, same recipe, same seeds, same verdict rules, no further re-tuning). It runs only if the main queue has finished with more than the needed wall clock left (catch ~1 h, push_t ~2.5 h); each one that does not run is reported as "not done". Its verdicts are labelled "E1 (exploratory)" and are never merged into the main table.

---

# Results (written after the runs)

## 1. Summary (the three appendix tables; generated numbers are in `table.md`, produced by `experiments/spike16a/report16.py`)

Common: "classical" = the strongest classical of section 0.4 (always the ES-tuned one; the default teacher was worse on CAL in all three tasks), "learned" = residual PPO (12 Nm bound) on top of it, deterministic, CAL-S-selected checkpoint, paired per episode with the classical (same seed = same hidden parameters). Success rate, paired bootstrap CI (10 000 resamples), difference in percentage points (pp). WIN rule = >= +10 pp with 95 % CI > 0 on HOLD and NEW; TIE = 90 % CI inside +-10 pp on both.

**Table A1: catch** (strength s = 1.0; real condition = drag x(1-0.5..+2), ball mass x(1-0.3..+1.5), position noise 0-8 mm, observation delay 0-3 steps, all drawn per episode)

| | HOLD (n=500) | NEW (n=200) | budget: episodes / wall / label source |
|---|---|---|---|
| default teacher (47 % on CAL at s=1.0) | not run on HOLD | not run on NEW | none |
| **strongest classical (ES-tuned, 12 parameters)** | **99.2 %** | **100.0 %** | 16 200 episodes / 6 min (6 workers) / simulator success flag |
| **learned (residual PPO, log-std0 -2.0, 3 seeds pooled)** | **98.7 %** | **99.3 %** | 16 227-16 266 episodes per seed (train 15.4 k + CAL-S 0.8 k) / 13 min per seed / privileged ball-tip distance in the reward only |
| learned - classical (95 % CI) | -0.5 pp [-1.1, +0.1] | -0.7 pp [-1.8, 0.0] | |
| 90 % CI | [-1.0, 0.0] | [-1.7, 0.0] | |
| **verdict** | **TIE** (WIN impossible: classical is at the ceiling) | | Holm-adj. p (HOLD win test) 0.98 |

Gate (injected 6 Nm bias; zero-residual classical 30 % on CAL-A): log-std0 -2.0 chosen (tie with -1.5, both 1.00 on CAL-S), GATE seeds: init 44.0 % -> 98.0 %, **+54.0 pp [+48.3, +59.7]: PASS**. Per-seed HOLD differences -0.6, -0.4, -0.4 pp; mean residual 1.1-3.5 Nm.

**Table A2: toss** (s = 1.5; real condition = drag x(1-0.5..+2), object mass x(1-0.3..+1.5), release delay 0-60 ms, joint friction Coulomb 0-4 Nm + viscous 0-0.8, q / qd sensor noise, all x1.5, drawn per episode)

| | HOLD (n=500) | NEW (n=200) | budget |
|---|---|---|---|
| default teacher (CAL-A at s=1.5: 52 %; CAL 53 %) | not run | not run | none |
| **strongest classical (ES-tuned, 6 parameters)** | **68.0 %** | **68.5 %** | 10 200 episodes / 12 min / simulator success flag |
| learned (residual PPO) | 68.6 % (descriptive) | 67.5 % (descriptive) | 10 222 episodes (train 9 422 + CAL-S 800), 1 seed, log-std0 -1.5 / 44 min / privileged object-basket distance and landing error in the reward only |
| learned - classical (95 % CI) | +0.6 pp [-1.2, +2.4] (90 % CI [-0.8, +2.0]) | -1.0 pp [-3.0, +1.0] (90 % CI [-2.5, +0.5]) | |
| **verdict** | **NO VERDICT: the sanity gate FAILED** (4 attempts, see below). The learned row above is a single-seed DESCRIPTIVE run started outside the pre-registered rule (after the gate failure, `S16_FORCE=1`); with the verdict rules it would read TIE on both sets, but it is not a verdict. | | |

Gate (fallback bias 10 Nm because no b <= 10 pushed the zero-residual classical to <= 40 % on CAL-A: 78, 76, 78, 82, 76, 72, 68 % for b = 1, 2, 3, 4, 6, 8, 10): GATE seeds, zero-residual init with bias 54.3 %. Stage A (0.5x budget): log-std0 -1.5 (chosen, CAL-S 0.69 vs 0.66) +1.0 pp [-0.3, +2.7]; -2.0 +0.0 pp [-1.0, +1.0]. Stage B (1x budget): -1.5 +0.3 pp [-0.7, +1.7]; -2.0 +1.3 pp [0.0, +3.0]. None reached +10 pp: **gate FAILED, "pipeline not validated in this environment", so the pre-registered real condition was not run** (the descriptive run above was added afterwards). **(inference)** The failure is largely a gate-design problem rather than evidence about toss: the toss teacher's computed-torque loop absorbs a constant bias (success barely changes with b), so the injected bias creates no representable correction to recover, and the gate degenerates into "can PPO improve the toss classical by 10 pp", which it cannot at this budget either.

**Table A3: Push-T** (s = 1.0; real condition = friction x(1-0.5..+1), block mass x(1-0.3..+1), COM offset +-15 mm, pose noise 0-4 mm / 0-0.03 rad, observation delay 0-8 steps)

| | HOLD (n=500) | NEW (n=200) | budget |
|---|---|---|---|
| default teacher (64 % on CAL-S at s=1.0) | not run | not run | none |
| **strongest classical (ES-tuned, 7 parameters)** | **99.0 %** | **99.5 %** | 2 100 episodes / 17 min / simulator success flag |
| **learned (residual PPO, log-std0 -2.0, 1 training seed)** | **99.4 %** | **98.0 %** | 2 114 episodes (train 1 314 + CAL-S 800) / 51 min / privileged coverage in the reward only |
| learned - classical (95 % CI) | +0.4 pp [-0.4, +1.2] | -1.5 pp [-3.5, 0.0] | |
| 90 % CI | [-0.2, +1.0] | [-3.0, -0.5] | |
| **verdict** | **TIE** (WIN impossible: classical is at the ceiling) | | Holm-adj. p 0.67 |

Gate (injected 2 Nm bias; zero-residual classical 38 % on CAL-A): log-std0 -2.0 chosen (CAL-S 0.98 vs 0.95), GATE seeds: init 30.7 % -> 92.0 %, **+61.3 pp [+55.3, +67.3]: PASS**. **Only ONE training seed** (not the pre-registered 3) because of the 8 h limit; mean residual 0.30 Nm.

## 2. Reading (inference unless stated)

* **Measured:** in no task did the learned residual beat the strongest classical by >= 10 pp; catch and Push-T are pre-registered TIEs (90 % CI inside +-10 pp on HOLD and NEW), toss has no verdict. The claim "learning has no >= 10 pp win" is therefore *not refuted* by these three tasks.
* **But the support is weak, and the reason is structural (inference).** In catch and Push-T the tuned classical is already at 99-100 % on the real condition, so a +10 pp win was arithmetically impossible: the TIEs say "the learner did no harm / no good where there is no room", not "the learner is as good as the classical where the classical is hard". The calibration rule (section 0.3) did not achieve the intended 60-85 % classical band for catch (CAL 100 %) and Push-T (CAL-S 100 %, HOLD 99 %); the rule allowed only one re-tune and was followed. Toss (classical 67.5 % CAL, 68.0 % HOLD) is the only task with room, and there the gate failed; the descriptive single-seed run (+0.6 / -1.0 pp) shows the residual changing nothing measurable (CAL-S selected checkpoints were all 0.75-0.76 = the init), consistent with a policy that learned ~nothing (inference). So the three tables add two "ceiling TIEs" and one "no verdict" to the seven-task picture; they should be cited as "no contradicting evidence", not as additional positive evidence of classical-parity under difficulty.
* **The default teacher is far from the tuned one** (catch 47 % -> 100 %, Push-T 64 % -> 100 %, toss 53 % -> 67.5 % on CAL at the final strength): a lot of the "classical" strength in this protocol is the equal-budget tuning itself (12 / 6 / 7 parameters, 16 200 / 10 200 / 2 100 episodes), which is what the paper's protocol claims.
* **Learner side:** the PPO pipeline demonstrably learns an injected constant correction in catch (+54 pp) and Push-T (+61 pp) but not in toss; on the real conditions it only slightly lowers catch (-0.5 / -0.7 pp) and NEW Push-T (-1.5 pp, 95 % CI upper bound 0.0), consistent with exploration noise costing a little at a ceiling.

## 3. Budget summary (three axes)

| | episodes | wall clock (machine-B, 6 workers) | label source |
|---|---|---|---|
| catch classical tuning | 16 200 | 6 min (+ 6 min first tune at s=0.75, discarded by the rule) | simulator success flag (+ smooth term in the fitness) |
| catch learned | 3 x ~16.25 k (+ gate 2 x ~8.1 k) | 3 x 13 min (+ gate 2 x 7 min) | privileged ball / tip state in the reward |
| toss classical tuning | 10 200 | 12 min | same |
| toss learned | gate: 2 x ~5.1 k + 2 x ~10.2 k; no pre-registered real run; descriptive run 10 222 | gate 2 x 21 min + 2 x 42 min; descriptive 44 min | same |
| push_t classical tuning | 2 100 (+ 2 100 first tune at s=1.5, discarded by the rule) | 17 min (+ 15 min) | same |
| push_t learned | 1 x 2 114 (+ gate 2 x ~1.06 k) | 51 min (+ gate 2 x 21 min) | privileged coverage in the reward |
Learner : classical episode ratio = 1.0x per seed (catch 1.00, toss 1.00 for the descriptive run, Push-T 1.01), so with 3 (catch) / 1 (Push-T) seeds the learner side spent 3.0x / 1.0x the classical tuning episodes in total, plus the gate runs. The classical wall clock is much shorter than the learner's (6 vs 13 min for catch, 17 vs 51 min for Push-T).

## 4. Local vs machine-B numerical check (section 0.9)

Strongest classical, real condition, CAL seeds 1000-1009 (10 episodes per task, 30 in total), numpy 2.4.6 / scipy 1.17.1 (local) vs numpy 2.2.6 / scipy 1.15.3 (machine-B), mujoco 3.17.0 on both: **outcome identical 30/30, episode length identical 30/30, joint-position trajectories (every physics step, `qpos`) bitwise identical (max |diff| = 0.0 in all 30 episodes); the only non-zero difference is 7.8e-16 in the Push-T `max_cov` summary (floating-point reduction order)**. Detail: `cmp_summary.txt`, `cmp_machineA.json`, `cmp_machineB.json`. (inference) the numpy / scipy version difference does not change these simulations; MuJoCo and the arena code do all the numerics.

## 5. Deviations (all of them)

1. Before the first run: a 3-episode-per-task smoke test on CAL seeds 1000-1002 and a 4-episode DEMO wall-clock probe (header). Budgets (ES generations, RL episodes) were set after the probe and before any run; the first draft of section 0 had larger numbers (101 / 61 / 31 evaluations, 2x RL for catch) which were cut to fit 8 h and rewritten in place before the first run.
2. Section 0.5 "gate not testable" clause and section 0.11 (E1) were added during the run (after the calibration results, before any gate / real number for the task concerned; for catch the gate number came before the E1 text). E1 was **not run** (no wall clock left).
3. Calibration (0.3): catch and Push-T ended outside the 60-85 % band (100 %), toss inside (67.5 %); the one-re-tune rule was followed (catch s 0.75 -> 1.0, Push-T s 1.5 -> 1.0).
4. Push-T was trained with **1** seed, not 3 (wall clock). A second seed was started and killed after ~1 min to give the wall clock to the toss descriptive run.
5. The toss real condition was **not run** under the pre-registered rule (gate failed). A single-seed descriptive run was added afterwards with `S16_FORCE=1` (log-std0 -1.5 = the Stage-A choice, same budget as a real run). It is labelled descriptive and never a verdict.
6. Episode budgets are overshot by at most one PPO iteration (e.g. catch 16 227-16 266 vs 16 200).
7. The 10-episode machine-B run of `local_cmp.py` ran as a single extra process next to the 6 training workers (7-8 busy CPUs on the second machine, transient).
8. The optional BC / flow student (0.8) was **not done**.
9. Held-out data: HOLD and NEW were each used once per final comparison for the classical and the learned policies. The test-log of the main repo was not touched; the locked seed sets were never used (the guard in `experiments/spike16a/h16.py` raises on them).
10. Push-T tuning: the "tuned" CAL-S success of 100 % is selection-biased (n=100, 21 candidates); HOLD 99.0 % shows the shortfall is small.

## 6. Reproduce

`experiments/spike16a/`: `h16.py` (conditions, teachers, wrapper), `s16.py {cal,gate,real} TASK`, `run_queue.sh`, `report16.py`, `local_cmp.py`, `cmp_report.py`. Machine B: `ssh machine-B 'cd ~/robot_arm_control && tmux new -d -s s16a "bash experiments/spike16a/run_queue.sh 2>&1 | tee -a ~/logs/s16a.log"'`. Result JSONs in `results/spike16a/<task>/`; the PPO pickles (selected policy weights) stayed on the second machine (`~/robot_arm_control/results/spike16a/<task>/ppo_*.pkl`).

