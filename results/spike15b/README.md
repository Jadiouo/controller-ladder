# spike15b: make the learned-side results of S5, S6 and S3 interpretable (review_r1 M4, M5)

Statements marked **(inference)** are interpretation, not measurement. Section 0 was written BEFORE any spike15b run produced a number and is not edited afterwards. Sections 1+ are filled in after the runs.

## 0. Pre-registered rules (frozen before any result)

**Why.** S5 (juggling, unknown wide error; spike06/07) and S6 (whip, idealised; spike09b) used residual PPO with log-std0 = -3.5 (0.36 Nm noise). spike13a showed that this exploration scale cannot learn even a known 6 Nm correction (gate: -3.0 pp [-8.5, +2.5]); log-std0 -2.0 / -1.5 can. S3 (learn_demo) was produced before the tracking-reference wrap fix (commit "tracking reference ... unwrap", 2026-10-30) and its hyper-parameters were chosen after one look at TEST.

**Hard constraints.** TEST (0-49), TEST-B (20000-20499), TEST-C (30000-30499) are never evaluated. `results/.test_access.log` baseline at start: 109 lines, md5 d55d921c676b404a60e70b4083df1ad9 (spike15a may add its own lines; none of mine). Selection (checkpoint, log-std0, S3 hyper-parameters) uses CAL only.

**Seed sets.**
* CAL-S5 = 1000-1049 + 2000-2049 (n=100; 2000-2049 is new for the juggling task; widened from n=50 because selection on n=50 has SE ~6 pp; disclosed deviation from the official CAL 1000-1049).
* CAL-S6 = 1000-1049 + 2000-2149 (n=200, as spike13a; for S6 the zero-residual init is ~100 % on CAL).
* Gate evaluation (S5, S6) = 702000-702499 (n=500, new). Real-condition evaluation = holdout 1100-1599 (n=500) and fresh 702500-702699 (n=200, new). Training scenarios: seeds >= 500000 (S5: sampled from 500000 + seed*1e6 + counter as spike07; S6: pool of 1200 fixed scenarios as spike09b).
* S3: CAL = 1000-1049 (official), holdout 1100-1599, fresh 702500-702699, data seeds = TRAIN_SEEDS[:200] (10000-10199), as learn_demo.

**G1 sanity gate (S5, S6, each in its own environment).** The best-classical base controller is kept; after the (probe / start) a constant torque bias b*(+1,-1,+1) Nm is added to the three arm joint torques (the residual is gated like the real condition). A residual of exactly -bias restores the base behaviour, so the learnable space is known and is a constant output. b = smallest value in {1,2,3,4,6} Nm whose zero-residual success on CAL-A (1000-1049, n=50) is <= 40 %; if none, 6.
* Recipe: the ORIGINAL PPO recipe of that setting (S5: spike07 final = 128 envs x 256, minibatch 8192, lr 1e-4, c_apex 0.3, hist features, 2.5 M steps; S6: spike09b train9 = 32 envs x 256, minibatch 2048, lr 1e-4, hist, 3 M steps) with ONLY the exploration scale changed. spike13a showed that the other "fixes" (separate grad clip, probe masking) do not matter, so they are not applied (disclosed).
* Stage A: log-std0 in {-2.0, -1.5}, 1 training seed each, the setting's original step budget. The log-std0 with the higher CAL score of its CAL-selected checkpoint is "chosen" (tie -> -2.0, the smaller noise). Only the chosen one is evaluated on the gate seeds.
* PASS = chosen run's CAL-selected checkpoint beats the zero-residual init on the gate seeds 702000-702499 (bias on, paired, n=500) by >= 10 pp with the paired bootstrap 95 % CI (10000 resamples of episodes) lower bound > 0.
* Stage B (only if Stage A does not pass; one extra shot, disclosed): evaluate the other log-std0 on the gate seeds too, and train both with 2x steps; PASS if any of those 4 runs passes by the same rule (the first one, in the order chosen-1x, other-1x, chosen-2x, other-2x, that passes defines the real-condition recipe). If none passes: write "pipeline NOT validated in this environment (gate failed)", do not run the real condition, S5/S6 learner stays uninterpretable.

**C1 real condition (only after the gate passed).** Recipe = the gate-passing recipe unchanged (log-std0, steps). Original conditions of the setting (S5: distribution C of spike06 = restitution U[0.32, 0.85], ball mass x U[1, 16], delay U[0, 400] ms, base = T-robust(spike06) = best classical in results/spike07/best.json; S6: idealised whip w=0.2, base = T-adapt, hist features). 3 training seeds (0, 1, 2); the checkpoint of each is selected on CAL (best of 7 equally spaced; the init is not a candidate); evaluated on holdout 1100-1599 and fresh 702500-702699 with the SAME checkpoints. Pooled estimate = per-episode mean over the 3 seeds; paired bootstrap CI of (RL - init). Zero-residual init == best classical per episode, so "vs init" and "vs strongest classical" are the same paired difference.
* **LEARNED** = pooled holdout diff >= +5 pp AND CI lower bound > 0 AND pooled fresh diff >= +5 pp (point estimate). Else **NOT LEARNED at this budget** (never "RL cannot learn").
* Extra labels (descriptive): DEGRADES if the pooled holdout CI upper bound < 0.
* **Ceiling clause (S6).** The base is at ~97 % on the holdout (spike09b: 96.8 %), so +5 pp is arithmetically impossible (max +3.2 pp). For S6 the LEARNED verdict is therefore declared **not testable (ceiling)**; the S6 evidence is (i) the gate (can the pipeline learn in this environment at all) and (ii) the diff vs init (does a correctly explored RL at least not degrade, or does the idealised environment's hypersensitivity break it). This is stated before the numbers.

**Budget (three axes) per run.** Axis 1 episodes (training + CAL evaluation, the classical reference counts from the original spikes), axis 2 wall clock, axis 3 label source (privileged information: reward uses simulator tip / ball state in both; hist variant uses no privileged observation).

**S3 (learn_demo rerun).** Scenario tracking/clean (Lissajous, 2-DoF), students BC-MLP, MIP, Diffusion (DDIM-10), Flow (10 steps and 1 step, one checkpoint), parameter counts as before (553,942 / 553,744 / 553,744 / 553,744), observation window 3+1, chunk 8, execute 4, 15000 steps, batch 256, AdamW + cosine, EMA 0.999 (all as `scripts/demo_students.py`). Data: 200 teacher (ctc_lqr) rollouts on seeds 10000-10199 with the FIXED reference, newly collected per DART sigma. Hyper-parameter grid (selection by CAL rmse of the main variant of each kind, 1 training seed 0 per config; bc: bc, mip: mip, diffusion: DDIM-10, flow: flow-10 steps): DART sigma in {0.05, 0.1, 0.2} x tau_limit, lr in {3e-4, 1e-3} (6 configs per kind; widths, depth, steps, batch fixed). Final: selected config per kind, 3 training seeds (0, 1, 2), evaluated on holdout 1100-1599 and fresh 702500-702699, paired with the teacher on the same seeds. Reported: rmse (primary), ss_rmse, relative difference to the teacher with paired bootstrap CI. Labels: WIN = relative rmse reduction >= 10 % with CI excluding 0; MATCH = CI of the relative difference inside +-10 %; otherwise WORSE (CI above +10 %) / INCONCLUSIVE. Training goes through a GPU job queue (.venv-gpu); evaluation on CPU.

**Priority if time runs out (10 h wall total):** S6 > S5 > S3.

---

# Results (written after the runs; numbers in `table.md`, generated by `experiments/spike15b/report15.py`)

## 1. Summary (S6 and S5; S3 in section 4)

* **S6 (whip, idealised, base T-adapt).** Gate PASSED: bias b = 6 Nm (no b <= 6 reached <= 40 % on CAL-A: 46 % at b=6; the fallback b=6 was used), log-std0 -1.5 chosen (CAL 50.5 -> 68.0 vs 62.0 for -2.0), gate seeds 702000-702499: init 48.0 %, RL 63.4 %, **+15.4 pp [+10.2, +20.6]**. So the PPO/env/reward plumbing learns a known 10.4 Nm constant correction in this environment with the corrected exploration scale. Real condition (same recipe, 3 seeds x 3 M steps): **holdout 1100-1599: init 96.8 %, RL 79.2 %, -17.6 pp [-20.3, -14.9]; fresh 702500-702699: init 96.5 %, RL 77.8 %, -18.7 pp [-23.2, -14.3]**. Verdict by the pre-registered rules: LEARNED = **not testable (ceiling)** (+5 pp impossible at 96.8 %); label **DEGRADES** (CI upper bound < 0 on both sets). Every one of the 21 CAL checkpoints (3 seeds x 7) is below the zero-residual init (97.0 %): 69-85 %.
* **S5 (juggling, unknown wide error, base T-robust(spike06)).** Gate PASSED: bias b = 1 Nm (b=1 already gives 0 % vs 76 % unbiased), log-std0 -2.0 chosen (CAL 63 % vs 62 %), gate seeds: init 0.0 %, RL 63.2 %, **+63.2 pp [+58.8, +67.4]**. Real condition (log-std0 -2.0, 3 seeds x 2.5 M steps): **holdout: init 78.6 %, RL 54.8 %, -23.8 pp [-26.9, -20.7]; fresh: init 81.0 %, RL 58.0 %, -23.0 pp [-28.2, -18.2]**. Verdict: **NOT LEARNED** (and DEGRADES). All 21 CAL checkpoints (3 seeds x 7) are below the init (80 % on CAL, n=100): 24-66 %.
* **Descriptive control S6-X1** (not in any verdict): the same recipe on the weak base T-nom (60.2 % holdout, 97 % reachable by T-adapt/T-oracle, so there IS headroom): holdout **-5.0 pp [-8.7, -1.5]**, fresh -3.3 pp [-8.7, +2.0] (CAL-selected checkpoints looked like +5.0, +11.5, +5.0 pp on CAL n=200 and vanished on disjoint seeds: selection noise).
* **Reading (inference).** With the exploration scale that spike13a showed to be necessary, the S5 and S6 results change in size, not in direction: the earlier "R-PPO is 2.7 pp (S5) / 8.7 pp (S6) below best classical" becomes 23.8 / 17.6 pp below, i.e. the larger exploration noise hurts these two fragile teachers more, and the mean action drifts to harmful values (CAL success falls monotonically over training in all runs). But now the negative result is interpretable in the sense required by M4: the same pipeline demonstrably learns a known representable correction in the same environments (gate), so "not learned" is not a plumbing failure. What is still not shown: that a residual can be learned on a REAL (non-injected) error. The gate error is a constant bias; the real errors are state/plant dependent and the base is already near its achievable optimum (S6 ceiling, spike08_diag for S5).

## 2. Gate details

| | S6 | S5 |
|---|---|---|
| bias calibration (zero residual, CAL-A n=50, bias b*(1,-1,1) Nm after probe / on every step) | b=0 100 %, 1: 78 %, 2: 72 %, 3: 66 %, 4: 56 %, 6: 46 % -> b = 6 (fallback) | b=0 76 % (= spike07 CAL), 1: 0 %, 2..6: 0 % -> b = 1 |
| Stage A runs (log-std0 -2.0 / -1.5), CAL selected | 62.0 / 68.0 % (init 50.5 %) | 63.0 / 62.0 % (init 0 %, CAL n=100) |
| chosen | -1.5 | -2.0 |
| gate seeds 702000-702499 (n=500) | init 48.0 -> 63.4 %, +15.4 [+10.2, +20.6] -> PASS | init 0.0 -> 63.2 %, +63.2 [+58.8, +67.4] -> PASS |
| Stage B needed | no | no |
| residual at selected checkpoint | 4.9 Nm of 10.4 Nm needed (47 %) | 1.35 Nm of 1.73 Nm needed (78 %) |

Caveats on the gates. (i) The S5 gate is easy: with 1 Nm bias the base fails after ~3 hits, so any consistent correction yields a large gain; it tests plumbing, not sensitivity. (ii) The S6 gate recovers about half of the needed residual and +15 of +52 pp available, in line with spike13a's whip gate (+10 of +21 pp). (iii) Both gates use one training seed; the gate pass is a single pre-registered shot (no stage B).

## 3. Real conditions (see table.md for per-seed numbers)

Recipe for S6: experiments/spike09b/train9.py unchanged except log-std0 -1.5 (floor -5), 3 M steps, 32 envs x 256, minibatch 2048, lr 1e-4, hist features, cap 12 Nm, pool of 1200 training scenarios; for S5: spike07 final recipe except log-std0 -2.0 (128 envs x 256, minibatch 8192, lr 1e-4, c_apex 0.3, 2.5 M steps), distribution C of spike06, base T-robust(spike06) (CAL 76 % in spike07; 78.6 % here on the holdout).

Per-seed holdout differences: S6 -15.0, -16.8, -21.0 pp; S5 -29.2, -14.4, -27.8 pp. All CIs exclude 0 on the negative side. "Strongest classical" = the zero-residual init (T-adapt in S6, T-robust in S5); the other classical rows of the earlier spikes (S6: T-oracle 98.8 %, T-robust 56.6 %, T-nom 56.8 % on the same holdout) are not changed by this spike.

Selection note. The CAL-selected checkpoint is never the init (rule inherited from spike07/13a). Had the init been a candidate, every run would have chosen it by construction (diff 0). That is a statement about the selection rule, not about learning.

## 5. Budget (three axes)

| setting | episodes (train + CAL eval), per run | sim / control steps | wall clock | label source |
|---|---|---|---|---|
| S6 gate (2 runs) | 8.2 k + 1.6 k | 3.0 M env steps | 10-12 min each | privileged tip distance in the reward only; bias known to the experimenter |
| S6 real (3 runs) | 9.4-9.5 k + 1.6 k = ~11.1 k; 3 runs 33 k | 3.0 M each | 12-25 min each (machine load 15-35) | privileged tip distance in the reward only |
| S6-X1 (3 runs, descriptive) | ~8.4 k + 1.6 k | 3.0 M each | 11-12 min each | same |
| S5 gate (2 runs) | ~1.9 k train + 800 CAL | 2.49 M train + 6.8-7.2 M CAL-eval steps | 16-17 min each | privileged ball state in the reward; bias known |
| S5 real (3 runs) | 0.3-0.5 k train + 800 CAL | 2.49 M train + 7.9-10.2 M CAL-eval = 10.4-12.7 M each (35.6 M total) | 12-13 min each | privileged ball state in the reward only |

Classical reference (from the original spikes): S6 10,350 episodes (spike09b budget.json) -> S6 real learner total 33 k = 3.2x (gate included 52.6 k = 5.1x; X1 adds 33 k); this exceeds the v7 2x rule, declared. S5 classical reference 118.2 M control steps (spike06): the S5 real learner 35.6 M = 0.30x (gate included 56 M = 0.47x), within the 2x rule. Evaluation episodes on the holdout / fresh / gate sets are not counted (the classical side is evaluated on the same sets).

## 4. S3 (learn_demo rerun, fixed reference, CAL-only selection, TEST untouched)

Setting as pre-registered. Data recollected with the fixed reference (3 x 200 teacher rollouts on seeds 10000-10199, DART sigma 0.05 / 0.1 / 0.2; 53 s each). Grid 3 DART x 2 lr x 4 kinds, 1 training seed, selection on CAL (1000-1049) by the rmse of the main variant. **All four kinds selected the smallest DART (0.05); the lr choice is immaterial (CAL differences <= 0.2 %).** The selection is at the edge of the grid (the trend in DART is monotone: 0.05 < 0.1 < 0.2 for every kind), so a smaller DART sigma might do even better; not explored (the grid was pre-registered). Parameter counts are as before: 553,942 (bc), 553,744 (mip, diffusion, flow).

| student (holdout 1100-1599, n=500; teacher rmse 0.00994) | rmse mean over 3 training seeds | student minus teacher, relative [95 % paired bootstrap CI] | label |
|---|---|---|---|
| bc | 0.01002 | +0.8 % [+0.6, +1.0] | MATCH (slightly worse, CI excludes 0) |
| mip | 0.01002 | +0.8 % [+0.6, +0.9] | MATCH |
| diffusion (DDIM-10) | 0.01056 | +6.2 % [+5.7, +6.6] | MATCH (worse, inside 10 %) |
| flow (10 steps) | 0.01015 | +2.0 % [+1.9, +2.2] | MATCH |
| flow (1 step) | 0.00999 | +0.5 % [+0.4, +0.6] | MATCH |

Fresh seeds 702500-702699 (n=200, teacher 0.01011): bc +0.6 % [+0.3, +0.8], mip +0.7 % [+0.4, +0.9], diffusion +6.5 % [+5.9, +7.3], flow-10 +2.1 % [+1.8, +2.5], flow-1 +0.6 % [+0.3, +0.8]. No failures (0 of 500 / 200 x 3 seeds x 5 variants). Training-seed spread is tiny (sd <= 1e-5 rad). **No student beats the teacher (WIN = none); all students track it within 0.5-6.5 %.** The teacher (CTC+LQR on the clean nominal plant) is exact, so by construction a student can only match it; this setting has no headroom (same reading as the original learn_demo).

Differences to the old table (results/learn_demo/table.md, TEST seeds 0-49, old reference, DART 0.1, lr 1e-3): old students +3..+10 % and DDIM-10 2-3x worse with training-seed instability (0.012-0.028 rad); new numbers are +0.5..+6.5 % and stable. The old and new numbers are not comparable one-to-one (different seeds, reference, DART); the change in diffusion (stable now) is attributable to the reference fix and/or the smaller DART/lr, not separated here (inference). The rmse itself is slightly lower than before (0.0099 vs 0.0105 for the teacher) because of the different seed set. ss_rmse: teacher 0.00017, bc 0.00051, mip 0.00066, flow-1 0.00058, flow-10 0.00171, diffusion 0.00196: the students lose the last digits of steady-state precision, the stochastic multi-step samplers most.

Budget: data 600 teacher rollouts (3 sigmas), 24 grid + 8 final training runs of 15000 steps (24 grid runs on GPU via GPU queue job #303, 36-62 s each; the 8 final runs on CPU, 91-173 s each, see deviation 6), CAL evaluation 24 x 50 + teacher 50 episodes, label source = the teacher's torque (CTC+LQR with nominal model; no privileged information beyond the teacher itself). Classical reference: the teacher needs no training or tuning episodes.

## 6. Deviations (all disclosed)

1. CAL widened for S5 (n=100: 1000-1049 + 2000-2049) and S6 (n=200: 1000-1049 + 2000-2149); official CAL is 1000-1049. S3 uses the official CAL.
2. The PPO "fixes" of spike13a (separate gradient clips, probe masking) were not applied; only the exploration scale was changed vs the original recipes (spike13a showed they do not matter). Consequence: differences to the old S5/S6 PPO results are attributable to the exploration scale (and to the new, larger CAL sets and the new evaluation seeds), not to anything else.
3. S6 gate bias fell back to b = 6 (46 % at b=6 > 40 %), as spike13a. S5 gate b = 1 gives 0 % (the grid had no smaller step), so the S5 gate is an easy one.
4. The gate chose log-std0 by CAL separately in each environment (S6: -1.5, S5: -2.0); the real runs used the gate's choice. Neither environment was run with the other value in the real condition (no sweep on the real condition; S5/S6 real are one recipe each).
5. S6-X1 (T-nom base) was added after the S5/S6 results were known (priority S6 > S5 > S3 respected for the verdict rows); descriptive only. Its CAL init (55 %) and holdout init (60.2 %) differ because of different seed sets.
6. S3 final training (8 runs) was moved from a GPU job queue to CPU after the GPU queue (job #313, behind a 1 h+ job of another project) had waited more than 1 hour (submitted ~15:08, switched ~16:07); the job was cancelled before it ran. Results are not affected (same code, CPU is what the old checkpoints were evaluated on). The 24 grid runs ran on GPU.
7. The S5 training of the three seeds ran as three separate processes (tags real_ls-2.0_s0..2 merged into real_ls-2.0.pkl by merge15j.py).
8. Evaluation sets for S5/S6 gate (702000-702499), fresh (702500-702699) and holdout (1100-1599). The holdout 1100-1599 was used by earlier spikes (selection of best-classical in spike09b, spike08_diag); here it is used only to evaluate, but it is not virgin. The fresh seeds are.
9. The wall-clock is inflated by the shared machine (load 15-35) and by waiting for the shared simulation lock (about 2.5 h in total waiting for another agent's evaluation).
10. results/.test_access.log: 109 lines at the start, 110 now; the added line (2026-11-02T13:04:07, experiments/spike15a/eval15a.py, TEST-C) is not from this spike. This spike evaluated no TEST / TEST-B / TEST-C seed (S3 evaluation used split="cal" or adhoc seeds only; S5/S6 code asserts seeds >= 1000 and 702xxx).
11. Consequence for review M5: the pre-condition "`.test_access.log` has a new record" in the review is not met by design (this rerun never touched TEST); the replacement for "a new TEST record" is the held-out holdout / fresh evaluation reported here.

## 7. Credibility and limits

* High: (a) S5, S6 gates pass with CI excluding 0 in the same environments and the same code as the real runs; (b) the real-condition effects are large (-17.6 / -23.8 pp) with CIs far from 0 on both the holdout and the new seeds, consistent across 3 training seeds each; (c) S3: tiny training-seed spread, no failures, selection used CAL only and no TEST, both held-out sets agree.
* Medium: (a) one gate run per environment with one training seed; the S6 gate recovers only about half of the injected error; (b) S5/S6 real use one recipe (no sweep on lr/cap/horizon on the real condition; spike13a found none helped on the whip); (c) the S6 verdict on "learned" is a ceiling statement; (d) S3 hyper-parameter grid is small and the optimum is at its edge.
* Not shown: that residual RL cannot improve S5/S6 in any configuration (the evidence is: pipeline works on injected errors, fails on the real ones under CAL-tuned exploration scale, with a descriptive T-nom control that also fails). The 'degradation' is a statement about the CAL-selected checkpoint of an on-policy method with 1.6-2.7 Nm exploration noise on teachers that are noise-sensitive (inference).
* Effect on the paper (inference; for the author to decide): K2 may now state that S5 and S6 residual PPO was re-run with the gate-validated exploration scale and still does not beat (is far below) the strongest classical; S6 can only support "does not degrade" claims in the negative; S3 can be kept in Table 1 as a MATCH-not-WIN row with the new numbers (rmse +0.5..+6.5 % vs teacher, no student wins); M4 option (a) and M5 option (re-run) are satisfied for these three settings. S5/S6 remain 3-seed, single-recipe results.

## 8. Files

`experiments/spike15b/`: rl15w.py (S6 PPO + gate bias), eval15w.py, calib_bias15w.py, rl15j.py (S5), eval15j.py, calib_bias15j.py, merge15j.py, s3.py (S3 stages), report15.py (table.md). `results/spike15b/`: table.md, s6_*.json/npy, s5_*.json/npy, s6_gate_calib.json, s5_gate_calib.json, runs_s5/, runs_s6/ (pickles with all checkpoints and CAL curves), s3/ (data_d*, ckpt/, cal_eval.json, selected.json, final_holdout.json, final_fresh.json, train_summary.json). Logs and chain scripts: `.scratch/run15b/`.
