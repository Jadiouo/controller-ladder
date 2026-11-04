# spike07: H9 (oracle headroom) and H10 (fair RL) on ball juggling, distribution C, cap N=300

Numbers: `table.md` (all CIs, subgroups, residual norms, CAL curves). PPO-collapse diagnosis: `diag.md`. Code: `experiments/spike07/` (`env7.py`, `calib7.py`, `select_best7.py`, `rl7.py`, `diag_table.py`, `merge_final.py`, `eval7.py`, `report7.py`, `run_*.sh`). experiments/spike06/, arena/, docs/ untouched.

## Verdict (rules from docs/03_hypotheses.md v5, applied unchanged)
- **H9 NOT SUPPORTED.** R-oracle (PPO-fixed + true e, ball mass, delay in the observation) 81% [69,91] vs best-classical 82% [70,92]: **-1.3 pp [-4.0, +0.0]** (required: >= +10 pp, CI excludes 0). Per seed 82/80/80%.
- **H10 REFUTED (not supported).** R-PPO-fixed 79% [68,90]: -2.7 pp [-6.7, 0.0]; R-SAC 63% [52,75]: -18.7 pp [-27.3, -10.7]. Holm-adjusted p = 1.0 for both (one-sided, "RL better").
- best-classical = spike06 T-robust (chosen on CAL, fixed before TEST): 82% [70,92], mean hits 246.9.

**What this does and does not say.** The pre-registered reading of H9 ("not supported -> even knowing the true parameters, residual learning gives almost no gain") is **not warranted by this run**. The RL here learned almost nothing (residual L2 norm 0.13-0.20 Nm mean, max 0.5-0.7 Nm, against a 12 Nm cap and CAL success never above the initialisation; see below), so the experiment shows "this PPO/SAC residual recipe adds nothing and cannot even stay at the init without CAL selection", i.e. an optimiser failure, not evidence about headroom (inference, not measured). I did not find a recipe that learns, in spite of 19 CAL-only PPO diagnosis runs and 14 SAC runs (5 pilots, 3 aborted, 6 sweep/final). The oracle result is a weak negative: an oracle that cannot learn is not an upper bound.

## What was done
1. **Classical (step 1).** T-robust retuned with kp upper bound 10000 (was 2500), CMA-ES on CAL (1000-1049), same budget as spike06 (pop 10, start + 8 generations = 81 evals), warm-started from spike06's T-robust (`calib_robust.log`, `robust.json`). Result kp = 4350 (the 2500 bound was not what limited it). **CAL: 76% / mean hits 229.36 vs spike06 76% / 229.38: not better**, so by the rule best-classical stays spike06's T-robust (`select_best.log`, `best.json`). On TEST the new tune happens to score 86% [76,94] (vs 82%), reported in table.md but NOT used (switching after seeing TEST would be selection on TEST). 61.8M sim steps, 1121 s wall (4 workers, loaded machine).
2. **PPO diagnosis (step 2)**, `diag.md`: hypothesis written before running; 19 single-seed CAL-only PPO runs (cap bootstrap, 4x batch, target-KL, BC, apex weight, gamma/fail penalty, exploration noise). Only the 4x batch helps; **pass criterion (final iterate >= init on CAL, no dip < 50%) was not reached by any config** (best D9/E5: first 3 of 4 checkpoints at the init level 74%, final 66%/58%). Root cause (supported, not proven): PPO's gradient signal for a <= 1 Nm residual on a 76%-successful chaotic task is below the batch noise; Adam turns it into a drift of the actor mean (residual norm grows 0.15 -> 0.35 Nm in every run) that lowers success; larger batches slow it. The cap-as-terminal bias is **not** the cause (bootstrapping at the cap did not help). Fix used = 128 envs x 256 (4x batch), minibatch 8192, lr 1e-4, log-std0 -3.5, apex weight 3 -> 0.3, plus CAL checkpoint selection. This is a mitigation, not a fix (see deviations).
3. **Final runs.** R-PPO-fixed, R-oracle: 3 training seeds (0,1,2) each, 2.5M steps, 7 equally spaced CAL checkpoints, selection by (CAL success, mean hits), untrained init not a candidate. R-SAC: tanh-Gaussian actor, twin critics, fixed alpha 0.001, BC 1-3 on the squashed action, critic-only warm-up 150k steps, 1.2M steps, 7 CAL checkpoints; CAL-only sweep of 2 configs x 3 seeds (A lr 3e-5 BC 1; B lr 1e-4 BC 3); B had the higher mean CAL success (61% vs 55%) and became R-SAC. All residuals are added to best-classical, all use the 16-lag history; R-oracle additionally sees (e, mass, delay) normalised. Selected checkpoints: PPO CAL 72/76/74%, oracle 76/76/74%, SAC 56/70/58%.
4. **TEST** (seeds 0-49) evaluated exactly once after freezing (`final_methods.json`, `eval_all.log`); one line appended to `results/.test_access.log` before the run (format as existing lines). Plant errors verified identical across all 13 policies (asserted), eval7.py refuses to run a second time.

## Results summary (TEST, 50 episodes; RL = mean of 3 seeds)
| method | success | mean hits | residual L2 mean / max (Nm) |
|---|---|---|---|
| T-nom | 0% | 2.3 | - |
| T-robust spike06 = best-classical | 82% [70,92] | 246.9 | - |
| T-robust kp<=10000 (not selected) | 86% [76,94] | 259.2 | - |
| T-adapt spike06 | 70% [56,82] | 210.6 | - |
| R-PPO-fixed | 79% [68,90] | 239.2 | 0.18 / 0.72 |
| R-SAC | 63% [52,75] | 206.2 | 0.43 / 2.85 |
| R-oracle | 81% [69,91] | 242.8 | 0.15 / 0.67 |
Residual norm = L2 over the 3 joints of the added torque, mean over active steps (max over all steps). The RL residual is small compared with torques in the tens of Nm, so R-PPO/R-oracle ~ best-classical by construction; CAL training curves (table.md section 5): PPO/oracle stay at 64-76% (init 76%) with residual growing 0.14 -> 0.33 Nm; SAC oscillates 10-70% (residual 0.4-0.7 Nm).

## Three-axis budget
Sim steps (control steps; classical reference = this spike's T-robust retune 61.8M, so the 2x limit = 123.6M):
| | train | CAL ckpt-eval | other | total |
|---|---|---|---|---|
| Classical retune | 61.8M | | | 61.8M |
| PPO diagnosis (19 runs, shared by PPO and oracle) | 45.5M | 39.9M | | 85.4M |
| Aborted final attempts (OOM kills, 6 runs each; approx., logs overwritten) | ~13M per method | ~23M per method | | ~36M per method |
| R-PPO-fixed final | 7.5M | 14.7M | | 22.2M |
| R-oracle final | 7.5M | 15.2M | | 22.7M |
| R-SAC (pilots + aborted attempt + 6 sweep/final runs) | 10.7M | 16.3M | | 27.0M |
Per-method total: **R-PPO-fixed ~144M, R-oracle ~144M (2.3x), R-SAC 27M (0.44x)**. Without the ~36M lost to my two OOM crashes: 107.6M (1.7x, within the limit). **The 2x limit is therefore exceeded for PPO and oracle under the strict reading** (reference = spike07's single T-robust retune); it holds if the reference is the full cost of producing best-classical (spike06 T-robust 57.1M + spike07 retune 61.8M = 118.9M, limit 237.8M). The diagnosis (85M) is charged fully to both PPO-based methods.
Wall clock: classical 1121 s; PPO final 3 parallel processes ~550 s each; oracle same; SAC 6 processes ~2000 s each; whole spike ~6.5 h incl. diagnosis rounds, two OOM crashes (~1.5 h) and waiting; machine load average 12-35 from other sessions.
Free parameters: T-robust 11 tunable classical parameters (kp range widened); PPO actor ~118k + critic ~118k weights (+3 log-std), oracle +3 inputs; SAC actor ~119k + 2 critics ~236k; tuned RL hyperparameters: ~7 (batch/envs, lr, log-std0, apex weight, gamma, fail penalty, steps) via 19 diagnosis runs; SAC: lr, BC, alpha, warm-up (2 final configs, 5 pilots).

## Deviations from the hypothesis card (all stated before the final runs unless marked)
1. **PPO not fixed**: the v5 pass criterion (final iterate >= init, no CAL dip < 50%) was not reached; TEST was run anyway with the best available mitigation (4x batch + CAL checkpoint selection). The card says not to run TEST before it is fixed; I judged it unreachable inside the compute budget and recorded it as a finding (user instruction allowed a stated deviation). The criterion was operationalised as: final CAL success >= init - 2 pp (one episode).
2. **Reward changed from spike06**: apex-error weight 3 -> 0.3 (+1/hit, -3 fail, -2e-6|res|^2 unchanged). Reason: per-hit reward 1 - 3*0.38 < 0 for this teacher. Effect confirmed only in combination with the bigger batch (single seeds).
3. **Budget over the strict 2x** for PPO/oracle (see above), because of two OOM crashes of my own parallel launch (8 then 6 processes under the 10 GB cap); not hidden by dropping the aborted steps.
4. **SAC**: auto-alpha diverged (alpha 337) -> fixed alpha 0.001; log-std clamped to [-5,-3]; BC term and critic-only warm-up added; the first pilots S1-S5 and one aborted final attempt were confounded by a gain-0.01 actor init (~1 Nm residual from step 0), so SAC tuning was thin; the second SAC attempt zero-initialises the mean head (PPO actors keep the gain-0.01 init, residual 0.14-0.2 Nm at the first checkpoint, not re-run). SAC replay uses the pre-step observation as next-observation at cap truncation. SAC gets 1.2M env steps (UTD 0.125), PPO 2.5M. The SAC config sweep (2 configs x 3 seeds) selects on CAL only.
5. PPO and oracle diagnosis used the training seed 100/101 only and single runs (CAL noise ~ +-6 pp per run): several of the diagnostic rankings are within noise.
6. Checkpoint spacing: 7 equally spaced points, so for PPO the first checkpoint (iteration 11/22 of 76) is already chosen in 4 of 6 PPO/oracle runs, i.e. the selected policy is close to the initialisation.
7. best-classical = spike06's T-robust although the new T-robust(kp10000) is better on TEST (86 vs 82%); this follows the card's CAL-only rule and ties on CAL (76% both, mean hits 229.36 vs 229.38).

## Limitations
- Single task, single distribution, 50 TEST episodes; CIs from paired bootstrap over episodes only (seed variation shown per seed). Training seeds 3 per method.
- CAL checkpoint selection and CMA-ES both used the same 50 CAL episodes (in-sample); the small gap CAL 76% -> TEST 82% shows no overfit of the classical, but RL selection among 7 noisy CAL evaluations has some selection noise.
- The negative result is about *these* PPO/SAC recipes; a learner with a better credit-assignment design (dense failure-avoidance shaping, longer horizon with larger gamma and much larger batch, model-based) was not tried. H9 negative must not be read as "no headroom".
- The 'train success' curve in table.md is biased and uninformative at this scale (see note there).
- Machine was shared (load 12-35); wall-clock numbers are inflated.

## Re-run (repo root, existing .venv, every heavy job wrapped with flock + systemd-run MemoryMax=10G)
```
experiments/spike07/run_calib7.sh; flock ... .venv/bin/python experiments/spike07/select_best7.py     # classical -> best.json
experiments/spike07/run_diag.sh; run_diag2.sh; run_diag3.sh; run_diag4.sh; run_diag5.sh   # diagnosis (diag.md; D2/D5/D6/S* were stopped early by hand)
experiments/spike07/run_finals_seq.sh   # PPO + oracle (hold A, B; its hold C = first SAC attempt, aborted)
experiments/spike07/run_final_sac2.sh   # SAC A/B x 3 seeds
.venv/bin/python experiments/spike07/merge_final.py                                     # CAL-only R-SAC config choice, final_methods.json
flock ... .venv/bin/python experiments/spike07/eval7.py                                 # LOCKED TEST, once
.venv/bin/python experiments/spike07/report7.py                                         # table.md
```
