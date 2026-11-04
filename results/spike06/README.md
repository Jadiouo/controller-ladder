# spike06: H8b (wide, non-saturating unknown error) on ball juggling, cap N=300

Numbers: `table.md`. Clip: `juggle_h8b.mp4` (best-classical T-robust left, R-on-best-hist seed 0 right, test episode 42: e=0.63, ball x7.2, delay 322 ms, first 40 s). Selection rule = spike05's (episode whose seed-median hit difference is closest to the median difference). Here almost all non-zero differences are of the same sign (classical reaches 300, RL fails), so the rule picked an episode where RL lost badly (classical 300 vs RL seed-median 6). It is therefore not the typical outcome in terms of the success-rate gap (-3 pp overall); I did not watch it frame by frame.

## Verdict
**H8b not supported (refuted).** Test success (reached 300 hits), 50 locked episodes: best-classical T-robust 82% [70,92]; R-on-best-hist 79% [67,90] (per-seed 78/80/78); R-on-best (no history) 81% [71,91] (84/80/80); T-adapt 70% [58,82]; T-nom 0%. Rule: R-on-best-hist >= best-classical + 10 pp with paired-bootstrap CI excluding 0. Observed -3.3 pp [-8.7,+0.0]. The conclusion that H8b allows ("under equal tuning budget, classical robust/adaptive control does not lose to residual RL on juggling") holds. Inference, not measured: the RL here mostly learned nothing useful on top of T-robust (see below), so this says "this PPO residual recipe adds nothing", not "no learner could".

## Pipeline and what was done
1. Distribution sweep (CAL seeds 1000-1049, spike05 params, no retuning; `sweep_dist.jsonl`; e range lower bound 0.32 needs an extended zeta map, `experiments/spike06/zeta_map6.py`, e down to ~0.31):

| cand. | e | ball mass x | delay ms | T-robust(s05) CAL success | T-adapt(s05) |
|---|---|---|---|---|---|
| A | .32-.85 | 1-8 | 0-200 | 88% | 100% |
| B | .32-.85 | 1-12 | 0-300 | 58% | 74% |
| **C (chosen)** | .32-.85 | 1-16 | 0-400 | 40% | **50%** |
| D | .32-.85 | 1-20 | 0-500 | 22% | 34% |
Rule: better of the two in 20-60% -> B (74%) fails, C (50%) is the mildest that passes, D passes too. Choice of C is mine (minimal widening); fixed before any RL or test run.
2. Classical retune (CMA-ES, CAL only, fitness = mean min(hits,300)/300 - 0.1*apex err, pop 10, start + 8 generations = 81 evals each, warm-started from the spike05 params of the same method, same search spaces as spike05). After retuning, CAL success is higher than the 20-60% target (in-sample, the CMA objective): T-robust 76%, T-adapt 64% -> **best-classical = T-robust** (CAL). Test: T-robust 82%, T-adapt 70%; retuned vs spike05 params on test: 82 vs 46 and 70 vs 60. So the task is not saturated (82% < 100%) but the retuned classical is above the 20-60% band; the band was specified for the pre-retune check.
3. RL: residual PPO (12 Nm cap, asserted: max observed residual 0.9 Nm) stacked on T-robust. Recipe: log-std init -3.5 (final: -4.5, from the sweep), floor -5 (clamped after each optimizer step), lr 1e-4 (final hist: 3e-5), 4 epochs, rest as spike05/diag (gamma .995, lambda .95, clip .2, 32 envs x 256, minibatch 2048, tanh 256x256, reward +1/hit -3|apex err| -3 fail -2e-6|res|^2, hist = 16 lags x 80 ms x 11 features). Training seeds >= 500000, plant error from seed.
   - Init check (before any training, CAL, T-robust 76%): dr mean 76% / sample 74%; hist mean 74% / sample 74% (sample drop <= 2 pp, rule <= 5 pp). Done at log-std -3.5; log-std -4.5 has less noise, not re-run.
   - Sweep (CAL only, 1 seed=100, 3M steps each, grid lr {3e-5,1e-4} x logstd0 {-3.5,-4.5}, both variants = 8 runs). Round 1 used the last iterate: all 8 were far below the classical (hist 36-58%, dr 18-34% CAL; `sweep_rl_round1/`). PPO training damages the policy. Round 2 (a **deviation**, added after seeing round 1) keeps the best of 7 checkpoints per run, scored on CAL (mean action; the untrained init is not a candidate). Selected: hist lr 3e-5, logstd0 -4.5 (CAL 76%); dr lr 1e-4, logstd0 -4.5 (CAL 76%). Round-2 curves (`sweep2_*.log`) show collapse to 0-30% CAL success at later checkpoints for most configs, and the selected checkpoints are mostly the earliest ones (iteration 60-180, 0.5-1.5M steps), i.e. close to the init.
   - Final: 3 seeds (0,1,2) x 2 variants x 3M steps, same checkpoint rule. Selected CAL success 72-76%. Selection on CAL gives the RL a CAL-selection advantage that classical also had (CMA best-of-81 on the same CAL).
4. Test (seeds 0-49, evaluated once, after everything was frozen, `eval_all.log`): plant errors verified identical across all methods (asserted). Paired bootstrap 5000 resamples over episodes; RL episode success = mean over 3 seeds.

## Results summary (see table.md for all metrics, CIs, subgroups)
| method | success | mean hits | apex err m | torque energy Nm^2 |
|---|---|---|---|---|
| T-nom | 0% | 2.3 | .489 | 216 |
| T-robust (spike05 params) | 46% | 145.8 | .280 | 789 |
| T-adapt (spike05 params) | 60% | 183.4 | .228 | 898 |
| T-robust (retuned) = best | 82% [70,92] | 246.9 | .384 | 2113 |
| T-adapt (retuned) | 70% [58,82] | 210.6 | .237 | 1418 |
| R-on-best | 81% [71,91] | 246.2 | .382 | 2109 |
| R-on-best-hist | 79% [67,90] | 237.6 | .385 | 2123 |
R-on-best-hist vs best: -3.3 pp [-8.7,+0.0], mean hits -9.3 [-25.1,+1.5]. R-on-best vs best: -0.7 pp [-3.3,+1.3]. Both RL variants beat T-adapt (+9 / +11 pp, CI excludes 0) only because they inherit T-robust. Subgroups (e, delay, mass each split at range midpoint): RL equals or is below best-classical in every cell (-6 to 0 pp, all CIs touch 0); all errors that hurt (delay >= 200 ms, ball mass >= 8.5x) hurt every method (T-robust 67% / 65%). Delay < 200 ms: 100% for T-robust and both RL.

## Three-axis budget
| | sim steps (control steps) | evaluations / updates | wall clock | free parameters |
|---|---|---|---|---|
| T-robust CMA | 57.06M | 81 evals x 50 CAL episodes | 3397 s (1 worker, concurrent with T-adapt, loaded machine) | 11 |
| T-adapt CMA | 61.18M | 81 evals | 3678 s | 16 |
| RL sweep round 1 | 24.0M | 8 runs | 1654 s train | |
| RL sweep round 2 | 24.0M train + 26.6M CAL checkpoint evals | 8 runs, 7 ckpt evals each | 3376 s | |
| RL final (6 runs) | 18.0M train + 28.4M ckpt evals | 366 PPO iterations/run | 2639 s | ~120k (hist) / ~75k (dr) weights |
Classical total 118.2M steps. RL total 66.0M train + 55.0M ckpt-eval = 121.0M (1.02x classical; 0.56x counting training steps only; limit was 2x). Test evaluation of 11 methods: 356 s. Whole project wall clock ~7 h incl. waiting (machine load average 35-60 throughout; times inflated). Smoke test estimate was well under the 10 h limit, so the full sweep (8 configs) was kept.

## Red-flag checks (no >40 pp gap occurred, checks done anyway)
Plant errors identical across all methods (asserted on test_raw); residual cap respected (max 0.89 Nm observed, bound 12 Nm); test seeds touched only by `eval6.py`, run once after freezing (calib asserts seeds >= 1000, training asserts seeds >= 500000); reward only uses simulator hits/apex error, nothing from the classical; retuned classical shows no overfit to CAL (test 82% vs CAL 76%).

## Re-run (repo root, existing .venv; every heavy job wrapped with flock + systemd-run MemoryMax=6G)
```
.venv/bin/python experiments/spike06/zeta_map6.py                       # (already folded into experiments/spike06/zeta_map6.json)
experiments/spike06/run_sweep.sh A 0.32 0.85 1 8 0 200                  # also B 1 12 0 300 / C 1 16 0 400 / D 1 20 0 500 ; then write results/spike06/dist.json (C)
experiments/spike06/run_calib6.sh                                       # CMA T-robust + T-adapt -> robust.json adapt.json
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python experiments/spike06/select_best.py   # best.json, calib_final.json
# init check: rl6.py --variant dr|hist --classical results/spike06/best.json --check 1 --out results/spike06/init_check_<v>.json
experiments/spike06/run_sweep_rl.sh                                     # round 2 sweep (round 1 = same without the ckpt rule, kept in sweep_rl_round1/)
experiments/spike06/run_final_rl.sh                                     # final 3 seeds x 2 variants
(flock ... ) .venv/bin/python experiments/spike06/eval6.py              # LOCKED test set -> test_raw.pkl
.venv/bin/python experiments/spike06/report6.py                         # table.md
(flock ... ) .venv/bin/python experiments/spike06/video6.py 0           # juggle_h8b.mp4
```
Code: `env6.py` (reuses experiments/spike05/env5.py with overridable distribution and extended zeta map), `sweep_dist.py`, `calib6.py` (reuses experiments/spike05/cma.py and calib_classical.py spaces), `rl6.py`, `select_best.py`, `eval6.py`, `report6.py`, `video6.py`.

## Deviations, assumptions, limitations
1. Checkpoint selection on CAL added after round 1 (all last-iterate RL policies were far below the classical). Not in the pre-registered recipe; it makes RL look better than a plain last-iterate, so the refutation is not caused by it. Without it RL would be much worse (CAL 18-58% vs 76%).
2. Retuned classical CAL success (76%/64%) exceeds the 20-60% band (band applied to the pre-retune spike05 params as H8b step 1 reads; ambiguous, I chose this reading). Test success 82% leaves room for improvement (18 pp) but the headroom is small compared with the +10 pp requirement only if the true gap were ~0.
3. Distribution choice C is judged only on spike05 parameters; "mass x16" and "delay 400 ms" are extreme (delay of 0-400 ms vs a ~1 s ball flight), a design choice; conclusions apply to this distribution only. Plant e below ~0.4 uses a zeta map extrapolation (one new drop-test point, e~0.31).
4. T-robust's kp hit its search upper bound (2500) and pred ~27/30; the classical search space might be limiting (same spaces as spike05, not widened). T-adapt got no extra advantage; its two estimators were not re-designed. Classical budgets equal (81 evals each); CMA warm-started from spike05 params; T-robust/T-adapt compared by CAL success only.
5. RL: 1 seed per sweep config (50-episode CAL scores are noisy: +/- ~6 pp), selection among 8 configs and 7 checkpoints inflates CAL scores; PPO instability (collapse to 0% CAL success at later checkpoints) is not diagnosed (my guesses: Adam step size independent of gradient SNR with a noisy return signal across very different plants; time-limit cap treated as terminal; neither tested). Larger batches, a KL penalty, longer horizon, value bootstrapping at the cap, or behaviour-cloning regularisation were not tried. So "RL can't help" is not established; "this recipe doesn't" is.
6. Final RL configs were chosen per variant (hist lr 3e-5, dr lr 1e-4), both logstd0 -4.5; the init check was run at -3.5 only.
7. RL residual remained tiny (max 0.9 Nm of 12 Nm), so R-on-best ~ T-robust by construction; small differences (-3 pp) are within bootstrap noise (CI upper bound 0.0).
8. The video episode is a failure case for RL (see top), by the symmetric rule; do not read it as typical.
9. Process: no review sub-agent loop was run (single session, user absent). Machine load made all wall-clock numbers inflated; used at most 2 CMA workers (1 per method) and one heavy job at a time.
