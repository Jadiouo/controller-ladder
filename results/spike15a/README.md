# spike15a: v8 locked confirmation of K3 on TEST-C (seeds 30000-30499, n=500)

Rules: `docs/03_hypotheses.md` sections v8 and v8.1 (applied verbatim, not edited). Code: `spike15a/` (eval15a.py, stats15.py, retrain_dec4.py, infer_dec4.py, compare_retrain.py). Full tables: `table.md`. Per-episode results: `testc_episodes.json`. Retrain check: `retrain_check.json`.
Setting as spike13b/14a: whip, conditions abc x1, plant step /4, plant of each seed seed-derived. All methods are the frozen spike13b / spike14a / spike10b implementations (imported, nothing retuned).

## 1. Verdicts (paired bootstrap over episodes, 20000 resamples, one shared resample index)

| hypothesis | statistic | CI | card rule | verdict | Holm-adjusted p (one-sided, family alpha 0.05) |
|---|---|---|---|---|---|
| H13 estimation dominates headroom | R1 - R0 = **+17.6 pp** | 95% [+12.6, +22.6] | point >= +10 and lower > 0 | **supported** | 0.00015 -> survives |
| H13b probe cost small | R4 - R2 = **+4.4 pp** | 95% [+0.8, +7.8] | upper < +5 | **not supported** (upper 7.8) | 0.390 -> does not survive |
| H14 estimator duel equivalent (retrained L-dec4, see 3) | L-dec4 - R0 = **+6.1 pp** | 90% [+2.2, +9.9] | inside +-10 (TOST) | supported before correction (upper end 9.9, 0.1 pp inside the margin) | 0.094 -> **does not survive Holm** |

Holm p-values: one-sided bootstrap p (H13: P*(diff <= 0); H13b: P*(diff >= 5); H14: max of the two TOST one-sided p), step-down over the family of three, p floor 1/20001. H14 raw p = 0.047, adjusted 0.094. Net result under v8.1: **H13 confirmed; H13b not confirmed; H14 not confirmed after Holm** (the unadjusted TOST passes only barely). The H14 point estimate is also not "no difference": the 90% CI excludes 0 on the positive side (L-dec4 is +6.1 pp, i.e. a small real gain over R0, within the +-10 pp margin). (measured)

## 2. Success rate on TEST-C (95% CI) and recovered fraction (X - R0)/(R1 - R0) with CI

| method | success % | vs R0 (pp) | recovered fraction |
|---|---|---|---|
| R0 T-adapt3-full | 65.6 [61.4, 69.6] | - | - |
| R1 true theta | 83.2 [79.8, 86.4] | +17.6 [+12.6, +22.6] | 1 |
| R2 + true chain state | 88.6 [85.8, 91.4] | | 1.31 [1.08, 1.65] |
| R4 T-oracle | 93.0 [90.6, 95.2] | +27.4 [+22.6, +32.2] | 1.56 [1.29, 1.99] |
| R5 canonical T-oracle | 94.4 [92.2, 96.4] | +28.8 [+24.2, +33.4] | |
| L-est-full (mean of 3 models) | 67.4 [64.9, 69.9] | +1.8 [-2.6, +6.3] | 0.10 [-0.18, 0.31] |
| L-dec4 (3 seeds merged; seeds 70.4 / 71.2 / 73.4) | 71.7 [69.2, 74.1] | +6.1 [+1.5, +10.7] | 0.34 [0.10, 0.53] |
| L-ens (mean theta of the 3 L-dec4) | 73.6 [69.8, 77.4] | +8.0 [+2.6, +13.6] | 0.45 [0.17, 0.71] |
| C-MAP (MAP + multi-start) | 63.2 [59.0, 67.4] | -2.4 [-8.0, +3.2] | -0.14 [-0.56, 0.16] |

L-ens and C-MAP are descriptive only (no verdict). Other paired differences: R2 - R1 +5.4 [+1.6, +9.4]; R5 - R4 +1.4 [-1.6, +4.4]; L-dec4 - L-est-full +4.3 [+1.1, +7.4]; L-dec4 - C-MAP +8.5 [+3.9, +13.1]; L-ens - C-MAP +10.4 [+4.8, +16.2]. All in `table.md`. No episode had a numerical blow-up.

## 3. Re-trained L-dec4 (deviation, v8.1)
spike14a did not save the L-dec4 weights (only predictions on cal/hold/new). They were re-trained with the identical recipe and seeds 0/1/2 (`retrain_dec4.py`: dec4, H=192, lr 2e-3, alpha 0.9, 38 epochs, checkpoint = lowest CAL loss; training data and CAL labels only, nn_train.py unmodified, run as GPU queue job #300).
Acceptance check on holdout 1100-1599 only (`compare_retrain.py`, `retrain_check.json`): the re-trained predictions are **bit-identical** to the stored spike14a predictions (theta RMS difference 0.0 for all three seeds), per-seed holdout success 71.2 / 70.4 / 69.4 % = stored 71.2 / 70.4 / 69.4 %; merged 70.33 % vs 70.33 % (difference 0.0 pp, limit 2 pp); episode agreement: merged 3-seed majority-vote 100 % (primary, limit 90 %), per-seed 100 / 100 / 100 %, exact equality of the 3-seed mean 100 %. **Accepted: same frozen model** (so H14 is not labelled "retrained model"; training was deterministic here). Agreement definition: v8.1 does not say which; the primary is the majority vote of the 3 seeds, all variants are reported and all equal 100 %.

## 4. Consistency with the holdout results (spike13b / spike14a, n=500)
| quantity | holdout | TEST-C |
|---|---|---|
| R0 | 69.2 | 65.6 (-3.6, CI of each ~+-4) |
| R1 | 86.0 | 83.2 |
| R2 | 89.0 | 88.6 |
| R4 / R5 | 90.8 / 92.0 | 93.0 / 94.4 |
| R1 - R0 | +16.8 [+12.0, +21.6] | +17.6 [+12.6, +22.6] |
| R4 - R2 (spike13b R2->R4 total) | +1.8 [-1.6, +5.4] | +4.4 [+0.8, +7.8] |
| L-dec4 - R0 | +1.1 [-3.3, +5.7] | +6.1 [+1.5, +10.7] |
| L-ens - R0 | +4.8 [-0.8, +10.4] | +8.0 [+2.6, +13.6] |
| C-MAP - R0 | -2.0 [-7.6, +3.6] | -2.4 [-8.0, +3.2] |
| L-est-full - R0 | -4.5 [-9.5, +0.4] (spike13b) | +1.8 [-2.6, +6.3] |
The ladder ends and the parameter block (H13) reproduce. The probe-cost block is larger than on the holdout (+4.4 vs +1.8, CIs overlap); the pre-registered bound (<5 upper) fails on TEST-C. The learned-estimator gains are larger than on the holdout (L-dec4 +6.1 vs +1.1, partly because R0 is lower here, 65.6 vs 69.2, while L-dec4 is about the same, 71.7 vs 70.3); the holdout and TEST-C CIs overlap. (inference) The holdout-selected estimators are not significantly better or worse than the holdout suggested, but the L-dec4 gain over R0 is now distinguishable from 0; whether it is a real small effect or sampling variation of R0 on this seed set cannot be separated with n=500.

## 5. Reverse-order ladder (descriptive only, not judged)
Chain state first: fitted theta + true chain state (Rrev1) 70.4 [66.4, 74.4] = **+4.8 [-0.8, +10.6]** over R0; then true theta: R2 - Rrev1 = **+18.2 [+13.4, +23.0]**. Forward order: parameters first +17.6 [+12.6, +22.6], then chain state +5.4 [+1.6, +9.4]. In both orders the parameter step is about +18 pp and the chain-state step is about +5 pp: the attribution does not depend on the order within the resolution of the CIs. (measured; Rrev1 is a new rung implemented in eval15a.py (RevProbe), not previously run.)

## 6. Procedure, access log, deviations
* Smoke test on CAL 1000-1009 before the locked run (no TEST-C): R0, R1, L-dec4 per seed, L-ens, C-MAP reproduced the stored spike14a CAL episode results 10/10 (C-MAP re-fitted), R2/R4/R5/L-est/Rrev1 ran. At smoke time L-dec4 used the stored predictions (weights did not exist yet); the inference path with re-trained weights was checked on holdout (section 3), not on CAL.
* TEST-C run once: `results/.test_access.log` has exactly one new line (110 lines now, `2026-11-02T13:04:07  spike15a/eval15a.py  v8 single locked TEST-C run, seeds=30000-30499`). No crash, no re-run. TEST (0-49) and TEST-B (20000-20499) untouched (guard in eval15a.py). The retrain acceptance check used holdout only.
* Run under flock + systemd-run (MemoryMax 10G), 8 workers, wall 7118 s (C-MAP fits 1794 s, R0 with fresh T-adapt3 fits 1718 s; the other methods 4-6 min each).
* Deviations: (1) L-dec4 re-trained (3, v8.1, accepted). (2) Holm threshold per v8.1: alpha 0.05 on one-sided bootstrap p (my initial 0.025 reading was replaced). (3) R0 for all contrasts = the spike13b pipeline run on TEST-C (T-adapt3-full via run10); the L-dec4 / L-ens / C-MAP episodes use spike14a's Lookup pipeline on the same probe data (reproduces R0 exactly on CAL/holdout). (4) L-est-full uses the spike13b path (run10), not spike14a's float32 lookup. (5) R3 was not run (not needed by any v8 judgement); H13b uses R4 - R2 as written in the card.
* Limits: one plant family, simulation only, in-distribution learned estimators; H14's TOST passes by 0.1 pp unadjusted and fails after Holm, so it should be reported as not confirmed under v8.1.
