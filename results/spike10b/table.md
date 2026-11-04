# spike10b results (whip, realistic conditions). best-classical = T-adapt3_full (chosen on holdout 1100-1399, n=300)

Primary endpoint set: TEST-B seeds 20000-20499 (n=500, one locked run). Secondary: TEST seeds 0-49 (n=50, one locked run). Multi-seed methods: per-test-seed mean over the 3 training seeds, statistics over test seeds. Paired percentile bootstrap, 20000 resamples.

## Main table, TEST-B (n=500)

| method | success % [95% CI] | closest approach (m) | hit speed (m/s) | energy (Nm^2/step) | residual norm (Nm) | unstable-episode % (plant blow-up) | diverged % |
|---|---|---|---|---|---|---|---|
| T-nom | 53.6 [49.2, 58.0] | 0.1182 | 8.97 | 462.6 | - | 0.00 | 0.00 |
| T-robust | 51.6 [47.2, 56.0] | 0.1185 | 10.14 | 461.8 | - | 0.00 | 0.00 |
| T-adapt | 54.8 [50.4, 59.2] | 0.1153 | 10.38 | 1276.4 | - | 0.00 | 0.00 |
| T-adapt2 | 54.4 [50.2, 58.8] | 0.1164 | 11.30 | 795.7 | - | 0.00 | 0.00 |
| T-adapt3_grad | 51.0 [46.6, 55.4] | 0.1348 | 11.41 | 592.2 | - | 0.00 | 0.00 |
| T-adapt3_full (= best-classical) | 65.6 [61.4, 69.8] | 0.0945 | 10.96 | 505.2 | - | 0.00 | 0.00 |
| T-adapt3_ood | 59.2 [54.8, 63.4] | 0.1172 | 10.75 | 1825.2 | - | 0.00 | 0.00 |
| T-oracle | 89.2 [86.4, 91.8] | 0.0733 | 9.99 | 493.9 | - | 0.00 | 0.00 |
| L-est-full | 66.5 [64.0, 69.1] | 0.0960 | 10.92 | 491.3 | - | 0.00 | 0.00 |
| L-est-eff | 64.4 [61.8, 67.1] | 0.1000 | 11.45 | 507.6 | - | 0.00 | 0.00 |
| L-est-OOD | 65.1 [62.5, 67.7] | 0.1005 | 11.19 | 508.5 | - | 0.00 | 0.00 |
| R-PPO-hist | 65.6 [61.4, 69.8] | 0.0967 | 10.99 | 503.9 | 0.622 | 0.00 | 0.00 |

## Same table, TEST (n=50)

| method | success % [95% CI] | closest approach (m) | hit speed (m/s) | energy (Nm^2/step) | residual norm (Nm) | unstable-episode % (plant blow-up) | diverged % |
|---|---|---|---|---|---|---|---|
| T-nom | 52.0 [38.0, 66.0] | 0.1077 | 9.35 | 456.7 | - | 0.00 | 0.00 |
| T-robust | 48.0 [34.0, 62.0] | 0.1173 | 11.20 | 580.2 | - | 0.00 | 0.00 |
| T-adapt | 50.0 [36.0, 64.0] | 0.1303 | 12.08 | 1223.8 | - | 0.00 | 0.00 |
| T-adapt2 | 44.0 [30.0, 58.0] | 0.1403 | 9.70 | 728.7 | - | 0.00 | 0.00 |
| T-adapt3_grad | 52.0 [38.0, 66.0] | 0.1289 | 10.67 | 581.7 | - | 0.00 | 0.00 |
| T-adapt3_full (= best-classical) | 68.0 [54.0, 80.0] | 0.0851 | 9.79 | 477.0 | - | 0.00 | 0.00 |
| T-adapt3_ood | 60.0 [46.0, 74.0] | 0.1050 | 9.94 | 1852.9 | - | 0.00 | 0.00 |
| T-oracle | 86.0 [76.0, 94.0] | 0.0732 | 9.33 | 454.2 | - | 0.00 | 0.00 |
| L-est-full | 65.3 [58.0, 72.7] | 0.1001 | 12.14 | 439.6 | - | 0.00 | 0.00 |
| L-est-eff | 61.3 [52.7, 70.0] | 0.1050 | 10.84 | 449.5 | - | 0.00 | 0.00 |
| L-est-OOD | 61.3 [52.7, 70.0] | 0.1212 | 10.22 | 445.6 | - | 0.00 | 0.00 |
| R-PPO-hist | 70.0 [56.0, 82.0] | 0.0839 | 11.15 | 472.7 | 0.577 | 0.00 | 0.00 |

## Per training seed (success %, TEST-B | TEST)

| method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| L-est-full | 68.8 / 72.0 | 62.4 / 68.0 | 68.4 / 56.0 |
| L-est-eff | 62.6 / 68.0 | 66.0 / 62.0 | 64.6 / 54.0 |
| L-est-OOD | 65.0 / 58.0 | 66.2 / 66.0 | 64.2 / 60.0 |
| R-PPO-hist | 65.6 / 70.0 |
## Paired differences vs best-classical

### TEST-B (n=500)

| method | diff vs best-classical (pp) | 95% CI | p (boot) |
|---|---|---|---|
| T-nom | -12.0 | [-18.0, -6.2] | 0.0001 |
| T-robust | -14.0 | [-20.0, -8.0] | 0.0001 |
| T-adapt | -10.8 | [-16.6, -5.0] | 0.0002 |
| T-adapt2 | -11.2 | [-16.8, -5.4] | 0.0004 |
| T-adapt3_grad | -14.6 | [-20.6, -8.6] | 0.0001 |
| T-adapt3_ood | -6.4 | [-12.2, -0.6] | 0.0333 |
| T-oracle | 23.6 | [18.8, 28.4] | 0.0001 |
| L-est-full | 0.9 | [-3.5, 5.4] | 0.6871 |
| L-est-eff | -1.2 | [-5.8, 3.5] | 0.6000 |
| L-est-OOD | -0.5 | [-5.1, 4.1] | 0.8444 |
| R-PPO-hist | 0.0 | [-4.0, 4.0] | 1.0000 |

### TEST (n=50)

| method | diff vs best-classical (pp) | 95% CI | p (boot) |
|---|---|---|---|
| T-nom | -16.0 | [-34.0, 2.0] | 0.1201 |
| T-robust | -20.0 | [-36.0, -4.0] | 0.0282 |
| T-adapt | -18.0 | [-38.0, 2.0] | 0.0930 |
| T-adapt2 | -24.0 | [-42.0, -6.0] | 0.0112 |
| T-adapt3_grad | -16.0 | [-32.0, 2.0] | 0.0809 |
| T-adapt3_ood | -8.0 | [-24.0, 8.0] | 0.4084 |
| T-oracle | 18.0 | [0.0, 36.0] | 0.0523 |
| L-est-full | -2.7 | [-17.3, 12.7] | 0.6859 |
| L-est-eff | -6.7 | [-22.0, 8.7] | 0.3736 |
| L-est-OOD | -6.7 | [-20.7, 8.0] | 0.3365 |
| R-PPO-hist | 2.0 | [-8.0, 12.0] | 0.8587 |

## Verdicts (primary endpoint: TEST-B; H11 and H12 Holm-corrected together)

### TEST-B

| hypothesis | candidate | diff (pp) | 95% CI | p | Holm p | >= +10 pp and CI > 0 and Holm p < .05 |
|---|---|---|---|---|---|---|
| H11 | L-est-full | 0.9 | [-3.5, 5.4] | 0.6871 | 1.0000 | no: NOT SUPPORTED |
| H12 | R-PPO-hist | 0.0 | [-4.0, 4.0] | 1.0000 | 1.0000 | no: NOT SUPPORTED |

### TEST (n=50, reported alongside; the verdict uses TEST-B)

| hypothesis | candidate | diff (pp) | 95% CI | p | Holm p | >= +10 pp and CI > 0 and Holm p < .05 |
|---|---|---|---|---|---|---|
| H11 | L-est-full | -2.7 | [-17.3, 12.7] | 0.6859 | 1.0000 | no: NOT SUPPORTED |
| H12 | R-PPO-hist | 2.0 | [-8.0, 12.0] | 0.8587 | 1.0000 | no: NOT SUPPORTED |

**H11: NOT SUPPORTED; H12: NOT SUPPORTED (TEST-B).**

Ceiling note: best-classical TEST-B success = 65.6 %, largest conceivable gain 34.4 pp.

## Secondary comparisons on TEST-B

| comparison | diff (pp) | 95% CI | p |
|---|---|---|---|
| L-est-full - T-adapt3_grad | 15.5 | [10.5, 20.5] | 0.0001 |
| L-est-full - T-adapt3_full | 0.9 | [-3.5, 5.3] | 0.6843 |
| L-est-full - L-est-eff | 2.1 | [-1.1, 5.2] | 0.1949 |
| L-est-OOD - T-adapt3_ood | 5.9 | [1.1, 10.8] | 0.0167 |
| L-est-OOD - L-est-full | -1.4 | [-4.9, 2.1] | 0.4477 |
| T-adapt3_ood - T-adapt3_grad | 8.2 | [2.0, 14.4] | 0.0107 |
| T-oracle - T-adapt3_full | 23.6 | [18.8, 28.4] | 0.0001 |
| T-oracle - L-est-full | 22.7 | [19.0, 26.3] | 0.0001 |

## Holdout (seeds 1100-1399, n=300; selection set, so optimistic for the selected method)

| method | success % [95% CI] | unstable % |
|---|---|---|
| nom | 55.7 [50.0, 61.3] | 0.00 |
| robust | 52.0 [46.3, 57.7] | 0.00 |
| adapt | 57.3 [51.7, 63.0] | 0.00 |
| adapt2 | 57.7 [52.0, 63.0] | 0.00 |
| adapt3_grad | 52.7 [47.0, 58.3] | 0.00 |
| adapt3_full | 69.7 [64.3, 75.0] | 0.00 |
| oracle | 90.0 [86.7, 93.3] | 0.00 |
| lest_full#0 | 65.0 [59.7, 70.3] | 0.00 |
| lest_full#1 | 63.7 [58.0, 69.0] | 0.00 |
| lest_full#2 | 62.3 [56.7, 67.7] | 0.00 |
| lest_eff#0 | 56.7 [51.0, 62.3] | 0.00 |
| lest_eff#1 | 63.3 [58.0, 68.7] | 0.00 |
| lest_eff#2 | 66.3 [61.0, 71.7] | 0.00 |
| lest_ood#0 | 62.7 [57.3, 68.0] | 0.00 |
| lest_ood#1 | 70.3 [65.0, 75.3] | 0.00 |
| lest_ood#2 | 63.7 [58.0, 69.0] | 0.00 |

### Holdout (n=300): learned methods (mean over 3 training seeds per holdout seed) vs best-classical, paired

| method | success % [95% CI] | diff vs best-classical (pp) | 95% CI | p |
|---|---|---|---|---|
| L-est-full (3 seeds) | 63.7 [60.2, 67.1] | -6.0 | [-12.1, 0.2] | 0.0601 |
| L-est-full minus T-adapt3_grad | | 11.0 | [4.6, 17.6] | 0.0005 |
| L-est-eff (3 seeds) | 62.1 [58.8, 65.3] | -7.6 | [-13.6, -1.4] | 0.0186 |
| L-est-OOD (3 seeds) | 65.6 [62.3, 68.9] | -4.1 | [-10.1, 2.0] | 0.1805 |

## Dose curve (holdout seeds 1100-1299, n=200 per level; success %, 95% CI; seed-averaged for L-est)

| method | x0.5 (all three factors) | x1 (all three) | x2 (all three) |
|---|---|---|---|
| T-adapt3_full (= best-classical) | 71.0 [64.5, 77.0] | 69.5 [63.0, 76.0] | 53.0 [46.0, 60.0] |
| L-est-eff | 67.5 [63.3, 71.5] | 61.2 [57.2, 65.2] | 56.8 [52.7, 61.0] |
| L-est-full | 64.3 [60.0, 68.5] | 61.2 [56.8, 65.3] | 60.3 [56.0, 64.7] |
| T-oracle | 92.0 [88.0, 95.5] | 89.5 [85.0, 93.5] | 91.0 [87.0, 95.0] |

Paired difference to best-classical (pp, 95% CI):

| method | x0.5 (all three factors) | x1 (all three) | x2 (all three) |
|---|---|---|---|
| L-est-eff | -3.5 [-10.3, 3.5] | -8.3 [-16.0, -0.5] | 3.8 [-3.8, 11.5] |
| L-est-full | -6.7 [-13.5, 0.2] | -8.3 [-16.2, -0.3] | 7.3 [-0.0, 14.7] |
| T-oracle | 21.0 [14.0, 28.0] | 20.0 [12.5, 27.5] | 38.0 [30.5, 45.5] |

## Single-factor table (each factor alone at x1) (holdout seeds 1100-1299, n=200 per level; success %, 95% CI; seed-averaged for L-est)

| method | a only: encoder noise | b only: per-link mass + chain friction | c only: actuator noise |
|---|---|---|---|
| T-adapt3_full (= best-classical) | 67.0 [60.5, 73.5] | 65.0 [58.5, 71.5] | 64.0 [57.5, 70.5] |
| L-est-eff | 64.8 [60.5, 69.0] | 65.2 [61.0, 69.2] | 64.5 [60.3, 68.7] |
| L-est-full | 59.7 [55.5, 63.8] | 70.5 [66.8, 74.2] | 65.0 [60.5, 69.3] |
| T-oracle | 86.0 [81.0, 90.5] | 93.0 [89.5, 96.5] | 89.0 [84.5, 93.0] |

Paired difference to best-classical (pp, 95% CI):

| method | a only: encoder noise | b only: per-link mass + chain friction | c only: actuator noise |
|---|---|---|---|
| L-est-eff | -2.2 [-9.5, 5.3] | 0.2 [-7.2, 7.5] | 0.5 [-6.2, 7.2] |
| L-est-full | -7.3 [-14.3, -0.3] | 5.5 [-1.2, 12.3] | 1.0 [-6.3, 8.3] |
| T-oracle | 19.0 [11.0, 27.0] | 28.0 [20.0, 36.0] | 25.0 [17.5, 32.5] |

Interpretation rule (fixed in v7.1): if a learned method wins at x1 only, it is a single-point result, not a general conclusion.

Unstable-episode % per method over all dose cells: adapt3_full: 0.00, lest_eff#0: 0.00, lest_eff#1: 0.00, lest_eff#2: 0.00, lest_full#0: 0.00, lest_full#1: 0.00, lest_full#2: 0.00, oracle: 0.00

## OOD table (TEST-B, evaluation at x1; OOD variants trained/tuned on x0.5 and x2)

| method | train/tune distribution | TEST-B success % [CI] | TEST success % |
|---|---|---|---|
| T-adapt3_full | x1 (CAL, in-distribution) | 65.6 [61.4, 69.8] | 68.0 |
| T-adapt3_ood | x0.5 + x2 (CAL) | 59.2 [54.8, 63.4] | 60.0 |
| L-est-full | x1 (training seeds, sim labels) | 66.5 [64.0, 69.1] | 65.3 |
| L-est-eff | x1 (no per-link labels) | 64.4 [61.8, 67.1] | 61.3 |
| L-est-OOD | x0.5 + x2 (training seeds, sim labels) | 65.1 [62.5, 67.7] | 61.3 |

## T-adapt3 identifiability on CAL (default config, 50 seeds, condition abc; fraction of variance explained vs predicting the CAL mean)

| parametrisation | quantity | RMSE | prior SD | frac. variance explained |
|---|---|---|---|---|
| grad | length_scale | 0.05747 | 0.1097 | 0.726 |
| grad | log_total_mass | 0.01767 | 0.5637 | 0.999 |
| grad | mass_log_slope | 0.1219 | 0.09313 | -0.714 |
| grad | log_damping | 1.017 | 1.101 | 0.146 |
| grad | friction_1e-3Nm | 3.097 | 0.5359 | -32.398 |
| grad | per_link_mass_deviation(log, mean-removed) | 0.1699 | 0.1623 | -0.096 |
| full | length_scale | 0.03156 | 0.1097 | 0.917 |
| full | log_total_mass | 0.01224 | 0.5637 | 1.000 |
| full | mass_log_slope | 0.05826 | 0.09313 | 0.609 |
| full | log_damping | 0.7554 | 1.101 | 0.529 |
| full | friction_1e-3Nm | 1.854 | 0.5359 | -10.963 |
| full | per_link_mass_deviation(log, mean-removed) | 0.1256 | 0.1623 | 0.401 |

**tune_grad**: episodes 900, wall 1738s, chosen CAL success 0.66, cfg `{"pm": "grad", "T": 0.6, "amp": 1.0, "loss": "soft_l1", "f_scale": 1.0, "sq": 0.01, "sv": 0.1, "preg": 0.0, "g_sd": 0.15, "fr_sd": 1.0, "prefixes": [0.5, 0.75, 1.0], "ngrid": 7, "nstart": 4, "nfev": 30}`

**tune_full**: episodes 750, wall 3007s, chosen CAL success 0.8, cfg `{"pm": "full", "T": 0.6, "amp": 0.6, "loss": "linear", "f_scale": 1.0, "sq": 0.01, "sv": 0.1, "preg": 1.0, "g_sd": 0.0, "fr_sd": 1.0, "prefixes": [0.5, 0.75, 1.0], "ngrid": 7, "nstart": 4, "nfev": 30}`

**tune_ood**: episodes 350, wall 748s, chosen CAL success 0.72, cfg `{"pm": "full", "T": 0.2, "amp": 0.6, "loss": "soft_l1", "f_scale": 1.0, "sq": 0.01, "sv": 0.1, "preg": 1.0, "g_sd": 0.0, "fr_sd": 1.0, "prefixes": [0.5, 0.75, 1.0], "ngrid": 7, "nstart": 4, "nfev": 30}`

## L-est CAL errors (seed-0 model of the chosen config; RMSE in output units: ls, log m_i..., log ds, fric in 1e-3 Nm)

- full: chosen 256x256, wd 0.0001, epochs 100; CAL z-MSE 0.1188 (prior 1.036); RMSE `[0.0257, 0.0937, 0.1069, 0.1107, 0.1172, 0.1159, 0.1191, 0.1219, 0.1199, 0.1288, 0.1427, 0.493, 0.5525]`
- eff: chosen 256x256, wd 0.0001, epochs 100; CAL z-MSE 0.2938 (prior 0.975); RMSE `[0.0261, 0.0296, 0.4873, 0.5576]`
- ood: chosen 128x128, wd 0.0, epochs 100; CAL z-MSE 0.1152 (prior 1.018); RMSE `[0.032, 0.1362, 0.1608, 0.1674, 0.1865, 0.1821, 0.1992, 0.1835, 0.1894, 0.1962, 0.2054, 0.6189, 0.9527]`

## Did PPO learn? (CAL 1000-1049: selected checkpoint minus initialisation, need >= +5 pp)

| run | init CAL % | selected CAL % | diff pp | learned | residual norm at selected ckpt (Nm) |
|---|---|---|---|---|---|
| R-PPO-hist#0 | 78.0 | 82.0 (it 314) | 4.0 | NOT LEARNED | 0.609 |

### Step 0: plant blow-ups with the original integration step (SUBF=1, as spike09/10a), CAL n=50

| method|cond | unstable episodes % | flagged diverged % | success % |
|---|---|---|---|
| nom|abc | 4.0 | 0.0 | 50 |
| adapt|abc | 2.0 | 0.0 | 60 |
| oracle|abc | 2.0 | 0.0 | 88 |
| nom|none | 4.0 | 0.0 | 58 |
| adapt|none | 0.0 | 0.0 | 100 |
| oracle|x2 | 0.0 | 0.0 | 90 |

### Step 0: integration step /2 (SUBF=2; used for CAL tuning), CAL n=50

| method|cond | unstable episodes % | flagged diverged % | success % |
|---|---|---|---|
| nom|abc | 0.0 | 0.0 | 52 |
| adapt|abc | 0.0 | 0.0 | 58 |
| oracle|abc | 0.0 | 0.0 | 88 |
| nom|none | 0.0 | 0.0 | 64 |
| adapt|none | 0.0 | 0.0 | 94 |
| oracle|x2 | 0.0 | 0.0 | 86 |

### Step 0: integration step /4 (SUBF=4; used for holdout, TEST-B, TEST), CAL n=50

| method|cond | unstable episodes % | flagged diverged % | success % |
|---|---|---|---|
| nom|abc | 0.0 | 0.0 | 48 |
| adapt|abc | 0.0 | 0.0 | 62 |
| oracle|abc | 0.0 | 0.0 | 90 |
| nom|none | 0.0 | 0.0 | 62 |
| adapt|none | 0.0 | 0.0 | 90 |
| oracle|x2 | 0.0 | 0.0 | 90 |

## Ceiling of the probe -> regress -> replay -> plan design (CAL 1000-1049, abc, n=50)

| estimator | success % [95% CI] |
|---|---|
| cheat-theta (true theta, probe+replay pipeline) | 74.0 [62.0, 86.0] |
| T-oracle | 90.0 [82.0, 98.0] |
| T-adapt3-full | 78.0 [66.0, 88.0] |
| L-est-full#0 | 56.0 [42.0, 70.0] |

## T-adapt3 identifiability with the TUNED configs (CAL, 50 seeds, abc)

| parametrisation | quantity | RMSE | prior SD | frac. variance explained |
|---|---|---|---|---|
| grad (T=0.6) | length_scale | 0.03444 | 0.1097 | 0.901 |
| grad (T=0.6) | log_total_mass | 0.02252 | 0.5637 | 0.998 |
| grad (T=0.6) | mass_log_slope | 0.09984 | 0.09313 | -0.149 |
| grad (T=0.6) | log_damping | 0.5454 | 1.101 | 0.755 |
| grad (T=0.6) | friction_1e-3Nm | 2.105 | 0.5359 | -14.425 |
| grad (T=0.6) | per_link_mass_deviation(log, mean-removed) | 0.1639 | 0.1623 | -0.020 |
| full (T=0.6) | length_scale | 0.02093 | 0.1097 | 0.964 |
| full (T=0.6) | log_total_mass | 0.01133 | 0.5637 | 1.000 |
| full (T=0.6) | mass_log_slope | 0.04738 | 0.09313 | 0.741 |
| full (T=0.6) | log_damping | 0.6395 | 1.101 | 0.663 |
| full (T=0.6) | friction_1e-3Nm | 0.5501 | 0.5359 | -0.053 |
| full (T=0.6) | per_link_mass_deviation(log, mean-removed) | 0.1187 | 0.1623 | 0.465 |

## Budget
```
{
 "classical": {
  "T-robust episodes (spike09a)": 9050,
  "T-adapt design episodes (spike09b, estimate)": 1300,
  "T-adapt2 tuning episodes (spike10a)": 600,
  "T-adapt2 wall_s": 415.55105805397034,
  "T-adapt3-grad tuning episodes (CAL, incl. 2 reference evals of T-adapt/T-adapt2)": 900,
  "T-adapt3-grad tuning wall_s": 1737.9922761917114,
  "T-adapt3-full tuning episodes (CAL, incl. 2 reference evals of T-adapt/T-adapt2)": 750,
  "T-adapt3-full tuning wall_s": 3006.6451189517975,
  "T-adapt3-ood tuning episodes (CAL, incl. 2 reference evals of T-adapt/T-adapt2)": 350,
  "T-adapt3-ood tuning wall_s": 747.5069055557251,
  "total for the 2x rule (T-robust + T-adapt2 + T-adapt3 grad+full)": 11300,
  "2x": 22600
 },
 "L-est": {
  "probe segment T (s)": 0.6,
  "training probe episodes (full+eff share one set; ood its own)": 24000,
  "full-episode equivalents (x T/2.5)": 5760.0,
  "CAL probe episodes": 400,
  "wall_s (data + sweep + 3 seeds x 3 variants)": 644.5981554985046,
  "label source": "simulator ground truth (full plant theta: length scale, 10 per-link masses, damping, Coulomb friction) for every training episode; classical methods use no labels"
 },
 "R-PPO-hist": {
  "R-PPO-hist main recipe (adopted) seed 0": {
   "train_episodes": 8204,
   "cal_eval_episodes": 400,
   "total_episodes": 8604,
   "steps": 2998272,
   "wall_s": 1358.7421414852142,
   "init_cal": 0.78,
   "selected_cal": 0.82,
   "learned": false,
   "res_mean_selected": 0.6091352055690212
  },
  "R-PPO-hist diagnostic round (4x batch, not adopted) seed 0": {
   "train_episodes": 8094,
   "cal_eval_episodes": 400,
   "total_episodes": 8494,
   "steps": 2981888,
   "wall_s": 668.9929718971252,
   "init_cal": 0.78,
   "selected_cal": 0.82,
   "learned": false,
   "res_mean_selected": 0.22988465365909302
  },
  "pool cache-fill (zero residual, 300 scenarios)": {
   "base": "('adapt3', {'pm': 'full', 'T': 0.6, 'amp': 0.6, 'loss': 'linear', 'f_scale': 1.0",
   "M": 300,
   "succ": 0.66,
   "mean_steps": 364.6533333333333,
   "unstable": 0.0,
   "wall": 926.7437777519226
  }
 },
 "label sources": {
  "classical (T-nom/robust/adapt/adapt2/adapt3)": "none: only measured arm q, qd and applied probe torques; tuned on CAL success",
  "L-est-*": "simulator ground-truth plant theta of every training episode (13 numbers for full/ood, 4 for eff)",
  "R-PPO-hist": "reward uses privileged chain-tip distance (simulator); observations are arm-only history",
  "T-oracle": "true plant given directly (reference)"
 }
}
```