# spike05 table: H8 (unknown, per-episode random plant error), ball juggling, cap N=300, locked test set seeds 0-49

Error distribution (per episode, from seed): restitution e~U[0.4,0.85], ball mass x U[1,5], sensor delay U[0,150] ms (rounded to 5 ms steps), 4 mm position noise.
RL rows pool all (seed, episode) pairs; CI = cluster bootstrap over the 50 episodes (5000 resamples). Medians of 300 are the cap (censored).

## 1. Main table

| method | #seeds | median hits [95% CI] | mean hits | frac reaching cap 300 | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |
|---|---|---|---|---|---|---|---|
| T-nom | 1 | 8 [5, 11] | 73.3 | 0.22 | 0.408 | 0.0398 | 171 |
| T-robust | 1 | 300 [300, 300] | 300.0 | 1.00 | 0.124 | 0.0280 | 633 |
| T-adapt | 1 | 300 [300, 300] | 300.0 | 1.00 | 0.042 | 0.0162 | 582 |
| R-DR | 3 | 9 [7, 12] | 71.7 | 0.20 | 0.329 | 0.1278 | 182 |
| R-DR-hist | 3 | 3 [3, 4] | 3.7 | 0.00 | 0.366 | 0.1248 | 207 |

per-seed medians: R-DR: [8, 9, 10]; R-DR-hist: [3, 3, 4]

Stronger classical opponent on the test median (conservative choice, against H8): **T-robust**.

## 2. Paired comparisons (relative gain of median, paired cluster bootstrap over episodes)

| A vs B | median A | median B | rel. gain [95% CI] | seed-level rel. gains |
|---|---|---|---|---|
| R-DR-hist vs T-nom | 3 | 8 | -62% [-73, -40] | -62%, -62%, -50% |
| R-DR-hist vs T-robust | 3 | 300 | -99% [-99, -99] | -99%, -99%, -99% |
| R-DR-hist vs T-adapt | 3 | 300 | -99% [-99, -99] | -99%, -99%, -99% |
| R-DR-hist vs T-robust (best) | 3 | 300 | -99% [-99, -99] | -99%, -99%, -99% |
| R-DR vs T-nom | 9 | 8 | +12% [-16, +80] | +0%, +19%, +25% |
| R-DR vs T-robust | 9 | 300 | -97% [-98, -96] | -97%, -97%, -97% |
| R-DR vs T-adapt | 9 | 300 | -97% [-98, -96] | -97%, -97%, -97% |
| R-DR vs T-robust (best) | 9 | 300 | -97% [-98, -96] | -97%, -97%, -97% |

| T-adapt vs T-robust | 300 | 300 | +0% [+0, +0] | |

## 3. Subgroups (median hits [95% CI], episodes in group)

| group | n | T-nom | T-robust | T-adapt | R-DR | R-DR-hist | winner (highest median) |
|---|---|---|---|---|---|---|---|
| e low (<0.625) | 20 | 3 [3, 5] | 300 [300, 300] | 300 [300, 300] | 8 [4, 18] | 3 [3, 4] | T-robust |
| e high (>=0.625) | 30 | 12 [8, 196] | 300 [300, 300] | 300 [300, 300] | 10 [8, 14] | 3 [3, 4] | T-robust |
| delay low (<75 ms) | 23 | 10 [8, 21] | 300 [300, 300] | 300 [300, 300] | 30 [18, 115] | 4 [3, 4] | T-robust |
| delay high (>=75 ms) | 27 | 5 [3, 9] | 300 [300, 300] | 300 [300, 300] | 4 [4, 6] | 3 [3, 3] | T-robust |
| ball mass low (<3x) | 24 | 13 [6, 300] | 300 [300, 300] | 300 [300, 300] | 11 [7, 23] | 3 [3, 4] | T-robust |
| ball mass high (>=3x) | 26 | 5 [3, 9] | 300 [300, 300] | 300 [300, 300] | 8 [5, 12] | 3 [3, 4] | T-robust |
| e low & delay low | 10 | 5 [3, 12] | 300 [300, 300] | 300 [300, 300] | 26 [15, 68] | 4 [3, 4] | T-robust |
| e low & delay high | 10 | 3 [2, 4] | 300 [300, 300] | 300 [300, 300] | 4 [3, 4] | 3 [3, 3] | T-robust |
| e high & delay low | 13 | 21 [9, 300] | 300 [300, 300] | 300 [300, 300] | 128 [14, 300] | 3 [3, 4] | T-robust |
| e high & delay high | 17 | 9 [5, 300] | 300 [300, 300] | 300 [300, 300] | 6 [4, 8] | 3 [3, 4] | T-robust |

Rank correlation (Spearman) of hits with plant parameters, per method (pooled over seeds):
- T-nom: e +0.67, ball mass -0.44, delay -0.41
- T-robust: e +nan, ball mass +nan, delay +nan
- T-adapt: e +nan, ball mass +nan, delay +nan
- R-DR: e +0.31, ball mass -0.23, delay -0.79
- R-DR-hist: e +0.01, ball mass +0.25, delay -0.43

T-adapt estimator quality on test episodes: corr(final e_hat, true e) = 0.73, mean abs err 0.173; delay: corr(mean d_obs[ms], true delay) = 1.00

## 4. H8 verdict

Rule (pre-registered in docs/03_hypotheses.md v3): R-DR-hist median >= 1.3 x the better of T-robust/T-adapt, paired bootstrap CI excludes 0. Observed vs T-robust: -99% [-99, -99] -> **H8 REFUTED (falsified)**.