# spike06 table: H8b (wide, non-saturating unknown error), ball juggling, cap N=300, locked test seeds 0-49

Error distribution (dist.json): restitution e~U[0.32, 0.85], ball mass x U[1.0, 16.0], sensor delay U[0.0, 400.0] ms (5 ms steps), 4 mm noise. best-classical (chosen on CAL only) = **T-robust**.
Success = reached the 300-hit cap. RL rows: per-episode success = mean over the 3 training seeds; CIs = paired bootstrap over the 50 test episodes (5000 resamples).

## 1. Main table

| method | #seeds | success [95% CI] | mean hits | median hits | apex err (m) | torque energy (Nm^2) |
|---|---|---|---|---|---|---|
| T-nom | 1 | 0% [0, 0] | 2.3 | 2 | 0.489 | 216 |
| T-robust (spike05 params, not retuned) | 1 | 46% [32, 60] | 145.8 | 28 | 0.280 | 789 |
| T-adapt (spike05 params, not retuned) | 1 | 60% [46, 74] | 183.4 | 300 | 0.228 | 898 |
| T-robust (retuned) | 1 | 82% [70, 92] | 246.9 | 300 | 0.384 | 2113 |
| T-adapt (retuned) | 1 | 70% [58, 82] | 210.6 | 300 | 0.237 | 1418 |
| R-on-best (no history) | 3 | 81% [71, 91] | 246.2 | 300 | 0.382 | 2109 |
| R-on-best-hist | 3 | 79% [67, 90] | 237.6 | 300 | 0.385 | 2123 |

per-seed success: R-on-best (no history): ['84%', '80%', '80%']; R-on-best-hist: ['78%', '80%', '78%']

## 2. Paired comparisons of success rate (A minus B, percentage points)

| A vs B | diff [95% CI] | per-seed diff of A | mean hits diff [95% CI] |
|---|---|---|---|
| R-on-best-hist vs T-robust (best) | -3.3 [-8.7, +0.0] | -4, -2, -4 | -9.3 [-25.1, +1.5] |
| R-on-best-hist vs T-adapt | +8.7 [+2.0, +16.7] | +8, +10, +8 | +27.0 [+7.2, +51.6] |
| R-on-best-hist vs T-nom | +78.7 [+67.3, +90.0] | +78, +80, +78 | +235.3 [+202.6, +267.6] |
| R-on-best (no history) vs T-robust (best) | -0.7 [-2.7, +1.3] | +2, -2, -2 | -0.7 [-8.0, +7.8] |
| R-on-best (no history) vs T-adapt | +11.3 [+4.0, +20.0] | +14, +10, +10 | +35.6 [+13.5, +61.5] |
| R-on-best (no history) vs T-nom | +81.3 [+70.7, +90.7] | +84, +80, +80 | +243.9 [+212.2, +271.7] |
| T-adapt vs T-robust | -12.0 [-22.0, -4.0] | | |

## 3. Subgroups (success [95% CI]; split at the midpoint of each range)

| group | n | T-nom | T-robust (retuned) | T-adapt (retuned) | R-on-best (no history) | R-on-best-hist | R-on-best-hist minus best-classical [CI] |
|---|---|---|---|---|---|---|---|
| e low (<0.58) | 20 | 0% [0, 0] | 75% [55, 95] | 70% [50, 90] | 77% [57, 95] | 75% [55, 90] | +0 [+0, +0] |
| e high | 30 | 0% [0, 0] | 87% [73, 97] | 70% [53, 87] | 84% [72, 96] | 81% [67, 93] | -6 [-14, +0] |
| delay low (<200 ms) | 23 | 0% [0, 0] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | +0 [+0, +0] |
| delay high | 27 | 0% [0, 0] | 67% [48, 82] | 44% [26, 63] | 65% [47, 81] | 60% [42, 79] | -6 [-16, +0] |
| ball mass low (<8.5x) | 24 | 0% [0, 0] | 100% [100, 100] | 92% [79, 100] | 99% [96, 100] | 96% [88, 100] | -4 [-12, +0] |
| ball mass high | 26 | 0% [0, 0] | 65% [46, 81] | 50% [31, 69] | 65% [46, 82] | 63% [44, 81] | -3 [-8, +0] |

## 4. H8b verdict

Rule (docs/03_hypotheses.md v4): R-on-best-hist success >= best-classical + 10 pp and paired bootstrap CI excludes 0. Observed vs T-robust: -3.3 pp [-8.7, +0.0] -> **H8b REFUTED (not supported)**.