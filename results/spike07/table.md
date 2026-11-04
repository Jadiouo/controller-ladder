# spike07 table: H9 (oracle headroom) and H10 (fair RL), ball juggling, cap N=300, locked TEST seeds 0-49 (evaluated once)

Distribution C (results/spike06/dist.json): restitution e~U[0.32, 0.85], ball mass x U[1.0, 16.0], sensor delay U[0.0, 400.0] ms. best-classical (chosen on CAL only) = **T-robust(spike06)**. All RL residuals are stacked on it (cap 12 Nm per joint).
Success = reached the 300-hit cap. RL rows: per-episode success = mean over the 3 training seeds; CIs = paired bootstrap over the 50 test episodes (5000 resamples).

## 1. Main table

| method | #seeds | success [95% CI] | mean hits | median hits | apex err (m) | torque energy (Nm^2) | residual L2 norm mean / max (Nm, TEST) |
|---|---|---|---|---|---|---|---|
| T-nom | 1 | 0% [0, 0] | 2.3 | 2 | 0.489 | 216 | - |
| T-robust(spike06) **[best-classical]** | 1 | 82% [70, 92] | 246.9 | 300 | 0.384 | 2113 | - |
| T-robust(kp10000) | 1 | 86% [76, 94] | 259.2 | 300 | 0.364 | 4613 | - |
| T-adapt(spike06) | 1 | 70% [56, 82] | 210.6 | 300 | 0.237 | 1418 | - |
| R-PPO-fixed (hist) | 3 | 79% [68, 90] | 239.2 | 300 | 0.385 | 2132 | 0.18 / 0.72 |
| R-SAC (hist) | 3 | 63% [52, 75] | 206.2 | 300 | 0.385 | 2126 | 0.43 / 2.85 |
| R-oracle (PPO-fixed + true e, mass, delay) | 3 | 81% [69, 91] | 242.8 | 300 | 0.385 | 2108 | 0.15 / 0.67 |

per-seed success (TEST): R-PPO-fixed (hist): ['78%', '80%', '80%']; R-SAC (hist): ['52%', '76%', '62%']; R-oracle (PPO-fixed + true e, mass, delay): ['82%', '80%', '80%']

per-seed residual norm mean/max (Nm, TEST): R-PPO-fixed (hist): ['0.20/0.72', '0.14/0.51', '0.20/0.59']; R-SAC (hist): ['0.42/2.85', '0.47/2.41', '0.40/1.94']; R-oracle (PPO-fixed + true e, mass, delay): ['0.18/0.67', '0.14/0.49', '0.13/0.51']

## 2. Paired comparisons of success rate (A minus B, percentage points)

| A vs B | diff [95% CI] | one-sided p (A<=B) | per-seed diff of A | mean hits diff [95% CI] |
|---|---|---|---|---|
| R-PPO-fixed (hist) vs T-robust(spike06) (best) | -2.7 [-6.7, +0.0] | 1.0000 | -4, -2, -2 | -7.7 [-19.8, +0.6] |
| R-PPO-fixed (hist) vs T-adapt(spike06) | +9.3 [+2.7, +17.3] | 0.0020 | +8, +10, +10 | +28.5 [+8.4, +53.9] |
| R-PPO-fixed (hist) vs T-nom | +79.3 [+68.0, +90.0] | 0.0000 | +78, +80, +80 | +236.9 [+205.4, +268.1] |
| R-SAC (hist) vs T-robust(spike06) (best) | -18.7 [-27.3, -10.7] | 1.0000 | -30, -6, -20 | -40.7 [-62.0, -21.3] |
| R-SAC (hist) vs T-adapt(spike06) | -6.7 [-14.7, +0.7] | 0.9620 | -18, +6, -8 | -4.4 [-25.9, +18.0] |
| R-SAC (hist) vs T-nom | +63.3 [+52.0, +74.7] | 0.0000 | +52, +76, +62 | +203.9 [+169.9, +234.7] |
| R-oracle (PPO-fixed + true e, mass, delay) vs T-robust(spike06) (best) | -1.3 [-4.0, +0.0] | 1.0000 | +0, -2, -2 | -4.1 [-12.1, +0.0] |
| R-oracle (PPO-fixed + true e, mass, delay) vs T-adapt(spike06) | +10.7 [+3.3, +20.0] | 0.0016 | +12, +10, +10 | +32.2 [+10.2, +59.4] |
| R-oracle (PPO-fixed + true e, mass, delay) vs T-nom | +80.7 [+70.0, +90.7] | 0.0000 | +82, +80, +80 | +240.5 [+208.5, +269.5] |

## 3. Subgroups (success [95% CI]; split at the midpoint of each range)

| group | n | T-nom | T-robust(spike06) **[best-classical]** | R-PPO-fixed (hist) | R-SAC (hist) | R-oracle (PPO-fixed + true e, mass, delay) | R-oracle minus best [CI] |
|---|---|---|---|---|---|---|---|
| e low (<0.58) | 20 | 0% [0, 0] | 75% [55, 90] | 75% [55, 90] | 57% [38, 75] | 75% [55, 90] | +0 [+0, +0] |
| e high | 30 | 0% [0, 0] | 87% [73, 97] | 82% [69, 93] | 68% [53, 81] | 84% [71, 96] | -2 [-7, +0] |
| delay low (<200 ms) | 23 | 0% [0, 0] | 100% [100, 100] | 100% [100, 100] | 96% [88, 100] | 100% [100, 100] | +0 [+0, +0] |
| delay high | 27 | 0% [0, 0] | 67% [48, 81] | 62% [43, 79] | 36% [22, 49] | 64% [46, 81] | -2 [-7, +0] |
| ball mass low (<8.5x) | 24 | 0% [0, 0] | 100% [100, 100] | 97% [92, 100] | 78% [64, 89] | 100% [100, 100] | +0 [+0, +0] |
| ball mass high | 26 | 0% [0, 0] | 65% [46, 85] | 63% [44, 81] | 50% [33, 67] | 63% [44, 81] | -3 [-8, +0] |

## 4. Verdicts (rules from docs/03_hypotheses.md v5)

**H9** (R-oracle >= best-classical + 10 pp, paired-bootstrap 95% CI excludes 0): observed -1.3 pp [-4.0, +0.0] -> **H9 NOT SUPPORTED**.

**H10** (at least one of R-PPO-fixed / R-SAC >= best-classical + 10 pp, CI excludes 0; two comparisons, Holm-corrected one-sided bootstrap p, alpha 0.05):

| comparison | diff pp [95% CI] | raw p | Holm-adjusted p | >=10 pp and CI>0 | passes with Holm (adj p<0.05) |
|---|---|---|---|---|---|
| R-PPO-fixed (hist) vs T-robust(spike06) | -2.7 [-6.7, +0.0] | 1.0000 | 1.0000 | no | no |
| R-SAC (hist) vs T-robust(spike06) | -18.7 [-27.3, -10.7] | 1.0000 | 1.0000 | no | no |

-> **H10 REFUTED (not supported)**.

## 5. CAL training-curve summary (checkpoints at 1/7 ... 7/7 of training; CAL = seeds 1000-1049, mean action, success = reached 300 hits; selected checkpoint marked *)

(initialisation = best-classical alone, CAL success 76%, mean hits 229.4)

Note: the train-success column is the rolling success of the last 60 completed training episodes; it is biased low early (only failures complete first) and uninformative for 128 envs x 2.5M steps; the CAL curve is the criterion.

| method / seed | CAL success at the 7 checkpoints (%) | selected (CAL success %, mean hits) | min CAL ckpt (%) | train success (last 60 train eps) first -> min -> last (%) | residual L2 mean at CAL ckpts (Nm) | train sim steps | ckpt-eval sim steps | train wall (s) |
|---|---|---|---|---|---|---|---|---|
| R-ppo#0 | 72 *72 70 64 64 68 68 | 72%, 217.0 | 64 | 0 -> 0 -> 68 | 0.14 0.21 0.24 0.27 0.29 0.30 0.32 | 2.5M | 4.8M | 549 |
| R-ppo#1 | *76 74 74 72 74 74 72 | 76%, 229.0 | 72 | 0 -> 0 -> 85 | 0.14 0.17 0.20 0.24 0.30 0.30 0.31 | 2.5M | 5.2M | 561 |
| R-ppo#2 | *74 72 64 68 70 66 68 | 74%, 222.5 | 64 | 0 -> 0 -> 68 | 0.20 0.24 0.26 0.26 0.27 0.31 0.32 | 2.5M | 4.8M | 534 |
| R-oracle#0 | *76 74 72 70 66 70 72 | 76%, 229.0 | 66 | 0 -> 0 -> 68 | 0.19 0.20 0.25 0.29 0.29 0.29 0.33 | 2.5M | 5.0M | 524 |
| R-oracle#1 | *76 74 76 74 74 74 70 | 76%, 228.9 | 70 | 0 -> 0 -> 68 | 0.14 0.17 0.21 0.24 0.28 0.29 0.31 | 2.5M | 5.2M | 535 |
| R-oracle#2 | *74 72 72 72 72 70 72 | 74%, 222.8 | 70 | 0 -> 0 -> 75 | 0.14 0.22 0.23 0.25 0.27 0.29 0.32 | 2.5M | 5.0M | 530 |
| R-sac#0 | 16 10 42 32 *56 42 22 | 56%, 180.4 | 10 | 0 -> 0 -> 17 | 0.68 0.44 0.40 0.39 0.42 0.49 0.50 | 1.2M | 2.7M | 2018 |
| R-sac#1 | 56 *70 32 66 36 56 36 | 70%, 214.9 | 32 | 0 -> 0 -> 30 | 0.40 0.47 0.35 0.40 0.44 0.52 0.53 | 1.2M | 3.9M | 2047 |
| R-sac#2 | *58 28 42 46 24 12 20 | 58%, 179.9 | 12 | 0 -> 0 -> 20 | 0.42 0.37 0.36 0.46 0.51 0.55 0.57 | 1.2M | 2.8M | 2006 |