# Whip and Push-T: classical teachers under model error (pilot)

Seeds: DEMO set (5000-5019; not calibration, not test); whip n=20, push_t n=20. One episode per seed, the same seeds in every row (same targets / block poses, only the TRUE task parameters differ; the planner never sees them). success = primary metric (0/1) with a percentile-bootstrap 95% CI (2000 resamples); `change` = paired difference to the no-error row in percentage points with the paired bootstrap 95% CI. Perturbation strengths are PILOT values (chosen on the CAL seeds, not calibrated; for the three block parameters of push_t no reasonable strength changed the result, see the README). whip is an under-actuated 13-DoF fling planned open loop for the chain; Push-T is contact-rich with many solutions (see the README for why its teacher turned out strong). The last row of each block is a baseline with the same low-level control and no search, to show what the planning contributes.

## Whip (3-DoF arm + 10-link free chain, target 1.15-1.70 m from the shoulder, arm reach 1.0 m, hit radius 8 cm, 2.5 s)

| condition | success | 95% CI | change vs no error (paired) | min dist (m) | tip speed at hit (m/s) |
|---|---|---|---|---|---|
| no model error | 20/20 = 100% | [100%, 100%] | - | 0.06 +- 0.01 (n=20) | 9.07 +- 1.87 (n=20) |
| chain mass +50% | 18/20 = 90% | [75%, 100%] | -10 pts [-25, +0] | 0.08 +- 0.03 (n=20) | 8.80 +- 1.73 (n=18) |
| chain mass -50% | 16/20 = 80% | [65%, 95%] | -20 pts [-35, -5] | 0.10 +- 0.04 (n=20) | 9.34 +- 2.39 (n=16) |
| chain damping x3 | 16/20 = 80% | [60%, 95%] | -20 pts [-40, -5] | 0.08 +- 0.03 (n=20) | 9.29 +- 1.63 (n=16) |
| chain damping x0.25 | 15/20 = 75% | [55%, 90%] | -25 pts [-45, -10] | 0.09 +- 0.03 (n=20) | 9.13 +- 2.33 (n=15) |
| chain length +10% | 13/20 = 65% | [45%, 85%] | -35 pts [-55, -15] | 0.10 +- 0.04 (n=20) | 9.60 +- 2.19 (n=13) |
| chain length -10% | 12/20 = 60% | [40%, 80%] | -40 pts [-60, -20] | 0.08 +- 0.01 (n=20) | 8.42 +- 1.96 (n=12) |
| uneven chain (mass, log-sd 0.3 per link) | 13/20 = 65% | [45%, 85%] | -35 pts [-55, -15] | 0.09 +- 0.03 (n=20) | 10.57 +- 2.46 (n=13) |
| random keyframes (no search) | 3/20 = 15% | [0%, 35%] | -85 pts [-100, -65] | 0.36 +- 0.15 (n=20) | 18.04 +- 22.32 (n=3) |

## Push-T (2-DoF arm, T block 0.20 x 0.20 m, success = coverage >= 0.9, 30 s)

| condition | success | 95% CI | change vs no error (paired) | final coverage | time to success (s) | pusher path (m) |
|---|---|---|---|---|---|---|
| no model error | 20/20 = 100% | [100%, 100%] | - | 0.90 +- 0.00 (n=20) | 12.55 +- 1.52 (n=20) | 2.55 +- 0.38 (n=20) |
| block friction x0.5 | 20/20 = 100% | [100%, 100%] | +0 pts [+0, +0] | 0.90 +- 0.00 (n=20) | 13.55 +- 2.10 (n=20) | 2.75 +- 0.45 (n=20) |
| block friction x2 | 20/20 = 100% | [100%, 100%] | +0 pts [+0, +0] | 0.90 +- 0.00 (n=20) | 12.96 +- 2.30 (n=20) | 2.55 +- 0.42 (n=20) |
| block mass x2 | 20/20 = 100% | [100%, 100%] | +0 pts [+0, +0] | 0.90 +- 0.00 (n=20) | 13.53 +- 1.53 (n=20) | 2.72 +- 0.29 (n=20) |
| block mass x4 | 20/20 = 100% | [100%, 100%] | +0 pts [+0, +0] | 0.90 +- 0.00 (n=20) | 14.28 +- 1.73 (n=20) | 2.87 +- 0.37 (n=20) |
| COM offset 20 mm | 19/20 = 95% | [85%, 100%] | -5 pts [-15, +0] | 0.89 +- 0.01 (n=20) | 13.68 +- 1.95 (n=19) | 3.04 +- 0.74 (n=20) |
| COM offset 40 mm | 20/20 = 100% | [100%, 100%] | +0 pts [+0, +0] | 0.90 +- 0.00 (n=20) | 14.82 +- 2.39 (n=20) | 2.96 +- 0.45 (n=20) |
| block pose noise 8 mm / 0.08 rad | 20/20 = 100% | [100%, 100%] | +0 pts [+0, +0] | 0.90 +- 0.00 (n=20) | 12.36 +- 1.99 (n=20) | 2.44 +- 0.43 (n=20) |
| sensor delay 20 ms (q, qd, block pose) | 19/20 = 95% | [85%, 100%] | -5 pts [-15, +0] | 0.89 +- 0.03 (n=20) | 13.77 +- 1.91 (n=19) | 5.75 +- 1.15 (n=20) |
| sensor delay 30 ms | 0/20 = 0% | [0%, 0%] | -100 pts [-100, -100] | 0.00 +- 0.00 (n=20) | n/a | 145.88 +- 27.36 (n=20) |
| random push primitive (no search) | 0/20 = 0% | [0%, 0%] | -100 pts [-100, -100] | 0.03 +- 0.04 (n=20) | n/a | 4.62 +- 0.37 (n=20) |

Secondary metrics are mean +- 95% t half-width over the episodes where the quantity exists (n in brackets): whip `min dist` is the closest approach of the chain tip BEFORE the episode ended (a hit ends the episode, so a successful episode shows its first-hit distance, < 8 cm); `tip speed` is the chain-tip speed at the first hit; Push-T `final coverage` is the overlap with the goal at the end of the episode (for a success: the value that triggered it), `time to success` only over successes; `pusher path` of the 30 ms sensor-delay row (tens of metres) is an unstable arm flailing, not a plan.

