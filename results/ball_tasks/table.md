# Ball tasks: classical teachers under model error (pilot)

Seeds 5000-5019 (n=20, DEMO set; not calibration, not test). One episode per seed, same seeds in every row (same balls / baskets, only the true task parameters or the sensing differ). success = primary metric (0/1); the interval is a percentile bootstrap 95% CI of the success rate (2000 resamples). Perturbation strengths are PILOT values picked on the demo seeds to show an effect; the formal strengths are to be calibrated on CAL seeds. Secondary: catch = relative speed ball vs basket just before the first contact (m/s, lower = softer; mean +- 95% t half-width over the episodes that touched the basket, n in the last column); toss = horizontal miss of the basket centre where the object crosses the rim plane (m). energy = mean sum(tau^2) per executed step.

## Catch (3-DoF arm with a basket, ball thrown in from outside the workspace; success = ball stays >= 0.5 s)

| condition | success | 95% CI | change vs no error | impact speed (m/s) | n | energy |
|---|---|---|---|---|---|---|
| no model error | 19/20 = 95% | [85%, 100%] | - | 0.70 +- 0.48 | 20 | 423 |
| air drag x2 | 18/20 = 90% | [75%, 100%] | -5 pts | 1.63 +- 0.36 | 20 | 387 |
| ball mass x2 | 16/20 = 80% | [60%, 95%] | -15 pts | 1.57 +- 0.84 | 20 | 520 |
| ball position noise 10 mm | 12/20 = 60% | [40%, 80%] | -35 pts | 1.55 +- 0.90 | 20 | 707 |
| ball position delay 10 ms | 6/20 = 30% | [10%, 50%] | -65 pts | 2.05 +- 0.90 | 20 | 522 |

## Toss (object held at the tip, released at t = 0.8 s, basket 0.7-1.8 m away, arm reach 1.0 m)

| condition | success | 95% CI | change vs no error | landing error (m) | n | energy |
|---|---|---|---|---|---|---|
| no model error | 20/20 = 100% | [100%, 100%] | - | 0.00 +- 0.00 | 20 | 72 |
| air drag x2 | 17/20 = 85% | [70%, 100%] | -15 pts | 0.14 +- 0.03 | 20 | 72 |
| object mass x2 | 15/20 = 75% | [55%, 95%] | -25 pts | 0.10 +- 0.03 | 20 | 83 |
| release 40 ms late | 15/20 = 75% | [55%, 95%] | -25 pts | 0.11 +- 0.08 | 20 | 68 |
| joint friction (Coulomb 5 Nm, viscous 1.0) | 12/20 = 60% | [40%, 80%] | -40 pts | 0.31 +- 0.20 | 20 | 326 |

