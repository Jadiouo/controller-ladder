# MJX-trained residual PPO, evaluated on the CPU arena (TEST_SEEDS 0-49, split='test')

RMSE in rad, transient included, +- = 95% t-CI over the 50 trajectories (policy value = per-trajectory mean over its PPO seeds). Paired rows: improvement = mean(A-B)/mean(A), positive = B better; CI over the 50 paired trajectories; WIN = CI of (A-B) > 0 and improvement >= 10% (protocol).

## W0

| method | clean RMSE | dist RMSE |
|---|---|---|
| spike02 RL (CPU torch, 2M x5) | 0.0081 +- 0.0010 | 0.0082 +- 0.0010 |
| spike03a C0 (untuned CTC) | 0.0097 +- 0.0011 | 0.0149 +- 0.0014 |
| spike03a C1 (tuned gains) | 0.0076 +- 0.0010 | 0.0078 +- 0.0010 |
| MJX PPO b2m (5 seeds) | 0.0081 +- 0.0010 | 0.0082 +- 0.0010 |
| MJX PPO b20m (3 seeds) | 0.0080 +- 0.0010 | 0.0081 +- 0.0010 |
| MJX PPO b2m_fresh (3 seeds) | 0.0081 +- 0.0011 | 0.0082 +- 0.0010 |

Per-seed mean RMSE (clean / dist): b2m: 0.0081/0.0082, 0.0082/0.0084, 0.0080/0.0081, 0.0081/0.0082, 0.0079/0.0081; b20m: 0.0080/0.0081, 0.0081/0.0082, 0.0080/0.0081; b2m_fresh: 0.0081/0.0082, 0.0081/0.0083, 0.0081/0.0082
arena `ctc_lqr` (this repo, today) vs spike03a C0 per-trajectory RMSE: max |diff| = 2.4e-17

| A | B | cond | mean(A-B) | 95% CI | improvement | WIN |
|---|---|---|---|---|---|---|
| spike03a C0 (untuned CTC) | MJX PPO b2m (5 seeds) | clean | +0.00165 | [+0.00131, +0.00199] | +17.0% | YES |
| spike03a C0 (untuned CTC) | MJX PPO b2m (5 seeds) | dist | +0.00670 | [+0.00530, +0.00811] | +45.0% | YES |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m (5 seeds) | clean | +0.00003 | [+0.00000, +0.00005] | +0.3% | no |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m (5 seeds) | dist | +0.00004 | [+0.00001, +0.00006] | +0.4% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m (5 seeds) | clean | -0.00051 | [-0.00105, +0.00004] | -6.7% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m (5 seeds) | dist | -0.00043 | [-0.00100, +0.00013] | -5.6% | no |
| spike03a C0 (untuned CTC) | MJX PPO b20m (3 seeds) | clean | +0.00170 | [+0.00138, +0.00202] | +17.5% | YES |
| spike03a C0 (untuned CTC) | MJX PPO b20m (3 seeds) | dist | +0.00679 | [+0.00537, +0.00820] | +45.6% | YES |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b20m (3 seeds) | clean | +0.00008 | [+0.00005, +0.00011] | +1.0% | no |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b20m (3 seeds) | dist | +0.00012 | [+0.00009, +0.00015] | +1.4% | no |
| spike03a C1 (tuned gains) | MJX PPO b20m (3 seeds) | clean | -0.00046 | [-0.00100, +0.00009] | -6.0% | no |
| spike03a C1 (tuned gains) | MJX PPO b20m (3 seeds) | dist | -0.00035 | [-0.00092, +0.00022] | -4.5% | no |
| spike03a C0 (untuned CTC) | MJX PPO b2m_fresh (3 seeds) | clean | +0.00162 | [+0.00127, +0.00197] | +16.7% | YES |
| spike03a C0 (untuned CTC) | MJX PPO b2m_fresh (3 seeds) | dist | +0.00667 | [+0.00525, +0.00808] | +44.7% | YES |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m_fresh (3 seeds) | clean | -0.00000 | [-0.00001, +0.00001] | -0.0% | no |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m_fresh (3 seeds) | dist | -0.00000 | [-0.00001, +0.00001] | -0.0% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m_fresh (3 seeds) | clean | -0.00054 | [-0.00109, +0.00002] | -7.1% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m_fresh (3 seeds) | dist | -0.00047 | [-0.00104, +0.00010] | -6.1% | no |

## W3

| method | clean RMSE | dist RMSE |
|---|---|---|
| spike02 RL (CPU torch, 2M x5) | 0.0081 +- 0.0010 | 0.0082 +- 0.0010 |
| spike03a C0 (untuned CTC) | 0.0219 +- 0.0020 | 0.0237 +- 0.0023 |
| spike03a C1 (tuned gains) | 0.0076 +- 0.0010 | 0.0077 +- 0.0010 |
| MJX PPO b2m (5 seeds) | 0.0081 +- 0.0010 | 0.0082 +- 0.0010 |
| MJX PPO b20m (3 seeds) | 0.0080 +- 0.0010 | 0.0081 +- 0.0010 |
| MJX PPO b2m_fresh (3 seeds) | 0.0081 +- 0.0010 | 0.0082 +- 0.0010 |

Per-seed mean RMSE (clean / dist): b2m: 0.0081/0.0082, 0.0082/0.0083, 0.0081/0.0082, 0.0081/0.0082, 0.0081/0.0082; b20m: 0.0080/0.0081, 0.0081/0.0082, 0.0080/0.0081; b2m_fresh: 0.0080/0.0081, 0.0082/0.0082, 0.0081/0.0082
arena `ctc_lqr` (this repo, today) vs spike03a C0 per-trajectory RMSE: max |diff| = 3.5e-17

| A | B | cond | mean(A-B) | 95% CI | improvement | WIN |
|---|---|---|---|---|---|---|
| spike03a C0 (untuned CTC) | MJX PPO b2m (5 seeds) | clean | +0.01378 | [+0.01158, +0.01597] | +63.0% | YES |
| spike03a C0 (untuned CTC) | MJX PPO b2m (5 seeds) | dist | +0.01552 | [+0.01306, +0.01798] | +65.4% | YES |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m (5 seeds) | clean | +0.00001 | [-0.00001, +0.00002] | +0.1% | no |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m (5 seeds) | dist | +0.00003 | [+0.00001, +0.00004] | +0.3% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m (5 seeds) | clean | -0.00048 | [-0.00112, +0.00016] | -6.3% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m (5 seeds) | dist | -0.00052 | [-0.00117, +0.00012] | -6.8% | no |
| spike03a C0 (untuned CTC) | MJX PPO b20m (3 seeds) | clean | +0.01385 | [+0.01165, +0.01605] | +63.3% | YES |
| spike03a C0 (untuned CTC) | MJX PPO b20m (3 seeds) | dist | +0.01559 | [+0.01313, +0.01806] | +65.7% | YES |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b20m (3 seeds) | clean | +0.00008 | [+0.00005, +0.00011] | +0.9% | no |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b20m (3 seeds) | dist | +0.00010 | [+0.00007, +0.00013] | +1.2% | no |
| spike03a C1 (tuned gains) | MJX PPO b20m (3 seeds) | clean | -0.00041 | [-0.00105, +0.00022] | -5.4% | no |
| spike03a C1 (tuned gains) | MJX PPO b20m (3 seeds) | dist | -0.00045 | [-0.00109, +0.00019] | -5.8% | no |
| spike03a C0 (untuned CTC) | MJX PPO b2m_fresh (3 seeds) | clean | +0.01379 | [+0.01159, +0.01600] | +63.0% | YES |
| spike03a C0 (untuned CTC) | MJX PPO b2m_fresh (3 seeds) | dist | +0.01553 | [+0.01306, +0.01800] | +65.5% | YES |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m_fresh (3 seeds) | clean | +0.00002 | [+0.00001, +0.00004] | +0.3% | no |
| spike02 RL (CPU torch, 2M x5) | MJX PPO b2m_fresh (3 seeds) | dist | +0.00004 | [+0.00003, +0.00005] | +0.5% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m_fresh (3 seeds) | clean | -0.00047 | [-0.00111, +0.00018] | -6.1% | no |
| spike03a C1 (tuned gains) | MJX PPO b2m_fresh (3 seeds) | dist | -0.00051 | [-0.00116, +0.00014] | -6.6% | no |
