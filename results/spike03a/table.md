# spike03a results: tuned classical opponents vs residual PPO (H5), and H2 re-judged under the v2 formula

Test set = seeds 0-49 (locked), same refs/init errors/pulses for every method. RMSE in rad (transient included). +- = 95% t-CI over the 50 trajectories (RL = per-trajectory mean over its 5 PPO seeds). settle = first time after which max-joint error stays < 0.02 rad (clean only; 4.0 s = never, censored). recov = mean over pulses of time from pulse end until error stays < 0.02 rad (<=1 s window, dist only; 0 = never left the band). sat = fraction of control steps with any joint at the cap. energy = int tau^2 dt (Nm^2 s).

## 0. Frozen protocol: W1/W2 strength re-decided on the calibration set only (seeds 1000-1049)

Rule: mildest strength where the untuned teacher C0 has RMSE >= 1.3x its W0 value in BOTH clean and dist (calibration W0: clean 0.0107, dist 0.0161).

| weakness | spike02 value (chosen after looking at test seeds) | new value (calibration only) | calib ratio clean / dist at new value |
|---|---|---|---|
| W1 torque cap | 32 Nm (first 27, then raised after a teacher-only look at test) | **27 Nm** | 2.00 / 1.50 (3/50 calib trajectories RMSE>0.1; cap 28 gives only 1.46/1.26, a cliff) |
| W2 delay | 30 ms (6 steps; spec 10 ms did not hurt) | **30 ms (6 steps)** | 1.34 / 1.41 (5 steps = 25 ms: 0.90/1.04) |

W3 (Coulomb 1.5 Nm + 0.3 viscous) unchanged from spike02 (not re-decided; the task only asks to record W1/W2). Students were NOT re-run for the new W1 value, so W1/W2 are not evaluated in this spike.

## 1. Tuning budget (calibration set only, objective = 0.5*RMSE_clean + 0.5*RMSE_dist, transient included)

| W | method | free params | evals | plant control steps (1 eval = 100 rollouts x 800 = 80k) | wall-clock | calib objective | tuned values |
|---|---|---|---|---|---|---|---|
| W0 | C0 | 0 | 0 | 0 | 0 | 0.01342 | Kp=400, Kd (LQR default) |
| W0 | C1 | 2 | 80 | 6,400,000 | 206 s | 0.00902 | Kp x9.24, Kd x4.43 |
| W0 | C2 | 3 (+ the same 2 gains re-tuned) | 112 | 8,960,000 | 313 s | 0.00900 | Kp x21.28, Kd x10.02, Ki=1549 1/s^2, extra torque clamp +-16 Nm |
| W0 | residual PPO (spike02) | ~19k net weights, 0 tuned hyperparameters | - | 2,000,000 x 5 seeds (10,000,000 total; 2M per policy) | 147 s per seed (spike02 log, loaded machine) | - | - |
| W3 | C0 | 0 | 0 | 0 | 0 | 0.02449 | Kp=400, Kd (LQR default) |
| W3 | C1 | 2 | 80 | 6,400,000 | 202 s | 0.00921 | Kp x17.35, Kd x6.83 |
| W3 | C2 | 3 (+ the same 2 gains re-tuned) | 112 | 8,960,000 | 318 s | 0.00907 | Kp x23.22, Kd x10.06, Ki=2023 1/s^2, extra torque clamp +-16 Nm |
| W3 | residual PPO (spike02) | ~19k net weights, 0 tuned hyperparameters | - | 2,000,000 x 5 seeds (10,000,000 total; 2M per policy) | 172 s per seed (spike02 log, loaded machine) | - | - |

## 2. Per-condition metrics (test set)

| W | cond | method | RMSE | ss-RMSE (t>0.5s) | settle (s) | recov (s) | sat. frac | energy |
|---|---|---|---|---|---|---|---|---|
| W0 | clean | C0 | 0.0097 +- 0.0011 | 0.0002 +- 0.0000 | 0.14 +- 0.01 | - | 0.0001 +- 0.0001 | 1007 +- 145 |
| W0 | clean | C1 | 0.0076 +- 0.0010 | 0.0000 +- 0.0000 | 0.08 +- 0.01 | - | 0.0042 +- 0.0008 | 1179 +- 150 |
| W0 | clean | C2 | 0.0076 +- 0.0010 | 0.0006 +- 0.0001 | 0.08 +- 0.01 | - | 0.0846 +- 0.0155 | 13806 +- 1828 |
| W0 | clean | ResidualPPO | 0.0081 +- 0.0010 | 0.0009 +- 0.0001 | 0.10 +- 0.01 | - | 0.0001 +- 0.0001 | 1018 +- 145 |
| W0 | dist | C0 | 0.0149 +- 0.0014 | 0.0109 +- 0.0017 | - | 0.08 +- 0.01 | 0.0001 +- 0.0001 | 1020 +- 145 |
| W0 | dist | C1 | 0.0078 +- 0.0010 | 0.0014 +- 0.0002 | - | 0.00 +- 0.00 | 0.0042 +- 0.0008 | 1194 +- 150 |
| W0 | dist | C2 | 0.0076 +- 0.0010 | 0.0009 +- 0.0001 | - | 0.00 +- 0.00 | 0.0795 +- 0.0149 | 13471 +- 1794 |
| W0 | dist | ResidualPPO | 0.0082 +- 0.0010 | 0.0016 +- 0.0001 | - | 0.00 +- 0.00 | 0.0001 +- 0.0001 | 1033 +- 144 |
| W3 | clean | C0 | 0.0219 +- 0.0020 | 0.0189 +- 0.0023 | 3.53 +- 0.31 | - | 0.0001 +- 0.0001 | 1035 +- 146 |
| W3 | clean | C1 | 0.0076 +- 0.0010 | 0.0012 +- 0.0002 | 0.07 +- 0.01 | - | 0.0062 +- 0.0013 | 1282 +- 155 |
| W3 | clean | C2 | 0.0076 +- 0.0010 | 0.0010 +- 0.0001 | 0.07 +- 0.01 | - | 0.5090 +- 0.0196 | 36444 +- 950 |
| W3 | clean | ResidualPPO | 0.0081 +- 0.0010 | 0.0010 +- 0.0001 | 0.10 +- 0.01 | - | 0.0001 +- 0.0002 | 1049 +- 146 |
| W3 | dist | C0 | 0.0237 +- 0.0023 | 0.0212 +- 0.0026 | - | 0.48 +- 0.08 | 0.0001 +- 0.0001 | 1046 +- 145 |
| W3 | dist | C1 | 0.0077 +- 0.0010 | 0.0015 +- 0.0002 | - | 0.00 +- 0.00 | 0.0062 +- 0.0013 | 1297 +- 154 |
| W3 | dist | C2 | 0.0076 +- 0.0010 | 0.0012 +- 0.0001 | - | 0.00 +- 0.00 | 0.5032 +- 0.0189 | 36273 +- 959 |
| W3 | dist | ResidualPPO | 0.0082 +- 0.0010 | 0.0017 +- 0.0001 | - | 0.00 +- 0.00 | 0.0001 +- 0.0002 | 1063 +- 145 |

## 3. Paired differences (50 trajectories). impr = mean(A-B)/mean(A): positive = B better than A

| W | cond | A -> B | RMSE diff (B-A) [95% CI] | impr | ss-RMSE impr | energy rel diff |
|---|---|---|---|---|---|---|
| W0 | clean | C0 -> C1 | -0.0022 [-0.0028, -0.0015] | +22.2% | +89.8% | +17.1% |
| W0 | clean | C1 -> C2 | +0.0000 [-0.0000, +0.0001] | -0.2% | -3275.8% | +1070.6% |
| W0 | clean | C0 -> C2 | -0.0021 [-0.0027, -0.0015] | +22.0% | -243.3% | +1270.9% |
| W0 | clean | C0 -> RL | -0.0016 [-0.0020, -0.0013] | +16.7% | -395.5% | +1.1% |
| W0 | clean | C1 -> RL | +0.0005 [-0.0000, +0.0011] | -7.1% | -4772.2% | -13.7% |
| W0 | clean | C2 -> RL | +0.0005 [-0.0000, +0.0011] | -6.8% | -44.3% | -92.6% |
| W0 | dist | C0 -> C1 | -0.0071 [-0.0083, -0.0060] | +47.9% | +87.1% | +17.0% |
| W0 | dist | C1 -> C2 | -0.0001 [-0.0002, -0.0001] | +1.8% | +36.7% | +1028.2% |
| W0 | dist | C0 -> C2 | -0.0073 [-0.0085, -0.0060] | +48.9% | +91.8% | +1220.0% |
| W0 | dist | C0 -> RL | -0.0067 [-0.0081, -0.0053] | +44.8% | +85.1% | +1.2% |
| W0 | dist | C1 -> RL | +0.0005 [-0.0001, +0.0010] | -6.0% | -15.2% | -13.5% |
| W0 | dist | C2 -> RL | +0.0006 [+0.0001, +0.0012] | -8.0% | -82.0% | -92.3% |
| W3 | clean | C0 -> C1 | -0.0143 [-0.0162, -0.0123] | +65.2% | +93.5% | +24.0% |
| W3 | clean | C1 -> C2 | -0.0000 [-0.0001, +0.0000] | +0.6% | +17.5% | +2741.9% |
| W3 | clean | C0 -> C2 | -0.0143 [-0.0163, -0.0123] | +65.4% | +94.6% | +3422.7% |
| W3 | clean | C0 -> RL | -0.0138 [-0.0160, -0.0116] | +62.9% | +94.5% | +1.4% |
| W3 | clean | C1 -> RL | +0.0005 [-0.0002, +0.0011] | -6.4% | +15.3% | -18.2% |
| W3 | clean | C2 -> RL | +0.0005 [-0.0000, +0.0011] | -7.1% | -2.7% | -97.1% |
| W3 | dist | C0 -> C1 | -0.0160 [-0.0183, -0.0138] | +67.6% | +93.1% | +24.0% |
| W3 | dist | C1 -> C2 | -0.0001 [-0.0002, +0.0000] | +0.8% | +17.5% | +2696.0% |
| W3 | dist | C0 -> C2 | -0.0161 [-0.0184, -0.0138] | +67.9% | +94.3% | +3368.4% |
| W3 | dist | C0 -> RL | -0.0155 [-0.0180, -0.0130] | +65.3% | +92.2% | +1.7% |
| W3 | dist | C1 -> RL | +0.0005 [-0.0001, +0.0012] | -7.2% | -12.6% | -18.0% |
| W3 | dist | C2 -> RL | +0.0006 [+0.0000, +0.0012] | -8.0% | -36.6% | -97.1% |

## 4. H5 verdict (rule in docs/03_hypotheses.md v2)

Rule operationalisation (fixed before looking at test numbers): gap = (RMSE_C2 - RMSE_RL)/RMSE_C2 per cell, paired CI over 50 trajectories. Cell = "TUNING" if gap <= 10% (or C2 better); "RL-GAIN" if gap > 10% and CI of RL-C2 entirely < 0; else "unclear". H5 supported if all four cells (W0/W3 x clean/dist) are TUNING; refuted if all four are RL-GAIN; otherwise mixed (reported per cell).

| W | cond | RMSE C2 | RMSE RL | gap | CI of RL-C2 | cell |
|---|---|---|---|---|---|---|
| W0 | clean | 0.0076 | 0.0081 | -6.8% | [-0.0000, +0.0011] | TUNING |
| W0 | dist | 0.0076 | 0.0082 | -8.0% | [+0.0001, +0.0012] | TUNING |
| W3 | clean | 0.0076 | 0.0081 | -7.1% | [-0.0000, +0.0011] | TUNING |
| W3 | dist | 0.0076 | 0.0082 | -8.0% | [+0.0000, +0.0012] | TUNING |

**H5: SUPPORTED (RL advantage mostly = untuned teacher).** Cells: ['TUNING', 'TUNING', 'TUNING', 'TUNING']

For context: what tuning alone removes from the spike02 "RL win" over C0: C0->C2 impr vs C0->RL impr (RMSE):

| W | cond | C0->C1 | C0->C2 | C0->RL |
|---|---|---|---|---|
| W0 | clean | +22.2% | +22.0% | +16.7% |
| W0 | dist | +47.9% | +48.9% | +44.8% |
| W3 | clean | +65.2% | +65.4% | +62.9% |
| W3 | dist | +67.6% | +67.9% | +65.3% |

## 5. H2 re-judged with the v2 formula (spike02 numbers: teacher and RL per-trajectory RMSE, 5 RL seeds averaged; no new runs)

extra contribution of weakness k = (Teacher_Wk - Teacher_W0) - (RL_Wk - RL_W0), per trajectory (same seeds), paired over 50. Operationalisation (mine; v2 gives the formula but no threshold): H2 "weakness-attributable win" in cell if extra contribution CI entirely > 0 AND mean extra >= 10% of mean Teacher_Wk RMSE. W4 (combined) is shown for information, not counted (card says single weaknesses). W1 uses the spike02 cap 32 Nm and W2 30 ms (students were not re-trained on the new calibration values).

| W | cond | Teacher_Wk - Teacher_W0 | RL_Wk - RL_W0 | extra [95% CI] | extra / Teacher_Wk | counts? | pass |
|---|---|---|---|---|---|---|---|
| W1 | clean | +0.0120 | +0.0055 | +0.0065 [-0.0019, +0.0149] | +29.9% | yes | no |
| W1 | dist | +0.0119 | +0.0072 | +0.0047 [-0.0024, +0.0118] | +17.6% | yes | no |
| W2 | clean | +0.0043 | +0.0042 | +0.0001 [-0.0010, +0.0012] | +0.7% | yes | no |
| W2 | dist | +0.0070 | +0.0114 | -0.0044 [-0.0063, -0.0025] | -20.0% | yes | no |
| W3 | clean | +0.0122 | +0.0000 | +0.0121 [+0.0100, +0.0143] | +55.5% | yes | YES |
| W3 | dist | +0.0088 | +0.0000 | +0.0088 [+0.0072, +0.0105] | +37.2% | yes | YES |
| W4 | clean | +0.0296 | +0.0189 | +0.0107 [+0.0057, +0.0158] | +27.3% | info | YES |
| W4 | dist | +0.0267 | +0.0227 | +0.0041 [+0.0003, +0.0078] | +9.7% | info | no |

**H2 (v2): SUPPORTED** (passing single-weakness cells: 2/6).

Spike02 original verdict for comparison: SUPPORTED (RL beat teacher by >=10% in 5 of 6 single-weakness cells, but it also beat it by 17%/45% at W0, which is what v2 corrects for).

