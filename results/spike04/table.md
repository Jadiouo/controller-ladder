# spike04 results: do tuned classical gains still tie residual RL once the sim has sensor noise + delay? (H7)

Test set = seeds 0-49 (locked). sigma_q = 0.001 rad (measured velocity = finite difference of noisy angles, sigma_qd = 0.283 rad/s). N1 = noise + 30 ms delay, N2 = noise only. RMSE in rad vs TRUE state, transient included. +- = 95% t-CI over 50 trajectories (RL rows = per-trajectory mean over 5 PPO seeds). settle = clean only, 4.0 = never (censored); recov = dist only. C* (the classical comparator and RL base) is chosen on calibration only: N1 -> C1, N2 -> C1F.

## 0. Noise-strength calibration (calibration seeds 1000-1049, C1-old = spike03a W0 gains Kp x9.24, Kd x4.43)

| condition | sigma_q | C0 RMSE clean/dist (ratio vs no noise) | C1-old RMSE clean/dist (ratio vs no noise) |
|---|---|---|---|
| N2 | 0 | 0.0107 / 0.0161 | 0.0089 / 0.0091 |
| N2 | 0.0001 | 0.0107 / 0.0161 (1.00x / 1.00x) | 0.0090 / 0.0092 (1.01x / 1.01x) |
| N2 | 0.0002 | 0.0107 / 0.0161 (0.99x / 1.00x) | 0.0090 / 0.0092 (1.01x / 1.01x) |
| N2 | 0.0005 | 0.0107 / 0.0160 (0.99x / 1.00x) | 0.0093 / 0.0095 (1.05x / 1.04x) |
| N2 | 0.001 **(chosen)** | 0.0107 / 0.0160 (1.00x / 0.99x) | 0.0124 / 0.0126 (1.39x / 1.38x) |
| N2 | 0.002 | 0.0118 / 0.0168 (1.10x / 1.04x) | 0.0225 / 0.0229 (2.52x / 2.51x) |
| N2 | 0.003 | 0.0201 / 0.0240 (1.87x / 1.49x) | 0.0351 / 0.0357 (3.93x / 3.92x) |
| N2 | 0.005 | 0.0650 / 0.0686 (6.07x / 4.25x) | 0.0646 / 0.0657 (7.23x / 7.21x) |
| N1 | 0 | 0.0144 / 0.0227 | 2.0259 / 2.0907 |
| N1 | 0.0001 | 0.2174 / 0.2196 (15.15x / 9.67x) | 2.2315 / 2.2906 (1.10x / 1.10x) |
| N1 | 0.0002 | 0.2177 / 0.2200 (15.17x / 9.69x) | 2.3111 / 2.3160 (1.14x / 1.11x) |
| N1 | 0.0005 | 0.2185 / 0.2198 (15.23x / 9.68x) | 2.3635 / 2.2514 (1.17x / 1.08x) |
| N1 | 0.001 | 0.2186 / 0.2235 (15.24x / 9.84x) | 2.2211 / 2.2273 (1.10x / 1.07x) |
| N1 | 0.002 | 0.2131 / 0.2196 (14.85x / 9.67x) | 2.3343 / 2.2860 (1.15x / 1.09x) |
| N1 | 0.003 | 0.1983 / 0.1985 (13.82x / 8.74x) | 2.2407 / 2.1935 (1.11x / 1.05x) |
| N1 | 0.005 | 0.1356 / 0.1360 (9.45x / 5.99x) | 2.0939 / 2.0904 (1.03x / 1.00x) |

Rule (H7.md + Amendment 1): smallest sigma with C1-old >= 1.3x its no-noise RMSE in both clean and dist, in N2. In N1 the delay alone already breaks C1-old (and makes C0 near-unstable: any noise lifts C0 from 0.014 to ~0.22), so the ratio is not usable there.

## 1. Tuning / training budget (three axes)

| cond | method | free DoF | evals / policies | plant control steps | wall-clock | calib objective | tuned values |
|---|---|---|---|---|---|---|---|
| N1 | C0 | 0 | 0 | 0 | 0 | 0.22107 | spike02 teacher |
| N1 | C1-old | 0 (spike03a) | 0 | 0 | 0 | 2.22421 | Kp x9.24, Kd x4.43 |
| N1 | C1-new | 2 | 80 | 6,400,000 | 291 s | 0.01687 | Kp x0.76, Kd x0.79 |
| N1 | C1-new+LPF | 3 | 112 | 8,960,000 | 409 s | 0.01743 | Kp x0.77, Kd x0.82, fc 65.0 Hz |
| N1 | RL(base C0) | - | NOT RUN (time budget rule, see README) | - | - | - | - |
| N1 | RL(base C1*) | ~19k actor weights, 0 tuned hyperparameters | 5 seeds | 10,000,000 (2M per policy) | 645 s per seed (loaded machine) | - | residual bound 16 Nm |
| N2 | C0 | 0 | 0 | 0 | 0 | 0.01336 | spike02 teacher |
| N2 | C1-old | 0 (spike03a) | 0 | 0 | 0 | 0.01253 | Kp x9.24, Kd x4.43 |
| N2 | C1-new | 2 | 80 | 6,400,000 | 261 s | 0.00952 | Kp x3.71, Kd x1.62 |
| N2 | C1-new+LPF | 3 | 112 | 8,960,000 | 398 s | 0.00947 | Kp x3.88, Kd x1.86, fc 67.5 Hz |
| N2 | RL(base C0) | - | NOT RUN (time budget rule, see README) | - | - | - | - |
| N2 | RL(base C1*) | - | NOT RUN (time budget rule, see README) | - | - | - | - |

## 2. Per-condition metrics (test set)

| cond | eval | method | RMSE | ss-RMSE (t>0.5s) | settle (s) | recov (s) | sat. frac | energy |
|---|---|---|---|---|---|---|---|---|
| N1 | clean | C0 | 0.1950 +- 0.0306 | 0.2074 +- 0.0327 | 4.00 +- 0.00 | - | 0.4203 +- 0.0306 | 23567 +- 1095 |
| N1 | clean | C1-old | 2.2870 +- 0.1345 | 2.4263 +- 0.1429 | 4.00 +- 0.00 | - | 0.9977 +- 0.0010 | 50164 +- 119 |
| N1 | clean | C1-new (=C*) | 0.0104 +- 0.0010 | 0.0038 +- 0.0006 | 0.36 +- 0.11 | - | 0.0008 +- 0.0006 | 1744 +- 210 |
| N1 | clean | C1-new+LPF | 0.0110 +- 0.0010 | 0.0047 +- 0.0006 | 0.55 +- 0.15 | - | 0.0008 +- 0.0007 | 1688 +- 204 |
| N1 | clean | RL(base C1*) | 0.0127 +- 0.0009 | 0.0067 +- 0.0006 | 1.00 +- 0.28 | - | 0.0003 +- 0.0003 | 1739 +- 207 |
| N1 | dist | C0 | 0.1936 +- 0.0305 | 0.2058 +- 0.0327 | - | 0.68 +- 0.05 | 0.4072 +- 0.0322 | 23083 +- 1157 |
| N1 | dist | C1-old | 2.3848 +- 0.1360 | 2.5309 +- 0.1453 | - | 0.68 +- 0.05 | 0.9979 +- 0.0011 | 50180 +- 122 |
| N1 | dist | C1-new (=C*) | 0.0214 +- 0.0024 | 0.0194 +- 0.0027 | - | 0.16 +- 0.02 | 0.0008 +- 0.0006 | 1829 +- 211 |
| N1 | dist | C1-new+LPF | 0.0219 +- 0.0023 | 0.0199 +- 0.0026 | - | 0.22 +- 0.03 | 0.0008 +- 0.0007 | 1809 +- 208 |
| N1 | dist | RL(base C1*) | 0.0219 +- 0.0022 | 0.0195 +- 0.0025 | - | 0.21 +- 0.03 | 0.0003 +- 0.0003 | 1862 +- 211 |
| N2 | clean | C0 | 0.0097 +- 0.0010 | 0.0011 +- 0.0001 | 0.14 +- 0.01 | - | 0.0003 +- 0.0002 | 1986 +- 245 |
| N2 | clean | C1-old | 0.0112 +- 0.0014 | 0.0071 +- 0.0017 | 1.83 +- 0.52 | - | 0.3085 +- 0.0310 | 17527 +- 1121 |
| N2 | clean | C1-new | 0.0077 +- 0.0010 | 0.0012 +- 0.0002 | 0.22 +- 0.19 | - | 0.0157 +- 0.0042 | 4072 +- 462 |
| N2 | clean | C1-new+LPF (=C*) | 0.0077 +- 0.0010 | 0.0013 +- 0.0002 | 0.16 +- 0.13 | - | 0.0140 +- 0.0039 | 3896 +- 443 |
| N2 | dist | C0 | 0.0149 +- 0.0014 | 0.0109 +- 0.0017 | - | 0.08 +- 0.01 | 0.0003 +- 0.0002 | 2001 +- 245 |
| N2 | dist | C1-old | 0.0114 +- 0.0015 | 0.0074 +- 0.0018 | - | 0.18 +- 0.07 | 0.3087 +- 0.0309 | 17539 +- 1121 |
| N2 | dist | C1-new | 0.0088 +- 0.0010 | 0.0040 +- 0.0006 | - | 0.02 +- 0.01 | 0.0157 +- 0.0042 | 4088 +- 462 |
| N2 | dist | C1-new+LPF (=C*) | 0.0087 +- 0.0009 | 0.0038 +- 0.0005 | - | 0.01 +- 0.00 | 0.0139 +- 0.0039 | 3912 +- 443 |

## 3. Paired differences (50 trajectories). impr = mean(A-B)/mean(A): positive = B better than A. CI is on RMSE(B)-RMSE(A)

| cond | eval | A -> B | RMSE diff [95% CI] | impr | ss-RMSE impr | energy rel diff |
|---|---|---|---|---|---|---|
| N1 | clean | C0 -> C1-old | +2.0921 [+1.9669, +2.2172] | -1073.1% | -1070.0% | +112.9% |
| N1 | clean | C0 -> C1-new | -0.1845 [-0.2152, -0.1538] | +94.6% | +98.2% | -92.6% |
| N1 | clean | C1-old -> C1-new | -2.2766 [-2.4110, -2.1422] | +99.5% | +99.8% | -96.5% |
| N1 | clean | C1-new -> C1-new+LPF | +0.0006 [+0.0005, +0.0007] | -5.6% | -23.3% | -3.2% |
| N1 | clean | C1-new -> RL(base C1*) | +0.0023 [+0.0019, +0.0027] | -21.7% | -77.1% | -0.2% |
| N1 | clean | C0 -> C1-new | -0.1845 [-0.2152, -0.1538] | +94.6% | +98.2% | -92.6% |
| N1 | clean | C0 -> RL(base C1*) | -0.1823 [-0.2126, -0.1519] | +93.5% | +96.8% | -92.6% |
| N1 | clean | C1-new -> RL(base C1*) | +0.0023 [+0.0019, +0.0027] | -21.7% | -77.1% | -0.2% |
| N1 | dist | C0 -> C1-old | +2.1912 [+2.0609, +2.3215] | -1132.0% | -1129.6% | +117.4% |
| N1 | dist | C0 -> C1-new | -0.1722 [-0.2013, -0.1431] | +89.0% | +90.6% | -92.1% |
| N1 | dist | C1-old -> C1-new | -2.3634 [-2.4984, -2.2284] | +99.1% | +99.2% | -96.4% |
| N1 | dist | C1-new -> C1-new+LPF | +0.0005 [+0.0004, +0.0007] | -2.5% | -2.5% | -1.1% |
| N1 | dist | C1-new -> RL(base C1*) | +0.0006 [+0.0002, +0.0010] | -2.6% | -0.4% | +1.8% |
| N1 | dist | C0 -> C1-new | -0.1722 [-0.2013, -0.1431] | +89.0% | +90.6% | -92.1% |
| N1 | dist | C0 -> RL(base C1*) | -0.1716 [-0.2007, -0.1426] | +88.7% | +90.5% | -91.9% |
| N1 | dist | C1-new -> RL(base C1*) | +0.0006 [+0.0002, +0.0010] | -2.6% | -0.4% | +1.8% |
| N2 | clean | C0 -> C1-old | +0.0015 [-0.0002, +0.0032] | -15.6% | -540.5% | +782.5% |
| N2 | clean | C0 -> C1-new | -0.0020 [-0.0027, -0.0013] | +20.6% | -3.6% | +105.0% |
| N2 | clean | C1-old -> C1-new | -0.0035 [-0.0048, -0.0022] | +31.3% | +83.8% | -76.8% |
| N2 | clean | C1-new -> C1-new+LPF | +0.0000 [-0.0000, +0.0001] | -0.3% | -11.6% | -4.3% |
| N2 | clean | C0 -> C1-new+LPF | -0.0020 [-0.0026, -0.0014] | +20.3% | -15.6% | +96.2% |
| N2 | dist | C0 -> C1-old | -0.0035 [-0.0044, -0.0025] | +23.5% | +32.5% | +776.6% |
| N2 | dist | C0 -> C1-new | -0.0061 [-0.0070, -0.0052] | +41.0% | +63.5% | +104.3% |
| N2 | dist | C1-old -> C1-new | -0.0026 [-0.0037, -0.0015] | +22.8% | +45.9% | -76.7% |
| N2 | dist | C1-new -> C1-new+LPF | -0.0001 [-0.0002, -0.0001] | +1.3% | +5.9% | -4.3% |
| N2 | dist | C0 -> C1-new+LPF | -0.0062 [-0.0071, -0.0053] | +41.8% | +65.7% | +95.5% |

## 4. H7 verdict (rule in H7.md)

WIN cell = RL(base C1*) beats C1* by >= 10% (mean RMSE) with paired CI of (RL - C1*) entirely < 0. Primary = N1 (YES if both N1 cells WIN, PARTIAL if one, else NO); N2 judged the same way (secondary).

| cond | eval | C1* | RMSE C1* | RMSE RL(base C1*) | impr | CI of RL-C1* | cell |
|---|---|---|---|---|---|---|---|
| N1 | clean | C1-new | 0.0104 | 0.0127 | -21.7% | [+0.0019, +0.0027] | RL worse (CI>0) |
| N1 | dist | C1-new | 0.0214 | 0.0219 | -2.6% | [+0.0002, +0.0010] | RL worse (CI>0) |

**H7 primary (N1, noise + delay): NO.  N2 (noise only, secondary): not run.**

Per-seed RL RMSE (RL base C1*, test, clean / dist) for the spread across PPO seeds:

- N1 RL(base C1*): 0.0134/0.0221, 0.0129/0.0227, 0.0129/0.0220, 0.0116/0.0211, 0.0127/0.0217

## 5. Sanity checks and reading notes (added by hand after report4.py)

- Warning rule (>50% win): the only >50% gaps are C0 -> C1-new (+89..95%) and C1-old -> C1-new (+99%) in N1, i.e. classical vs classical. Cause (checked): with a 30 ms actuator delay, C0 (Kp 400, Kd 40, crossover ~40 rad/s) has about 7 deg phase margin; any noise excites a limit cycle (C0 N1 saturation fraction 0.42, calib RMSE 0.014 without noise -> 0.22 with sigma 1e-4), and C1-old (Kp x9.2) diverges even without noise (calib RMSE 2.0). Not an RL-vs-classical gap, no test-set/bound/noise problem found.
- Checks done: (1) eval4.py asserts (q_meas - q_true) == the per-seed noise array to 2e-16, and the noise array depends only on the trajectory seed, so all methods (C0, C1-*, RL and its base) see the same measured stream; (2) run_rl4.py asserts max |RL command - base command| <= 16 Nm (passed for all 5 seeds); (3) test seeds 0-49 were used only by eval4 / the end-of-training eval of run_rl4.py; all tuning used seeds 1000-1049, RL training used seeds >= 500000.
- RL(base C1*) rows come from the end-of-training test evaluation inside run_rl4.py (same evaluate4, same noise); classical rows are evaluated in eval4.py.
- N2 RL not run (see README): the N2 section only has the classical opponents.
