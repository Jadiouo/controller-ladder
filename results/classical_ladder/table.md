# classical ladder: PID, CTC+LQR, iLQR, iLQR-MPC, MPPI (clean nominal plant)

Seeds 5000-5004 (n=5, DEMO set; not calibration/test). Mean +- 95% t-CI over seeds. Tracking error = joint rad (glyph reference); pole error = theta (rad), tip distance is the separate tip_rmse (m). energy = mean sum(tau^2) per executed step, sat = fraction of executed steps. n/a = undefined. iLQR / iLQR-MPC / MPPI see only the NOMINAL model (ctx.make_model()) and the reference through obs.ref_preview.

## Metrics

| scenario | controller | n | rmse | ss_rmse | sat | energy |
|---|---|---|---|---|---|---|
| tracking/clean | pid | 5 | 0.0074 ± 0.0057 | 0.0044 ± 0.0019 | 0.000 ± 0.000 | 176.3 ± 24.5 |
| tracking/clean | ctc_lqr | 5 | 0.0061 ± 0.0058 | 0.0001 ± 0.0000 | 0.000 ± 0.000 | 176.0 ± 24.6 |
| tracking/clean | ilqr | 5 | 0.0056 ± 0.0054 | 0.0000 ± 0.0000 | 0.000 ± 0.000 | 178.1 ± 24.0 |
| tracking/clean | ilqr_mpc | 5 | 0.0056 ± 0.0054 | 0.0000 ± 0.0000 | 0.000 ± 0.000 | 178.1 ± 24.0 |
| tracking/clean | mppi | 5 | 0.0073 ± 0.0053 | 0.0031 ± 0.0003 | 0.000 ± 0.000 | 247.4 ± 25.5 |
| pole_balance/clean | pid | 5 | 0.0050 ± 0.0035 | 0.0007 ± 0.0005 | 0.000 ± 0.000 | 54.0 ± 19.6 |
| pole_balance/clean | ctc_lqr | 5 | 0.0068 ± 0.0039 | 0.0034 ± 0.0025 | 0.000 ± 0.000 | 63.7 ± 3.1 |
| pole_balance/clean | ilqr | 5 | 0.0059 ± 0.0033 | 0.0023 ± 0.0017 | 0.000 ± 0.000 | 64.3 ± 1.3 |
| pole_balance/clean | ilqr_mpc | 5 | 0.0064 ± 0.0040 | 0.0019 ± 0.0012 | 0.000 ± 0.000 | 57.1 ± 10.8 |
| pole_balance/clean | mppi | 5 | 0.0086 ± 0.0028 | 0.0063 ± 0.0018 | 0.000 ± 0.000 | 136.4 ± 3.4 |

## Compute time per control step

Measured around `act()` over all steps of all seeds. wall = elapsed time, cpu = thread CPU time (less sensitive to other jobs: the machine is shared and was heavily loaded, so wall numbers are inflated and noisy; cpu is the better estimate of the cost). The control period is 5 ms. `ilqr` plans the whole episode offline at the first call; that time is NOT in the per-step numbers and is listed in the last column (mean per episode, wall seconds). PID / CTC+LQR timings include only python overhead.

| scenario | controller | steps | wall mean ms | wall p95 ms | cpu mean ms | cpu p95 ms | offline plan s/episode |
|---|---|---|---|---|---|---|---|
| tracking/clean | pid | 4000 | 0.01 | 0.02 | 0.01 | 0.02 | 0.00 |
| tracking/clean | ctc_lqr | 4000 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| tracking/clean | ilqr | 4000 | 0.00 | 0.01 | 0.00 | 0.00 | 0.17 |
| tracking/clean | ilqr_mpc | 4000 | 4.16 | 10.64 | 4.16 | 10.64 | 0.00 |
| tracking/clean | mppi | 4000 | 20.13 | 25.70 | 20.13 | 25.70 | 0.00 |
| pole_balance/clean | pid | 5000 | 0.05 | 0.08 | 0.05 | 0.08 | 0.00 |
| pole_balance/clean | ctc_lqr | 5000 | 0.06 | 0.08 | 0.06 | 0.08 | 0.00 |
| pole_balance/clean | ilqr | 5000 | 0.01 | 0.01 | 0.01 | 0.01 | 0.40 |
| pole_balance/clean | ilqr_mpc | 5000 | 4.50 | 12.51 | 4.50 | 12.51 | 0.00 |
| pole_balance/clean | mppi | 5000 | 22.30 | 28.76 | 22.30 | 28.75 | 0.00 |

## Paired difference vs CTC+LQR (same seeds)

diff = candidate - ctc_lqr; impr. > 0 means the candidate is better; rmse rows compare only seeds both survived.

| scenario | candidate vs ctc_lqr | metric | n | diff [95% CI] | impr. | WIN (CI & >=10%) | note |
|---|---|---|---|---|---|---|---|
| tracking/clean | pid | rmse | 5 | 0.0013 [0.0008, 0.0018] | -21.4% | no | common survivors only (5/5) |
| tracking/clean | pid | ss_rmse | 5 | 0.0043 [0.0024, 0.0062] | -5217.7% | no | common survivors only (5/5) |
| tracking/clean | pid | energy | 5 | 0.2 [-0.0, 0.5] | -0.1% | no |  |
| tracking/clean | ilqr | rmse | 5 | -0.0004 [-0.0008, -0.0000] | +7.1% | no | common survivors only (5/5) |
| tracking/clean | ilqr | ss_rmse | 5 | -0.0000 [-0.0001, -0.0000] | +50.4% | YES | common survivors only (5/5) |
| tracking/clean | ilqr | energy | 5 | 2.1 [-1.0, 5.1] | -1.2% | no |  |
| tracking/clean | ilqr_mpc | rmse | 5 | -0.0004 [-0.0008, -0.0000] | +7.1% | no | common survivors only (5/5) |
| tracking/clean | ilqr_mpc | ss_rmse | 5 | -0.0000 [-0.0001, -0.0000] | +50.4% | YES | common survivors only (5/5) |
| tracking/clean | ilqr_mpc | energy | 5 | 2.1 [-1.0, 5.1] | -1.2% | no |  |
| tracking/clean | mppi | rmse | 5 | 0.0012 [0.0007, 0.0017] | -19.9% | no | common survivors only (5/5) |
| tracking/clean | mppi | ss_rmse | 5 | 0.0030 [0.0028, 0.0033] | -3680.3% | no | common survivors only (5/5) |
| tracking/clean | mppi | energy | 5 | 71.3 [67.8, 74.9] | -40.5% | no |  |
| pole_balance/clean | pid | survival_time | 5 | 0.00 [0.00, 0.00] | +0.0% | no |  |
| pole_balance/clean | pid | rmse | 5 | -0.0018 [-0.0026, -0.0010] | +26.8% | YES | common survivors only (5/5) |
| pole_balance/clean | pid | ss_rmse | 5 | -0.0026 [-0.0050, -0.0003] | +78.0% | YES | common survivors only (5/5) |
| pole_balance/clean | pid | tip_rmse | 5 | 0.0740 [-0.0043, 0.1523] | -390.7% | no | common survivors only (5/5) |
| pole_balance/clean | pid | energy | 5 | -9.7 [-31.0, 11.6] | +15.3% | no |  |
| pole_balance/clean | ilqr | survival_time | 5 | 0.00 [0.00, 0.00] | +0.0% | no |  |
| pole_balance/clean | ilqr | rmse | 5 | -0.0009 [-0.0015, -0.0003] | +12.8% | YES | common survivors only (5/5) |
| pole_balance/clean | ilqr | ss_rmse | 5 | -0.0011 [-0.0020, -0.0002] | +33.1% | YES | common survivors only (5/5) |
| pole_balance/clean | ilqr | tip_rmse | 5 | -0.0014 [-0.0036, 0.0009] | +7.2% | no | common survivors only (5/5) |
| pole_balance/clean | ilqr | energy | 5 | 0.6 [-1.3, 2.5] | -0.9% | no |  |
| pole_balance/clean | ilqr_mpc | survival_time | 5 | 0.00 [0.00, 0.00] | +0.0% | no |  |
| pole_balance/clean | ilqr_mpc | rmse | 5 | -0.0004 [-0.0013, 0.0004] | +6.2% | no | common survivors only (5/5) |
| pole_balance/clean | ilqr_mpc | ss_rmse | 5 | -0.0014 [-0.0031, 0.0003] | +42.8% | no | common survivors only (5/5) |
| pole_balance/clean | ilqr_mpc | tip_rmse | 5 | 0.0640 [-0.0243, 0.1524] | -338.2% | no | common survivors only (5/5) |
| pole_balance/clean | ilqr_mpc | energy | 5 | -6.6 [-14.7, 1.5] | +10.4% | no |  |
| pole_balance/clean | mppi | survival_time | 5 | 0.00 [0.00, 0.00] | +0.0% | no |  |
| pole_balance/clean | mppi | rmse | 5 | 0.0019 [-0.0001, 0.0038] | -27.3% | no | common survivors only (5/5) |
| pole_balance/clean | mppi | ss_rmse | 5 | 0.0029 [0.0013, 0.0045] | -85.6% | no | common survivors only (5/5) |
| pole_balance/clean | mppi | tip_rmse | 5 | 0.0357 [0.0025, 0.0688] | -188.3% | no | common survivors only (5/5) |
| pole_balance/clean | mppi | energy | 5 | 72.6 [68.8, 76.5] | -114.0% | no |  |

Wall-clock of the whole script: 241 s.
