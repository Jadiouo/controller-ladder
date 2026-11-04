# arena demo: PID vs CTC+LQR

Seeds 5000-5019 (n=20, demo set; not calibration/test). Mean +- 95% t-CI over seeds. Tracking error = joint rad; pole error = theta (rad) only, tip distance is the separate tip_rmse (m). energy = mean sum(tau^2) per executed step, sat = fraction of executed steps; early failures are charged a fixed error penalty (README, design decisions). n/a = undefined (no pulses in the scenario); n column = valid seeds; 0/1 metrics use a bootstrap CI.

## Metrics

| scenario | controller | n | rmse | ss_rmse | settle_time | recovery_time | sat | energy | alive_frac | ee_rmse | fell | survival_time | tip_rmse | tip_settle_time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracking/clean | pid | 20 | 0.0112 ± 0.0024 | 0.0043 ± 0.0007 | 0.14 ± 0.02 | n/a | 0.000 ± 0.000 | 179.2 ± 8.1 | 1.00 ± 0.00 | 0.0075 ± 0.0017 |  |  |  |  |
| tracking/clean | ctc_lqr | 20 | 0.0101 ± 0.0025 | 0.0001 ± 0.0000 | 0.13 ± 0.02 | n/a | 0.000 ± 0.000 | 179.6 ± 8.3 | 1.00 ± 0.00 | 0.0065 ± 0.0017 |  |  |  |  |
| tracking/hard (payload+30%, noise, 10ms delay, pulses) | pid | 20 | 0.0144 ± 0.0020 | 0.0103 ± 0.0009 | 2.98 ± 0.29 | 0.16 ± 0.08 | 0.000 ± 0.000 | 254.9 ± 12.1 | 1.00 ± 0.00 | 0.0102 ± 0.0013 |  |  |  |  |
| tracking/hard (payload+30%, noise, 10ms delay, pulses) | ctc_lqr | 20 | 0.0135 ± 0.0018 | 0.0092 ± 0.0008 | 3.00 ± 0.29 | 0.06 ± 0.01 | 0.000 ± 0.000 | 256.7 ± 12.5 | 1.00 ± 0.00 | 0.0087 ± 0.0012 |  |  |  |  |
| pole_balance/clean | pid | 20 | 0.0040 ± 0.0010 | 0.0006 ± 0.0002 | 0.10 ± 0.05 | n/a | 0.000 ± 0.000 | 61.5 ± 7.0 | 1.00 ± 0.00 |  | 0.00 [0.00, 0.00] | 5.00 ± 0.00 | 0.0749 ± 0.0202 | 5.0000 ± 0.0000 |
| pole_balance/clean | ctc_lqr | 20 | 0.0055 ± 0.0012 | 0.0028 ± 0.0007 | 0.09 ± 0.03 | n/a | 0.000 ± 0.000 | 64.9 ± 1.2 | 1.00 ± 0.00 |  | 0.00 [0.00, 0.00] | 5.00 ± 0.00 | 0.0160 ± 0.0040 | 1.0937 ± 0.3945 |
| pole_balance/hard (noise, 5ms delay) | pid | 20 | 0.0042 ± 0.0009 | 0.0016 ± 0.0001 | 0.10 ± 0.05 | n/a | 0.000 ± 0.000 | 65.5 ± 7.2 | 1.00 ± 0.00 |  | 0.00 [0.00, 0.00] | 5.00 ± 0.00 | 0.0748 ± 0.0198 | 5.0000 ± 0.0000 |
| pole_balance/hard (noise, 5ms delay) | ctc_lqr | 20 | 0.0056 ± 0.0011 | 0.0032 ± 0.0007 | 0.09 ± 0.03 | n/a | 0.000 ± 0.000 | 68.2 ± 1.2 | 1.00 ± 0.00 |  | 0.00 [0.00, 0.00] | 5.00 ± 0.00 | 0.0162 ± 0.0039 | 1.1425 ± 0.3792 |

## Paired difference, CTC+LQR vs PID (same seeds)

diff = ctc_lqr - pid; impr. > 0 means CTC+LQR is better; WIN = CI entirely on the better side and >= 10%. rmse rows compare only seeds both controllers survived (n column); survival_time is the primary metric of pole_balance. Holm-corrected decision over all rows of this table in the extra column.

| scenario | candidate vs pid | metric | n | diff [95% CI] | impr. | WIN (CI & >=10%) | WIN (Holm) | note |
|---|---|---|---|---|---|---|---|---|
| tracking/clean | ctc_lqr | rmse | 20 | -0.0011 [-0.0014, -0.0008] | +10.0% | no | no (p=0.0000) | common survivors only (20/20) |
| tracking/clean | ctc_lqr | ss_rmse | 20 | -0.0042 [-0.0049, -0.0035] | +98.1% | YES | YES (p=0.0000) | common survivors only (20/20) |
| tracking/clean | ctc_lqr | energy | 20 | 0.4 [-0.2, 1.0] | -0.2% | no | no (p=0.9905) |  |
| tracking/hard (payload+30%, noise, 10ms delay, pulses) | ctc_lqr | rmse | 20 | -0.0009 [-0.0016, -0.0002] | +6.3% | no | no (p=0.0763) | common survivors only (20/20) |
| tracking/hard (payload+30%, noise, 10ms delay, pulses) | ctc_lqr | ss_rmse | 20 | -0.0012 [-0.0022, -0.0001] | +11.2% | YES | no (p=0.2069) | common survivors only (20/20) |
| tracking/hard (payload+30%, noise, 10ms delay, pulses) | ctc_lqr | energy | 20 | 1.8 [0.4, 3.2] | -0.7% | no | no (p=0.0926) |  |
| pole_balance/clean | ctc_lqr | survival_time | 20 | 0.00 [0.00, 0.00] | +0.0% | no | no (p=1.0000) |  |
| pole_balance/clean | ctc_lqr | rmse | 20 | 0.0015 [0.0012, 0.0018] | -36.4% | no | no (p=0.0000) | common survivors only (20/20) |
| pole_balance/clean | ctc_lqr | ss_rmse | 20 | 0.0022 [0.0016, 0.0029] | -374.5% | no | no (p=0.0000) | common survivors only (20/20) |
| pole_balance/clean | ctc_lqr | tip_rmse | 20 | -0.0589 [-0.0798, -0.0381] | +78.7% | YES | YES (p=0.0001) | common survivors only (20/20) |
| pole_balance/clean | ctc_lqr | energy | 20 | 3.5 [-3.4, 10.3] | -5.6% | no | no (p=1.0000) |  |
| pole_balance/hard (noise, 5ms delay) | ctc_lqr | survival_time | 20 | 0.00 [0.00, 0.00] | +0.0% | no | no (p=1.0000) |  |
| pole_balance/hard (noise, 5ms delay) | ctc_lqr | rmse | 20 | 0.0014 [0.0010, 0.0017] | -32.2% | no | no (p=0.0000) | common survivors only (20/20) |
| pole_balance/hard (noise, 5ms delay) | ctc_lqr | ss_rmse | 20 | 0.0016 [0.0010, 0.0023] | -101.7% | no | no (p=0.0004) | common survivors only (20/20) |
| pole_balance/hard (noise, 5ms delay) | ctc_lqr | tip_rmse | 20 | -0.0586 [-0.0788, -0.0384] | +78.4% | YES | YES (p=0.0001) | common survivors only (20/20) |
| pole_balance/hard (noise, 5ms delay) | ctc_lqr | energy | 20 | 2.7 [-4.4, 9.8] | -4.1% | no | no (p=1.0000) |  |

Wall-clock for the 8 cells x 20 seeds: 25 s.
