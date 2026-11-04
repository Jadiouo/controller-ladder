# Spike01 results (mean ± 95% CI)

Teacher CI: t-interval over the 50 test trajectories (teacher is deterministic). Student CI: t-interval over training seeds (n=10; each seed's value = mean over the same 50 test trajectories). e = nominal payload mass error relative to the true 1.5 kg.


## No disturbance

| e | Teacher RMSE (rad) | BC RMSE | BC+DART RMSE | Teacher energy (Nm²s) | BC energy | BC+DART energy |
|---|---|---|---|---|---|---|
| -30% | 0.0148 ± 0.0010 | 0.0377 ± 0.0085 | 0.0159 ± 0.0002 | 1001.9 ± 145.1 | 995.2 ± 3.7 | 996.9 ± 1.2 |
| +0% | 0.0097 ± 0.0011 | 0.0344 ± 0.0076 | 0.0115 ± 0.0003 | 1007.1 ± 144.9 | 998.8 ± 3.6 | 998.1 ± 1.3 |
| +10% | 0.0101 ± 0.0010 | 0.0312 ± 0.0050 | 0.0118 ± 0.0003 | 1008.7 ± 144.8 | 1000.6 ± 2.9 | 998.2 ± 1.6 |
| +30% | 0.0122 ± 0.0009 | 0.0311 ± 0.0051 | 0.0137 ± 0.0003 | 1012.0 ± 144.6 | 1000.5 ± 2.7 | 997.4 ± 1.8 |
| +100% | 0.0203 ± 0.0014 | 0.0312 ± 0.0029 | 0.0217 ± 0.0004 | 1022.0 ± 143.8 | 996.9 ± 4.1 | 993.4 ± 2.1 |

## With random torque pulses

| e | Teacher RMSE (rad) | BC RMSE | BC+DART RMSE | Teacher energy (Nm²s) | BC energy | BC+DART energy |
|---|---|---|---|---|---|---|
| -30% | 0.0200 ± 0.0015 | 0.1853 ± 0.0704 | 0.0220 ± 0.0005 | 1015.1 ± 144.9 | 1020.9 ± 34.4 | 1010.2 ± 1.7 |
| +0% | 0.0149 ± 0.0014 | 0.1859 ± 0.0608 | 0.0180 ± 0.0007 | 1020.5 ± 144.6 | 1101.8 ± 199.3 | 1011.8 ± 1.4 |
| +10% | 0.0147 ± 0.0013 | 0.1565 ± 0.0376 | 0.0181 ± 0.0008 | 1022.2 ± 144.5 | 1126.3 ± 168.0 | 1012.6 ± 1.9 |
| +30% | 0.0156 ± 0.0012 | 0.1771 ± 0.0571 | 0.0193 ± 0.0010 | 1025.6 ± 144.3 | 1277.6 ± 306.1 | 1014.0 ± 3.3 |
| +100% | 0.0217 ± 0.0014 | 0.0861 ± 0.0164 | 0.0411 ± 0.0321 | 1035.7 ± 143.4 | 1114.4 ± 43.4 | 1126.3 ± 213.7 |

## Paired difference: student RMSE − teacher RMSE (same 50 test trajs), mean ± 95% CI over seeds; and fraction of (seed,traj) pairs where student < teacher

| cond | e | method | diff | frac student better |
|---|---|---|---|---|
| clean | -30% | bc | 0.0229 ± 0.0085 | 0.00 |
| clean | -30% | dart | 0.0011 ± 0.0002 | 0.13 |
| clean | +0% | bc | 0.0247 ± 0.0076 | 0.00 |
| clean | +0% | dart | 0.0018 ± 0.0003 | 0.02 |
| clean | +10% | bc | 0.0211 ± 0.0050 | 0.00 |
| clean | +10% | dart | 0.0017 ± 0.0003 | 0.05 |
| clean | +30% | bc | 0.0189 ± 0.0051 | 0.00 |
| clean | +30% | dart | 0.0015 ± 0.0003 | 0.14 |
| clean | +100% | bc | 0.0108 ± 0.0029 | 0.07 |
| clean | +100% | dart | 0.0013 ± 0.0004 | 0.23 |
| dist | -30% | bc | 0.1652 ± 0.0704 | 0.00 |
| dist | -30% | dart | 0.0020 ± 0.0005 | 0.13 |
| dist | +0% | bc | 0.1710 ± 0.0608 | 0.00 |
| dist | +0% | dart | 0.0031 ± 0.0007 | 0.07 |
| dist | +10% | bc | 0.1418 ± 0.0376 | 0.00 |
| dist | +10% | dart | 0.0034 ± 0.0008 | 0.07 |
| dist | +30% | bc | 0.1615 ± 0.0571 | 0.00 |
| dist | +30% | dart | 0.0037 ± 0.0010 | 0.05 |
| dist | +100% | bc | 0.0644 ± 0.0164 | 0.00 |
| dist | +100% | dart | 0.0193 ± 0.0321 | 0.07 |

## Training diagnostics (mean over seeds): final normalized MSE, rmse of the data-collection rollouts

| e | method | final loss | collection RMSE |
|---|---|---|---|
| -30% | bc | 0.0142 | 0.0153 |
| -30% | dart | 0.0139 | 0.0177 |
| +0% | bc | 0.0230 | 0.0104 |
| +0% | dart | 0.0232 | 0.0128 |
| +10% | bc | 0.0218 | 0.0108 |
| +10% | dart | 0.0282 | 0.0129 |
| +30% | bc | 0.0225 | 0.0129 |
| +30% | dart | 0.0324 | 0.0144 |
| +100% | bc | 0.0302 | 0.0213 |
| +100% | dart | 0.0432 | 0.0219 |
