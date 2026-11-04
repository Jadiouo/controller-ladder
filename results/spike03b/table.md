# spike03b table: mirror-law teacher vs students on ball juggling (locked test set, 50 episodes, hits capped at 100)


## nominal: Cond(nominal: e=0.8, m x1.0, sigma=0.2mm, delay=0ms)

| method | n seeds | median hits [95% CI] | mean hits | rel. median vs teacher [95% CI, paired bootstrap] | seed-level rel. medians | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |
|---|---|---|---|---|---|---|---|---|
| teacher | 1 | 100 [100, 100] | 100.0 | - | - | 0.057 | 0.0015 | 123 |

## restitution: Cond(e=0.4: e=0.4, m x1.0, sigma=0.2mm, delay=0ms)

| method | n seeds | median hits [95% CI] | mean hits | rel. median vs teacher [95% CI, paired bootstrap] | seed-level rel. medians | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |
|---|---|---|---|---|---|---|---|---|
| teacher | 1 | 26 [23, 26] | 25.0 | - | - | 0.438 | 0.0075 | 113 |
| BC+DART | 10 | 25 [25, 26] | 25.1 | -2% [-4, +9] | -10%, +2%, -2%, +2%, -10%, -6%, +2%, -2%, -4%, +10% | 0.436 | 0.0102 | 114 |
| residual PPO | 3 | 100 [100, 100] | 96.7 | +292% [+285, +335] | +292%, +292%, +292% | 0.159 | 0.1503 | 148 |
| CONTROL: classically re-tuned teacher {"E_NOM": 0.8, "KAPPA_MAX": 0.8} | 1 | 100 [100, 100] | 100.0 | +292% [+285, +335] | +292% | 0.319 | nan | 168 |

residual bound check: max |cmd - teacher| over all RL evals = 12.00 Nm (limit 12). RL train steps/seed: 3000000; train wall s: [225, 231, 290]

## mass: Cond(mass x5.0: e=0.8, m x5.0, sigma=0.2mm, delay=0ms)

| method | n seeds | median hits [95% CI] | mean hits | rel. median vs teacher [95% CI, paired bootstrap] | seed-level rel. medians | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |
|---|---|---|---|---|---|---|---|---|
| teacher | 1 | 2 [2, 100] | 45.1 | - | - | 0.083 | 0.1846 | 2764 |
| BC+DART | 10 | 2 [2, 6] | 31.2 | -20% [-97, +20] | -20%, +20%, -20%, +120%, +340%, +0%, -20%, -20%, -20%, -20% | 0.098 | 0.1788 | 2758 |
| residual PPO | 3 | 100 [100, 100] | 61.7 | +3900% [+0, +4900] | -20%, +3900%, +3900% | 0.084 | 0.1998 | 1928 |
| CONTROL: classically re-tuned teacher {"MODEL_MS": 2.0} | 1 | 100 [100, 100] | 100.0 | +3900% [+0, +4900] | +3900% | 0.037 | nan | 2159 |

residual bound check: max |cmd - teacher| over all RL evals = 12.00 Nm (limit 12). RL train steps/seed: 3000000; train wall s: [459, 1056, 895]

## sensing: Cond(sigma 4mm + delay 150ms: e=0.8, m x1.0, sigma=4.0mm, delay=150ms)

| method | n seeds | median hits [95% CI] | mean hits | rel. median vs teacher [95% CI, paired bootstrap] | seed-level rel. medians | apex err (m) | hit x-drift (m) | torque energy (Nm^2) |
|---|---|---|---|---|---|---|---|---|
| teacher | 1 | 20 [18, 22] | 19.8 | - | - | 0.392 | 0.0166 | 154 |
| BC+DART | 10 | 18 [17, 19] | 19.7 | -8% [-18, +3] | +0%, -13%, -10%, -13%, +23%, -8%, +3%, -8%, -13%, -8% | 0.398 | 0.0195 | 153 |
| residual PPO | 3 | 8 [7, 10] | 13.4 | -59% [-66, -50] | -90%, -59%, +5% | 0.271 | 0.1678 | 169 |
| CONTROL: classically re-tuned teacher {"pred": 10} | 1 | 100 [100, 100] | 100.0 | +413% [+355, +471] | +413% | 0.132 | nan | 223 |

residual bound check: max |cmd - teacher| over all RL evals = 12.00 Nm (limit 12). RL train steps/seed: 3000000; train wall s: [518, 790, 900]

## H6 verdict

- restitution: teacher median 26 -> residual PPO median 100 (+292%, paired CI [+285%, +335%]); per-seed [292, 292, 292]%. Point estimate >= +30%: YES; CI lower bound > 0: YES.
- mass: teacher median 2 -> residual PPO median 100 (+3900%, paired CI [+0%, +4900%]); per-seed [-20, 3900, 3900]%. Point estimate >= +30%: YES; CI lower bound > 0: no.
- sensing: teacher median 20 -> residual PPO median 8 (-59%, paired CI [-66%, -50%]); per-seed [-90, -59, 5]%. Point estimate >= +30%: no; CI lower bound > 0: no.

**H6 SUPPORTED** (rule: in at least one error condition residual-PPO median >= teacher median x 1.3, and the paired-bootstrap CI of the relative gain excludes 0). Conditions meeting it: ['restitution'].