# spike09b TEST results (whip, seeds 0-49, single locked run). best-classical = T-adapt (selected on holdout 1100-1599)

## Main table (multi-seed methods: per-test-seed mean over the 3 training seeds, then statistics over the 50 test seeds)

| method | success % [95% CI] | closest approach (m) | hit speed (m/s, hits only) | energy (Nm^2/step) | residual norm (Nm, mean over swing) |
|---|---|---|---|---|---|
| T-nom | 52.0 [38.0, 66.0] | 0.1032 | 9.11 | 464.8 | - |
| T-robust | 68.0 [54.0, 80.0] | 0.0975 | 9.56 | 581.2 | - |
| T-adapt (= best-classical) | 98.0 [94.0, 100.0] | 0.0600 | 9.35 | 1508.2 | - |
| T-oracle | 100.0 [100.0, 100.0] | 0.0558 | 11.02 | 637.0 | - |
| L-est | 82.7 [76.7, 88.7] | 0.0746 | 9.28 | 1353.7 | - |
| R-PPO-hist | 89.3 [83.3, 94.7] | 0.0680 | 9.49 | 1475.2 | 0.475 |
| R-oracle | 86.0 [77.3, 93.3] | 0.0767 | 9.43 | 1469.0 | 0.433 |

Per training seed (success %):

| method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| L-est | 86.0 | 82.0 | 80.0 |
| R-PPO-hist | 92.0 | 88.0 | 88.0 |
| R-oracle | 86.0 | 84.0 | 88.0 |

## Paired differences vs best-classical (percentage points; paired percentile bootstrap over the 50 TEST seeds, 20000 resamples)

| method | diff pp | 95% CI | p (boot) |
|---|---|---|---|
| T-nom | -46.0 | [-60.0, -32.0] | 0.0001 |
| T-robust | -30.0 | [-44.0, -16.0] | 0.0001 |
| T-adapt | 0.0 | [0.0, 0.0] | 1.0000 |
| T-oracle | 2.0 | [0.0, 6.0] | 0.7250 |
| L-est | -15.3 | [-22.0, -8.7] | 0.0001 |
| R-PPO-hist | -8.7 | [-15.3, -2.7] | 0.0041 |
| R-oracle | -12.0 | [-20.7, -4.7] | 0.0001 |

### H9w (headroom): T-oracle or R-oracle vs best-classical (Holm-strict)

| candidate | diff pp | 95% CI | p | Holm p | >=10 pp and CI > 0 and Holm p < .05 |
|---|---|---|---|---|---|
| T-oracle | 2.0 | [0.0, 6.0] | 0.7250 | 0.7250 | no |
| R-oracle | -12.0 | [-20.7, -4.7] | 0.0001 | 0.0002 | no |

### H9w spec-literal (no Holm requirement)

| candidate | diff pp | 95% CI | p | Holm p | >=10 pp and CI > 0 and Holm p < .05 |
|---|---|---|---|---|---|
| T-oracle | 2.0 | [0.0, 6.0] | 0.7250 | 0.7250 | no |
| R-oracle | -12.0 | [-20.7, -4.7] | 0.0001 | 0.0002 | no |

### H10w (fair learning): L-est or R-PPO-hist vs best-classical

| candidate | diff pp | 95% CI | p | Holm p | >=10 pp and CI > 0 and Holm p < .05 |
|---|---|---|---|---|---|
| L-est | -15.3 | [-22.0, -8.7] | 0.0001 | 0.0001 | no |
| R-PPO-hist | -8.7 | [-15.3, -2.7] | 0.0041 | 0.0041 | no |

**H9w: strict NOT SUPPORTED, spec-literal NOT SUPPORTED.  H10w: REFUTED (neither reaches +10 pp with CI > 0).**

Ceiling note: best-classical TEST success = 98.0 %, so the largest conceivable gain is 2.0 pp; a +10 pp win is arithmetically impossible on TEST.

### Secondary: L-est (learned estimator) vs T-adapt (model fit): -15.3 pp, 95% CI [-22.0, -8.7], p = 0.0001

## Success % by true chain-length error quartile (n per cell in header; ~12 each, wide CIs)

| method | len -20..-10 % (n=13) | len -10..0 % (n=8) | len 0..+10 % (n=14) | len +10..+20 % (n=15) |
|---|---|---|---|---|
| T-nom | 46.2 | 75.0 | 64.3 | 33.3 |
| T-robust | 69.2 | 25.0 | 85.7 | 73.3 |
| T-adapt | 100.0 | 100.0 | 92.9 | 100.0 |
| T-oracle | 100.0 | 100.0 | 100.0 | 100.0 |
| L-est | 89.7 | 75.0 | 88.1 | 75.6 |
| R-PPO-hist | 97.4 | 91.7 | 83.3 | 86.7 |
| R-oracle | 97.4 | 87.5 | 73.8 | 86.7 |

## "Did it learn?" check (CAL 1000-1049: selected checkpoint minus initialisation, need >= +5 pp)

| run | init CAL succ | selected ckpt CAL succ | diff pp | learned | residual norm at selected ckpt (Nm) |
|---|---|---|---|---|---|
| R-PPO-hist#0 | 100.0 | 86.0 (ckpt it 209) | -14.0 | NOT LEARNED | 0.472 |
| R-PPO-hist#1 | 100.0 | 90.0 (ckpt it 209) | -10.0 | NOT LEARNED | 0.649 |
| R-PPO-hist#2 | 100.0 | 88.0 (ckpt it 52) | -12.0 | NOT LEARNED | 0.294 |
| R-oracle#0 | 100.0 | 94.0 (ckpt it 209) | -6.0 | NOT LEARNED | 0.502 |
| R-oracle#1 | 100.0 | 86.0 (ckpt it 52) | -14.0 | NOT LEARNED | 0.341 |
| R-oracle#2 | 100.0 | 84.0 (ckpt it 105) | -16.0 | NOT LEARNED | 0.443 |

## Holdout (1100-1599) used for best-classical selection

| method | success % | CI |
|---|---|---|
| T-nom | 56.8 | [52.4, 61.2] |
| T-robust | 56.6 | [52.2, 61.0] |
| T-adapt | 96.8 | [95.2, 98.2] |
| T-oracle | 98.8 | [97.8, 99.6] |

## Budget (episodes)
```
{
 "classical_reference": {
  "T-robust episodes (spike09a)": 9050,
  "T-adapt design/tuning episodes on CAL": 1300,
  "note": "~1100 episodes of exploratory probe/estimator design on CAL seeds (p4-p9 scripts, not all logged) + 200 logged in adapt_probe_cal.json; total ~1300 (estimate)",
  "total": 10350,
  "2x": 20700
 },
 "learners": {
  "R-PPO-hist": {
   "0": {
    "train_episodes": 9942,
    "cal_eval_episodes": 400,
    "total": 10342,
    "steps": 2998272,
    "wall_s": 465.43836307525635,
    "init_cal": 1.0,
    "selected_cal": 0.86,
    "learned": false,
    "res_mean_selected": 0.4718080110499951,
    "ckpt_cal": [
     0.82,
     0.84,
     0.76,
     0.86,
     0.82,
     0.82,
     0.84
    ]
   },
   "1": {
    "train_episodes": 9940,
    "cal_eval_episodes": 400,
    "total": 10340,
    "steps": 2998272,
    "wall_s": 466.4117524623871,
    "init_cal": 1.0,
    "selected_cal": 0.9,
    "learned": false,
    "res_mean_selected": 0.6493691412403635,
    "ckpt_cal": [
     0.84,
     0.84,
     0.82,
     0.9,
     0.86,
     0.78,
     0.86
    ]
   },
   "2": {
    "train_episodes": 9886,
    "cal_eval_episodes": 400,
    "total": 10286,
    "steps": 2998272,
    "wall_s": 484.6600558757782,
    "init_cal": 1.0,
    "selected_cal": 0.88,
    "learned": false,
    "res_mean_selected": 0.294379560876555,
    "ckpt_cal": [
     0.88,
     0.82,
     0.8,
     0.78,
     0.84,
     0.84,
     0.86
    ]
   }
  },
  "R-oracle": {
   "0": {
    "train_episodes": 9959,
    "cal_eval_episodes": 400,
    "total": 10359,
    "steps": 2998272,
    "wall_s": 506.932914018631,
    "init_cal": 1.0,
    "selected_cal": 0.94,
    "learned": false,
    "res_mean_selected": 0.5023104113183017,
    "ckpt_cal": [
     0.88,
     0.82,
     0.84,
     0.94,
     0.84,
     0.76,
     0.76
    ]
   },
   "1": {
    "train_episodes": 9923,
    "cal_eval_episodes": 400,
    "total": 10323,
    "steps": 2998272,
    "wall_s": 535.3387141227722,
    "init_cal": 1.0,
    "selected_cal": 0.86,
    "learned": false,
    "res_mean_selected": 0.34075603961122425,
    "ckpt_cal": [
     0.86,
     0.82,
     0.84,
     0.78,
     0.78,
     0.82,
     0.78
    ]
   },
   "2": {
    "train_episodes": 9843,
    "cal_eval_episodes": 400,
    "total": 10243,
    "steps": 2998272,
    "wall_s": 488.52821159362793,
    "init_cal": 1.0,
    "selected_cal": 0.84,
    "learned": false,
    "res_mean_selected": 0.4426334929214659,
    "ckpt_cal": [
     0.82,
     0.84,
     0.84,
     0.8,
     0.84,
     0.78,
     0.78
    ]
   }
  },
  "L-est": {
   "probe_only_episodes": 12400,
   "probe_len_s": 0.2,
   "full_episode_equivalent": 992.0,
   "wall_s": 143.00129079818726,
   "per_model": "3 models share the same data; 3 training seeds differ in init only",
   "mlp_params": 97539
  }
 },
 "free_params": {
  "T-robust": 5,
  "T-adapt": "3 fitted per episode + probe design (T, amp) =2 chosen on CAL",
  "L-est": 97539,
  "R-PPO": "actor+critic ~167936",
  "PPO deviation": "3M steps/run (spec cap: 2x classical); no hyperparameter sweep"
 }
}
```