# spike03a: tuned classical opponents (C0/C1/C2) vs residual PPO on the 2-DoF tracking task (H5), plus H2 re-judged under v2

Verdicts and all numbers: `table.md`. Hypothesis card (v2) untouched; spike01/spike02 code and results untouched; code in `spike03a/`, outputs here.

## Re-run (repo root; each heavy step wrapped as below, CPU only, 2 torch threads)
```
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$H .venv/bin/python spike03a/calib03.py          # step 1: W1 cap / W2 delay on calibration seeds 1000-1049 -> calib_W.json
$H .venv/bin/python spike03a/tune.py W0 W3       # steps 2-3: CMA-ES for C1, C2 on calibration seeds -> tuned_W0_W3.json, tune.log
$H .venv/bin/python spike03a/evaluate.py         # step 4: test seeds 0-49 -> eval.pkl (RL re-evaluated from spike02's saved actor weights; matches spike02 stored RMSE bit-for-bit, max diff 0.0)
.venv/bin/python spike03a/report.py              # -> table.md
```
Files: `spike03a/{lib,calib03,tune,evaluate,report}.py`; results `calib_W.{json,log}`, `tune.log`, `tune_run1_bounds10x_aborted.log`, `tuned_W0_W3.json`, `eval.{pkl,log}`, `table.md`.

## What was done
- Plant, references, init errors, pulses, test set, residual bound 16 Nm, teacher model: exactly spike02 (imports `spike/sim.py`, `spike02/sim2.py`).
- C0 = spike02 teacher (CTC with exact nominal model + LQR Kp=400, Kd from `lqr_gain()`), untouched.
- C1 = same CTC, Kp and Kd scaled (2 free params), tuned per W (W0 and W3 separately, like the RL which was trained per W) on calibration seeds 1000-1049 only. Objective = 0.5 * mean RMSE (clean) + 0.5 * mean RMSE (3 pulses), transient included (same as the main metric).
- C2 = C1-structure plus integral of tracking error (conditional-integration anti-windup); the extra torque (integral contribution) is clamped to +-16 Nm (= RL residual bound); 3 free params (Kp scale, Kd scale, Ki) re-tuned jointly, starting from the C1 optimum. Disturbance observer NOT done (integral only, the card allows either).
- Optimiser: own small CMA-ES (the `cma` package is not in the venv; I did not install anything), popsize 8; C1 10 generations, C2 14 generations; parameter box Kp,Kd scale in [0.2, 100], Ki in [10, 1e6].
- Test metrics: RMSE, steady-state RMSE (t>0.5 s), settling time (first time after which max-joint error stays < 0.02 rad, clean only, 4 s = never), pulse recovery time (per pulse: time from pulse end until error stays < 0.02 rad within <=1 s or the next pulse, mean over pulses, 0 if never out of band), saturation fraction, torque energy. Paired 95% t-CI over the 50 trajectories; RL = mean over its 5 PPO seeds per trajectory.

## Three-axis budget (per W; W0 and W3 are the same size)
| axis | C1 | C2 (on top of C1) | residual PPO (spike02) |
|---|---|---|---|
| free degrees of freedom | 2 | 3 (+ the 2 gains re-tuned) | ~19k actor weights (+ ~19k critic); hyperparameters not tuned (0 tuning rounds, one 256k smoke run) |
| simulation steps (plant control steps, 5 ms) | 6.4 M (80 evals x 80k) | 9.0 M (112 evals x 80k) | 2 M per policy, 10 M for the 5 seeds |
| wall-clock (loaded machine, load avg ~26) | ~205 s | ~315 s | ~150-170 s per seed, ~13-14 min for 5 seeds |
| evaluations / tuning rounds | 80 | 112 | 1 per seed, no tuning |
Extra, not in the table: the aborted first tuning run (see below) used ~6.4 M + a partial C2 stage of W0 steps; the W1/W2 re-calibration used 26 evals (~2.1 M steps). Total classical budget incl. both W and the aborted run is about 39 M steps, i.e. the classical side used MORE simulator steps than one RL policy (about 7.7x for C1+C2 in one W) but similar wall-clock to the RL 5-seed bundle, with ~4 orders of magnitude fewer free parameters. Calibration used a 50-trajectory set with 2 conditions per eval; the test set was never used for any tuning decision in this spike.

## Post-hoc / deviation list (honest)
1. Parameter box for the gain scales was first [0.2, 10]; the W0 C1 optimum hit the Kp upper bound (x10.0). I aborted that run (log kept: `tune_run1_bounds10x_aborted.log`) and restarted with [0.2, 100] to avoid under-tuning the classical side. Decision made from calibration numbers only, before any test evaluation. Optimum then was interior (Kp x9.2, Kd x4.4); C2 found x21-23 and x10 (still far from the 100 bound).
2. W1 and W2 new strengths only recorded; students not re-run, so W1/W2 are not evaluated here, and the H2 re-judgement uses the spike02 W1 (cap 32 Nm) and W2 (30 ms) data.
3. H5 per-cell rule and the H2 v2 threshold (extra >= 10% of Teacher_Wk, CI > 0) are my operationalisations of the card; the card gives the formula but no threshold for H2 v2. Both were written into `report.py` before `eval.pkl` existed.
4. RL was not retrained; spike02 policies were re-evaluated for the new metrics (verified identical RMSE).
5. Settling threshold 0.02 rad and the 1 s recovery window are my choices (not in the card).
6. Own CMA-ES instead of the `cma` package; only one optimiser seed per stage (no repeat), so the tuned values are one draw.

## Limitations / things to read carefully
- **C2's low RMSE comes with chatter**: Kp x21-23 with the 200 Hz loop drives saturation to 8% (W0) and 51% (W3) of steps and torque energy 13x/35x of C0 (the RMSE objective has no effort term). C1 achieves the same RMSE with sat 0.4-0.6% and energy +17-24% vs C0, and is the practically meaningful classical opponent; RL's own energy is ~C0's (+1%). So "tuned classical ties/beats RL on RMSE" holds, but only C1 does so at comparable effort; C2's integral term adds essentially nothing over C1 (C1->C2 within 2% in all cells), i.e. C2 mostly = C1 with even higher gain.
- Comparison is on RMSE-tuned classical vs RL tuned on its own reward (tracking error plus a tiny residual penalty); effort was not constrained for either side.
- RL has 12x the interaction data of BC but also no tuning, and uses extra error features; a better-trained RL might close the 6-8% gap (inference, untested). The residual bound (16 Nm) and a 200 Hz loop limit what either side can do.
- One plant, one task family; the W3 weakness (friction) is the kind of thing a stiffer loop absorbs, so W3 says little about weaknesses that gain cannot fix (cap, delay were not run).
- With only the CMA-ES single seed, the 5-6% differences between C1/C2 and RL (CI touching 0) are not resolved by this spike.
