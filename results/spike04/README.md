# spike04: does the tuned classical controller (C1) still tie residual RL once the simulator has sensor noise + delay? (H7)

Verdict and all numbers: `table.md`. Pre-registration: `H7.md` (written before any run; Amendment 1 appended before any noise result). Code in `spike04/` (imports spike/, spike02/, spike03a/ unchanged; nothing else modified). Outputs here.

## Result in one paragraph
**H7 = NO** (N1 = noise sigma 1e-3 rad + 30 ms delay). C1-new (re-tuned) has RMSE 0.0104 (clean) / 0.0214 (dist); residual PPO on top of it (5 seeds x 2M steps) has 0.0127 / 0.0219, i.e. RL is 21.7% WORSE clean (CI of RL-C1 [+0.0019,+0.0027]) and 2.6% worse with pulses (CI [+0.0002,+0.0010]). The premise was only half right: noise/delay do punish the spike03a gains (C1-old diverges under delay; in N2 noise-only C1-old loses 39% vs no noise and the re-tuned Kp falls from x9.2 to x3.7; in N1 to x0.76), but the re-tuned classical controller still ties or beats RL. RL has no gain that classical retuning cannot reach in this task, at least in N1 with this RL recipe. N2 (noise only) RL was not run (see deviations), so for N2 only the classical result is available.

## Re-run (repo root)
```
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$H .venv/bin/python spike04/calib_sigma.py            # sigma_q from calibration seeds 1000-1049 -> calib_sigma.json (then set json["sigma"]=0.001, see deviation 1)
$H .venv/bin/python spike04/tune4.py N2               # CMA-ES C1-new and C1-new+LPF, calibration only -> tuned_N2.json
$H .venv/bin/python spike04/tune4.py N1               # -> tuned_N1.json
for s in 0 1 2 3 4; do $H .venv/bin/python spike04/run_rl4.py --cond N1 --base C1STAR --seeds $s --steps 2000000; done   # -> raw_rl_N1_C1STAR.pkl
$H .venv/bin/python spike04/eval4.py                  # locked test seeds 0-49 -> eval.pkl
.venv/bin/python spike04/report4.py                   # -> table.md (sections 0-4)
```
torch 2 threads, CPU only. `--base C0` and `--cond N2` are supported by run_rl4.py but were not run.

## What was done
- Plant = spike03a W0 (nominal true plant, cap 80 Nm, no friction). N1 adds the W2 actuator delay (30 ms = 6 control steps); N2 has no delay. Sensor delay is NOT modelled (actuator side only, as in W2).
- Noise: i.i.d. Gaussian sigma_q = 1e-3 rad on measured joint angles; measured velocity = finite difference of the noisy angles over one 5 ms step (sigma_qd = 0.283 rad/s, correlated). References are noise-free. One fixed noise array per trajectory seed (seed + 11,000,000) shared by all methods; metrics use the true state; RL reward uses the true state.
- sigma calibration (calibration seeds only, `calib_sigma.log/json`, table.md section 0): grid 1e-4..5e-3; smallest sigma where C1-old (spike03a gains, Kp x9.24, Kd x4.43) RMSE >= 1.3x its noise-free value in clean and dist = 1e-3 (1.39x / 1.38x). 5e-4 gives only 1.05x.
- Opponents: C0; C1-old; C1-new (CMA-ES, own implementation copied from spike03a, pop 8, 10 generations, box [0.2,100], objective 0.5*RMSE_clean + 0.5*RMSE_dist, calibration seeds, 80 evals); C1-new+LPF (3rd parameter = first-order LPF cutoff on measured q and qd, box [1,100] Hz, start 30 Hz, 14 generations, 112 evals). C* = lower calibration objective: N1 -> C1-new (0.01687 vs 0.01743), N2 -> C1-new+LPF (0.00947 vs 0.00952, a tie; LPF cutoff 67 Hz is nearly a no-op). The LPF did not help in either condition (in N1 it is slightly worse on test).
- Residual PPO: spike02 code and hyperparameters unchanged (obs features, 16 Nm bound, 2M steps, 64 envs, horizon 200, no tuning), trained in N1 with the noisy observation, base = C1-new (fixed). 5 seeds.
- Test: seeds 0-49, clean and pulses; metrics as spike03a; paired 95% t-CI.

## Three-axis budget (N1)
| axis | C1-new | C1-new+LPF | residual PPO (base C1-new) |
|---|---|---|---|
| free DoF | 2 | 3 | ~19k actor weights, 0 tuned hyperparameters (spike02 recipe reused) |
| simulation steps | 6.4 M (80 evals) | 9.0 M (112 evals) | 2 M per policy, 10 M for 5 seeds |
| wall-clock | 291 s | 409 s | 578-753 s per seed (machine shared and loaded) |
Extra: sigma calibration 93 s; N2 tuning 261 s + 398 s. Calibration for classical used only seeds 1000-1049. Classical used 15.4 M sim steps in N1 vs RL 10 M, so the budget is not in RL's favour on the step axis, and RL is ~4 orders of magnitude larger in parameters. Total elapsed about 6.5 h, of which the large majority was queueing on the shared heavy.lock; compute was about 1.5 h.

## Deviations from the design (honest list)
1. sigma rule: the original rule required the 1.3x criterion in both N1 and N2. In N1 the delay alone makes C1-old diverge (calib RMSE 2.03 without noise), so the criterion was meaningless; decided before any noisy result and written in H7.md Amendment 1: sigma from N2 only. The `sigma` field in calib_sigma.json was patched by hand from null to 0.001 (script's own output was null; `sigma_N2_only` = 0.001).
2. Scope reduction: a smoke run (51k steps took 36 s on the loaded machine, i.e. ~25 min per 2M-step seed, 20 runs ~ 8 h > 6 h) triggered your rule, so: only N1 got RL, only with base C1-new; no RL(base C0); no RL in N2. N2 tuning and classical evaluation were done (cheap). Later actual training time was ~10-12 min per seed, so the full design would probably have fitted compute-wise but not queue-wise. N2 RL / base C0 runs were queued, then cancelled for time.
3. The first launch of RL seed 2 produced no output and no result (killed silently, cause unknown); it was rerun with the same seed/command. Seeds 0-4 all present in the final pkl.
4. C* picking, per-cell WIN rule, YES/PARTIAL/NO aggregation: my operationalisation, all in H7.md before runs.
5. One CMA-ES seed per stage; one noise realisation per trajectory; one sigma (no sweep over sigma for H7).
6. RL test numbers are the end-of-training evaluation inside run_rl4.py, not a re-evaluation in eval4.py.
7. Joint-angle noise is the only sensor model; velocity noise derived by differencing (the other option, independent velocity noise, not run).

## Limitations / reading carefully
- The negative result is for this RL recipe: spike02's features multiply the noisy errors by 20 (position) and 3 (velocity) and the MLP has no memory or filtering, so the policy sees raw differenced velocity (0.28 rad/s noise). Likely reason it loses clean RMSE vs its base (inference: its residual partly reacts to noise; not tested). An RL with frame stacking / filtered inputs / tuned hyperparameters or longer training could behave differently, and the result does NOT show that no learned controller could beat C1 under noise+delay.
- In N1 the dominant effect is the delay (C0 near-unstable, C1-old diverges); noise strength matters little there (C0 RMSE ~0.22 for any sigma <= 2e-3). So N1 mainly tests "delay-limited gain", and sigma=1e-3 is not what drives its result.
- Only N1 has RL; the noise-only claim for RL is untested. N2 classical-only results: tuning cut Kp from x9.2 to x3.7 and still beat C0 by 21% (clean) / 41% (dist); C1-old (untouched) is 16% worse than C0 clean (energy +783%, saturation 31%) but better on dist.
- RMSE-tuned classical in N1 has settle 0.36 s vs RL 1.00 s (clean) and recovery 0.16 vs 0.21 s; RL better only in saturation fraction (0.0003 vs 0.0008) and not in energy.
- Test set: 50 trajectories, paired t-CI; differences of 2.6% on dist are significant but tiny.
