# spike05: H8 (unknown, per-episode random plant error) on ball juggling, cap N=300

Numbers and verdict: `table.md`. Clip: `juggle_h8.mp4` (T-adapt left, R-DR-hist seed 0 right, test episode 5: e=0.81, ball x3.4, delay 45 ms, first 40 s).

## Verdict
**H8 refuted.** Test medians (50 episodes): T-nom 8 [5,11], T-robust 300 [300,300], T-adapt 300 [300,300], R-DR 9 [7,12] (3 seeds), R-DR-hist 3 [3,4] (3 seeds). R-DR-hist vs best classical: -99% [-99,-99] (rule: >= +30% and CI excluding 0). Both classical controllers reach the 300 cap in 50/50 test episodes (and 50/50 calibration episodes), so the cap is a ceiling: *no* learner could have shown a >= 30% gain at N=300. The refutation is therefore "classical saturates the task", plus RL (as trained here) is far below it. Inference: with this RL budget and setup, learning adds nothing; I cannot say what a much better-tuned RL would do, but it could at most tie at the cap.

## Re-run (repo root, existing .venv; every heavy job wrapped in flock + systemd-run MemoryMax=6G; CPU only)
```
.venv/bin/python experiments/spike05/zeta_map5.py                    # (already folded into zeta_map5.json: only zeta=0.05 -> e=0.86 kept, see deviations)
experiments/spike05/run_calib.sh                                     # CMA-ES T-robust + T-adapt, calibration seeds only -> robust.json adapt.json, calib_*.log
experiments/spike05/run_rl_all.sh                                    # calib_final.py, then residual PPO: dr and hist, seeds 0,1,2, 3M steps -> raw_rl_*.pkl, run_rl_*.log
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python experiments/spike05/eval_all.py   # LOCKED test set -> test_raw.pkl
.venv/bin/python experiments/spike05/report5.py                      # table.md
(flock ... ) .venv/bin/python experiments/spike05/video5.py T-adapt 0  # juggle_h8.mp4
```
Code: `env5.py` (env, parametrised teacher, T-adapt estimator, history), `cma.py`, `calib_classical.py`, `run_rl5.py`, `eval_all.py`, `report5.py`, `video5.py`. Reuses `experiments/spike03b/env.py` unmodified.

## Setup
- Plant and teacher as spike03b (3-DoF arm, mirror law, alpha-beta filter, CTC; residual cap 12 Nm = 20% of 60 Nm, asserted: max observed 12.00 Nm). Changes: hit cap N=300, episode limit 300 s.
- Per-episode error (drawn from the episode seed, identical for all methods; verified identical across all methods' stored params): restitution e ~ U[0.40,0.85] (via zeta map), **ball mass x U[1,5]** (0.05-0.25 kg), sensor delay U[0,150] ms rounded to 5 ms steps, 4 mm ball-position noise. The controller is never told these.
- Seeds: calibration 1000-1049 (CMA only; `assert seeds.min()>=1000`), RL training >= 500000 (initial condition and plant error from the seed), test 0-49 evaluated once, after everything was frozen.

## Methods and hyperparameters
- **T-nom**: spike03b teacher (kappa_max .4, e_m .8, kE 1.0, Kx 1.1, vxmax 1.2, kp 900, kd 60, a .5, b .1, phimax .35, no extrapolation).
- **T-robust**: CMA-ES (own 60-line implementation, mu/mu_w, popsize 10, sigma0 0.2 in a unit hypercube) over 11 params: kappa_max, kE, Kx, vxmax, kp, kd, pred (extrapolation steps), e_m, filter a, b, phimax. Fitness on calibration = mean min(hits,120)/120 - 0.1*mean min(apex_err,0.5). Result (robust.json): kappa_max .89, e_m .42, kE .99, Kx .67, vxmax .5, kp 943, kd 29, a .44, b .20, phimax .24, pred 8.5 steps.
- **T-adapt**: same teacher, but e_m and the extrapolation are online estimates. Impact detection from encoders (joint velocity jump not explained by commanded acceleration, threshold `thr`); at each impact: ballistic least-squares fit of the delayed noisy ball samples before impact -> incoming velocity and the delay (time the fitted trajectory reaches the racket vs the encoder-detected impact time); ballistic fit after impact -> outgoing velocity; e_obs = -(v_out - v_r).n / (v_in - v_r).n (racket velocity from encoders, plate normal); recursive averaging e_hat += max(bmin_e, 1/(n+n0e))*(e_obs - e_hat), same for delay. e_hat goes into the kappa and tilt formulas; extrapolation = c0 + cd*d_hat steps. CMA-ES over 16 params (the 9 shared + e0, n0e, bmin_e, d0, cd, c0, thr), same objective/budget style. Result (adapt.json): kappa_max .77, kE .96, Kx 1.56, kp 1473, kd 102, a .31, b .10, phimax .52, e0 .74, n0e 4.3, bmin_e .22, d0 12, cd .88, c0 -2.6, thr .107. Estimator quality on the test episodes: corr(delay_hat, true delay)=1.00; corr(e_hat, e)=0.73, mean abs error 0.17 (e_hat is the *effective* restitution of the whole racket+arm, biased low vs the nominal e; it is what the mirror law needs).
- **R-DR**: residual PPO on T-nom, spike03b observation (23-d), spike03b PPO settings (gamma .995, lambda .95, lr 3e-4, 8 epochs, minibatch 2048, clip .2, 32 envs x 256 horizon, log-std init -2, reward +1/hit, -3|apex err|, -3 failure, -2e-6|res|^2), DR over the full test distribution. Network tanh 256x256 (spike03b used 128x128; widened for both variants so capacity is not a confound).
- **R-DR-hist**: identical, plus history stacking (no GRU): 16 lags with stride 16 control steps (80 ms; 1.28 s span) of 11 features (ball estimate x,z,vx,vz, kappa, tracking error q-qr (3), previous residual action (3)); input 23+176=199-d.
- Budget parity for RL: 3,000,000 env steps per seed, 3 seeds each (the cap is "3-5 seeds"; chosen after the smoke test projected >8 h for more).

## Three-axis budget
| | sim steps (control steps, 5 ms) | evaluations / updates | wall clock | free parameters |
|---|---|---|---|---|
| T-robust CMA | 43.6M (71 evals x 50 episodes, cap 120) | 71 | 3850 s (2 workers, concurrent with T-adapt, loaded machine) | 11 |
| T-adapt CMA | 40.2M (61 evals) | 61 | 3904 s (2 workers) | 16 |
| R-DR (per seed) | 3.0M | 366 PPO iterations | 2088 / 1758 / 1199 s | ~75k net weights |
| R-DR-hist (per seed) | 3.0M | 366 | 903 / 434 / 406 s | ~120k net weights |
RL total 18M steps; classical total 84M steps (classical used MORE simulation than RL, because each fitness evaluation replays 50 episodes). Machine load average was 60-90 throughout, so wall clock is inflated and the shared lock added ~1-2 h of waiting (not counted). Test evaluation of all 9 runs: ~200 s.

## Calibration-set check (calib_final.json, N=300): T-nom median 5, T-robust 300 (50/50 at cap), T-adapt 300 (50/50). No visible overfit to the 50 calibration episodes: test 50/50 at cap too.

## Red-flag checks (classical >> RL by far)
Test seeds identical for all methods and parameters identical across methods (checked in test_raw.pkl); RL residual cap asserted <= 12 Nm; CMA only touches seeds >= 1000 (asserted); the classical controllers use only encoders and delayed noisy ball position (no true e/mass/delay); reward irrelevant to classical. T-nom vs T-robust (8 vs 300) is a >3x gap but explained: nominal parameters cannot survive e<0.6 / delay>75 ms (spike03b's own result) and the CMA solution is fitted to exactly this distribution.

## Deviations, assumptions, limitations
1. "Ball mass" implemented literally as ball mass x U[1,5] (spike03b's "mass" condition scaled the *arm* masses). Heavy-ball effects reach the arm only through contact impulses.
2. Zeta map: the spike03b map ends at e=0.82; the drop test at zeta 0.01-0.04 was erratic (e measured 0.4-14), so only zeta=0.05 (e=0.86) was added; e in (0.82,0.85] is mapped by interpolation to zeta .05-.06. Plant e (and also the effective e at juggling speeds, +-0.05) is approximate.
3. CMA-ES budget was cut short (stopped by hand at ~60-70 evaluations, 6-7 generations, because the shared lock and load made each evaluation ~55 s; fitness had already saturated near 0.99 with cap 120). Robust 71 vs adapt 61 evaluations (unequal by one generation). T-adapt tuned from its own default start, not warm-started from T-robust. Fitness cap 120 < 300; the result holds at 300 (calib and test).
4. Tie: T-robust and T-adapt are both at the cap; the verdict compares against T-robust only because the table picks the "stronger" by test median (tie -> first). Against T-adapt the numbers are identical. Secondary metrics favour T-adapt (apex error 0.042 vs 0.124 m, x-drift 0.016 vs 0.028 m), and the clip uses T-adapt.
5. **RL is probably under-tuned**: one hyperparameter setting (spike03b's), no sweep, no GRU, no curriculum, residual on T-nom only (not on T-robust), 3 seeds. Training-episode medians stayed at 3-6 hits, i.e. near the base teacher (T-nom median 5-8); R-DR-hist is *worse* than T-nom (median 3 vs 8, CI [3,4]), which I did not diagnose (could be history features hurting a barely-trained policy, or a bug in the history path; unverified). A residual-on-T-robust variant would answer "does RL add anything on top of strong classical", but at a 300-hit cap it can at best tie.
6. Cap ceiling: both classical controllers saturate, so differences above 300 hits are invisible; no episode-length extension was run. Torque energy of the classical controllers is ~3.5x that of T-nom/RL (582-633 vs ~180 Nm^2), their cost for robustness.
7. Video episode rule (fixed before inspection): of the episodes where seed-median R-DR-hist and the classical hit counts differ, the one whose difference is closest to the median difference (episode 5). Only seed 0 shown. I did not watch the clip frame by frame.
8. Process: no review sub-agent loop was run (single-session, user absent); `.scratch/smoke5_*.pkl` are smoke artefacts.
