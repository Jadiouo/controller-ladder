# spike09b: H9w / H10w on the whip task (TEST seeds 0-49, run once)

Code: `experiments/spike09b/` (c9b.py shared; step1a_probe_select, step1b_holdout, step3_lest, step4a_pool, train9, freeze9b, eval9b, report9b, video9b, verify_repro). Numbers: `table.md`. Video: `whip_test_episode.mp4` (T-nom | T-adapt = best-classical | R-PPO-hist#0, TEST seed 12; episode rule in video9b.py, fixed before TEST: among seeds where T-nom fails and best-classical succeeds, the one whose min-distance gap is closest to the median).

## Result
- best-classical = **T-adapt** (holdout 1100-1599, n=500: 96.8% vs T-nom 56.8%, T-robust 56.6%; T-oracle 98.8%).
- TEST (n=50): T-adapt 98%, T-oracle 100% (+2 pp, CI [0,6]), R-oracle 86% (-12), R-PPO-hist 89% (-8.7), L-est 83% (-15). **H9w not supported, H10w refuted** (spec-literal and Holm-strict agree).
- Ceiling: best-classical is at 98%, so a +10 pp gain is arithmetically impossible. The result is a ceiling effect, not evidence that learning fails in general (inference).
- No PPO run passed the "learned" check (CAL selected checkpoint minus init >= +5 pp): all 6 are NOT LEARNED (init = exactly zero residual = 100% CAL; selected 84-94%). Residual norm 0.3-0.65 Nm mean; the learned residual made things worse.

## Methods
- Task/plant: spike09a dist (w=0.2; chain length +-20%, mass x0.37-2.7, damping x0.135-7.4, uniform over links; drawn from the seed). Success = chain tip within 8 cm.
- T-nom, T-robust: spike09a parameters (planner on assumed chain + gains).
- **T-adapt** (classical system identification): 0.2 s probe (CTC on a small bump, amplitude [0.15,0.15,0.12] rad, returns to rest), probe time counts inside the 2.5 s episode. Estimator: grid + nonlinear least squares (progressive prefixes) of (length scale, mass scale, damping scale) so that an open-loop replay of the recorded torques on a nominal-model chain matches the measured (q, qd). The estimated chain is then replayed over the probe to reconstruct the chain's full state, and the CMA-ES/closed-loop planner plans from that state with 2.3 s left. Probe length/amplitude chosen on CAL (4 configs, 200 episodes; `adapt_probe_cal.json`).
- T-oracle: planner gets true (ls, ms, ds) at t=0, no probe. R-oracle: PPO residual on best-classical, observation also has the true parameters.
- **L-est**: MLP (2x256 tanh, 97.5k params) trained on 12,000 probe-only episodes (training seeds >= 500000) mapping the probe history (measured minus nominal-chain replay, 120 features) to (ls, ms, ds); hyper-parameters (hidden, weight decay, epochs) chosen only on CAL regression error; 3 training seeds. Then the same state reconstruction + planner as T-adapt. Only difference from T-adapt: the estimator.
- **R-PPO / R-oracle**: PPO (spike07 recipe: log-std0 -3.5, floor -5, lr 1e-4, 4 epochs, 256x256, batch 32x256), residual cap 12 Nm, active only after the probe; obs = state, target, base torque, tracking error, time + 8 lagged (q, qd) history (hist) (+ true parameters for oracle). Reward = 10 x decrease of running-min tip distance + 5 success - 5 failure (privileged tip position only in the reward). 1,200 fixed training scenarios (target seed 500000+i, independent chain draw). 7 equally spaced checkpoints evaluated on CAL; best chosen (init not a candidate). 3 training seeds each.

## Budget (three axes; details in `budget.json`, table.md)
- Episodes: classical reference = T-robust 9,050 + T-adapt design ~1,300 (estimate; ~1,100 exploratory CAL episodes were not individually logged) = ~10,350; 2x = 20,700. Each PPO run: ~9,900 training + 400 CAL-eval (+50 init) = ~10,300 episodes: within budget. L-est: 12,400 probe-only episodes of 0.2 s each (~990 full-episode equivalents); 12,400 < 20,700 even counted as plain episodes.
- Wall-clock: PPO ~8 min per run (8 workers, CPU); L-est ~2.4 min; T-robust 70 min; T-adapt ~1.2 s fit + ~4 s planning per episode.
- Free parameters: T-robust 5; T-adapt 3 fitted per episode + 2 probe design choices; L-est ~97.5k; PPO actor+critic ~0.17M each-ish.

## Deviations and limitations (read these)
1. T-adapt reconstructs the chain state by model replay and plans from it. This goes beyond "least-squares parameter fit, hand to planner", and was needed: without it (planner assuming a resting chain) T-adapt scored only 40-80% on CAL because the probe leaves the chain swinging. L-est uses the same pipeline.
2. The simulator is noise-free and the plant lies exactly in the estimator's model class, so system identification is essentially exact (log-error < 1e-6). T-adapt therefore ~ T-oracle. This strongly favours the classical side; with sensor noise or unmodelled structure the picture could differ (not tested).
3. T-adapt and L-est are given the support (box) of the error distribution as bounds; T-nom is not.
4. The whip is hypersensitive: iid action noise of 0.036 Nm/step already drops T-adapt from 100% to 68% on CAL, so any exploration noise hurts; PPO with sigma = 0.36 Nm trains at ~78% success. "Not learned" may partly reflect this exploration cost (inference).
5. Residual is switched off during the probe; PPO output layer zero-initialised (a random-init residual of 0.1 Nm alone cost 38 pp on CAL); 3 M steps per run (not 4 M); one hyper-parameter setting, no sweep.
6. One diagnostic round (CAL only, not in TEST): PPO-hist on weak base T-nom (CAL 56%), 1 seed: checkpoints CAL 52-62% (selected 62%, +6 pp, but the other six are 52-58% and the run's training success went 58%->63%); regarded as noise level, so even with headroom this recipe did not clearly learn.
7. H9w also reported without the Holm requirement (spec-literal); same verdict. Multi-seed methods are summarised as per-test-seed mean over the 3 training seeds (pre-specified in report9b.py).
8. L-est/PPO checkpoints and hold-out choices were frozen (`final_methods.json`, md5s) before TEST; `verify_repro.json`: T-adapt reproduced exactly on 16 holdout seeds with a fresh cache. Adapt-tag estimate cache keys hash the probe history; a prior review (sonnet) found no TEST leak.
9. n=50 TEST seeds; quartile cells have n=8-15 (descriptive only).
