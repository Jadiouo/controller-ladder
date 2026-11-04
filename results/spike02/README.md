# spike02: does residual RL on a classical teacher with un-modelled weaknesses beat the teacher?

## Re-run (from repo root; spike01 code in experiments/spike/ untouched, spike02 imports experiments/spike/sim.py)
```
.venv/bin/python experiments/spike02/calib.py                  # weakness calibration, teacher only
.venv/bin/python experiments/spike02/teacher_table.py          # -> teacher_check.md (teacher vs W0 on test set)
experiments/spike02/run_all.sh                                 # BC+DART/DAgger (10 seeds) then residual PPO (5 seeds) for W0..W4, one process at a time, systemd-run MemoryMax=6G, 2 torch threads, CPU only
.venv/bin/python experiments/spike02/report.py                 # -> table.md, rmse_by_W.png
# ablation (no torque pulses in RL training, + fresh eval seeds 2000-2049), 3 seeds:
.venv/bin/python experiments/spike02/run_rl.py --w W0 --seeds 0,1,2 --steps 2000000 --no_pulses --fresh_eval --out results/spike02/abl_nopulse_W0.pkl   (same for W1)
```
Files: experiments/spike02/{sim2,bc_lib,run_bc,run_rl,report,calib,teacher_table}.py, run_all.sh; results: table.md (all numbers + H1-H4 verdicts), rmse_by_W.png, teacher_check.md, calib.log, raw_*.pkl, run_*.log.
Sanity: W0 teacher results are bit-identical to spike01 e=0 (checked, max abs diff 0.0 for RMSE and energy).

## Setup
- Plant/teacher/refs/test set/disturbance: exactly spike01 (teacher = CTC+LQR, e=0 nominal model, gains not retuned, never given knowledge of the weakness). Test = seeds 0-49, clean and with 3 torque pulses; same set for every method. Training seeds disjoint (BC 100000+, RL 500000+, calibration 1000-1049, fresh-eval ablation 2000-2049).
- Weaknesses (true plant only; teacher model unchanged). Calibrated on teacher only (calib.py, seeds 1000-1049; criterion teacher RMSE >= 1.3x W0 in both conditions):
  - W1 torque cap **32 Nm** (instead of 80). First chose 27 (x2.0 on calib seeds) but on the test seeds 5/50 teacher runs diverged (cliff edge), so raised to 32 after a teacher-only look at the test seeds. Saturation time fraction of the teacher on the test set: 1.6% (clean).
  - W2 actuator delay **30 ms** (6 control steps). The specified 10 ms did not hurt the teacher (x0.94 RMSE), 20-25 ms neither (x0.90 at 25 ms), 30 ms gives x1.45; 40 ms makes the teacher unstable.
  - W3 Coulomb 1.5 Nm (tanh(qd/0.05) smoothing) + extra viscous 0.3 Nm s/rad on both joints (0.5/0.3 -> total damping stays on top of the model's own).
  - W4 = all three (cap 32, delay 30 ms, friction 1.5/0.3).
  - Teacher RMSE on test (clean / dist, x of W0): W0 0.0097/0.0149; W1 0.0217 (2.24x)/0.0268 (1.80x); W2 0.0141 (1.45x)/0.0219 (1.47x); W3 0.0219 (2.25x)/0.0237 (1.59x); W4 0.0394 (4.05x)/0.0416 (2.79x). See teacher_check.md.
- BC+DART: spike01 settings (MLP 10-128-128-128-2, Adam 1e-3 cosine, batch 1024, 20 epochs, 200 trajectories, DART sigma 3 Nm, no tuning), teacher rollouts collected in the weak plant. 10 seeds. Plain BC not rerun (spike01 covers it).
- DAgger: 5 rounds x 40 trajectories = 200 trajectories total (same count as BC); round 0 teacher only (20 epochs), round 1 beta=0.5, rounds 2-4 beta=0 (student acts), labels = clean teacher action, aggregate dataset, warm-started net, 10 epochs per round, normaliser fixed after round 0. 10 seeds.
- Residual PPO (own implementation, torch, 2 threads): action = clip(teacher torque + 16 Nm * clip(a,-1,1), +-80), plant then applies cap/delay/friction. Residual bound 16 Nm = 20% of the 80 Nm nominal limit, same for all W (verified in code: assert max |cmd - teacher| <= 16 on every eval; held). Observation (16-d) = spike01's 10 inputs + teacher action (2) + position/velocity error (4, added by me as convenience features, deviation). Reward = -200*|q-qref|^2 (sum of joints, after the step) - 1e-4*|residual|^2. Training episodes: fresh random refs/init errors each batch, half of them with torque pulses. 64 parallel envs, horizon 200, gamma .99, lambda .95, lr 3e-4, 10 epochs, minibatch 2048, clip .2, net tanh 128x128 actor and critic, log-std init -1.5. **Budget: 2,000,000 env steps per (W, seed)** (=2500 episodes), 5 seeds per W. Hyperparameters were not tuned (only one 256k-step smoke run on W1 before the main run). BC-type students use 160k samples (200 trajectories) so RL gets ~12x more interactions; this is the fairness-card "state the steps" item and a real asymmetry.
- Statistics: per trajectory, student RMSE averaged over seeds, minus teacher RMSE; 95% t-CI over the 50 paired trajectories; "win" = CI < 0 and improvement >= 10%. I additionally report seed-level CIs (seed-CI-hi) as robustness.

## Wall-clock (loaded machine, load avg 25-50)
Calibration + smoke tests ~10 min. BC+DART+DAgger 10 seeds each: 7.3-10 min per W (about 43 min total). Residual PPO 5 seeds: 12.5-17 min per W (about 70 min total; ~2.5-3.5 min per seed). Ablation 6 runs ~20 min. Total pipeline ~2.5 h wall-clock, well under the 4 h limit so the full 5-W matrix was run.

## Results in one paragraph
Full numbers and verdicts: table.md. BC+DART and DAgger never win (best +12.9% for DART in W4-clean but CI includes 0; DAgger max +1.3%). Residual PPO beats the teacher in every W under both conditions (CI<0), with improvements of 17/45% (W0 clean/dist), 37/42% (W1), 12/10% (W2), 63/65% (W3), 31/26% (W4); significant-by-card wins in 9 of 10 cells (W1-clean CI touches 0 because 2 teacher-diverging trajectories give huge variance). H1 supported, H2 supported (but see next), **H3 falsified** (RL wins 17%/45% with no weakness), H4 supported.

## Why the H2 "support" must not be read as "weaknesses are what lets the student win" (inference)
- RL reaches almost the same RMSE (~0.0081 clean, ~0.0082 dist) in W0 and in W3, i.e. it fully removes the friction penalty but also beats the teacher where there is no weakness. In W0 the teacher's RMSE is mostly the initial-error transient plus pulse rejection; the fixed LQR/CTC gain is not RMSE-optimal and the 16 Nm residual acts as extra learned feedback. So much of the RL gain comes from "teacher is a fixed finite-gain controller that was not tuned for this metric", not from an un-modelled weakness. The disturbance pulses themselves are an un-modelled effect that the teacher has no mechanism to reject.
- The weakness-attributable part is better judged relative to the W0 RL floor: W3 (friction): RL recovers entirely to the W0 level; W1/W2/W4: RL does not return to the W0 floor (0.0136 / 0.0123 / 0.0270 vs 0.0081).
- Red-flag checks (RL > 50% in W3, 45% in W0-dist): (1) reward uses only q, qref and residual size, no test information; (2) test seeds 0-49 disjoint from RL training seeds (500000+), same test set/disturbances for all methods, teacher W0 identical to spike01; (3) residual bound verified on every evaluation; (4) ablation: RL trained **without any pulses** (3 seeds) still gets W0 dist RMSE 0.0084 vs teacher 0.0149 (43.7%), and on **fresh unseen trajectories (seeds 2000-2049)** improvements are W0 17.3%/43.5%, W1 19.8%/29.8% (clean/dist) - so no train-on-test or pulse-memorisation explanation. 5 RL seeds agree to within ~1% RMSE.
- Not checked: whether a simply re-tuned/stiffer teacher (e.g. higher Kp, or adding an integrator) would match RL. This is the key missing control: the "win" is relative to a teacher whose gains were fixed at spike01 values by protocol, so it may just show that the gain choice was suboptimal. Recommend a "tuned-classical" baseline before claiming a paper thesis.

## Anomalies / limitations
- One plant, one task family, one teacher; RL and BC hyperparameters not equivalently treated (RL 12x more interaction data, extra error features in RL obs).
- W1 teacher has a heavy tail (2/50 test trajectories with RMSE > 0.1 because torque cap barely exceeds the gravity-holding need); W1-clean mean-RMSE CI for RL therefore touches 0, although all 5 seeds and 72% of trajectories improve. W4 similar (3/50).
- DART at W1/W4 appears to "improve" in some seeds (seed-win 1.0 in W4) but the trajectory-level CI includes 0 and improvement <10%/13% - not a win by the card; within the noise of 2-3 diverging trajectories.
- W2 energy: RL halves the teacher's torque energy (-46%); the teacher with delay oscillates (chatter), RL damps it.
- Calibration chose weakness sizes after seeing teacher results (not students), W1 cap adjusted after a teacher-only look at the test seeds; delay raised from the specified 10 ms to 30 ms because 10 ms did not hurt.
- Saturation fraction is the fraction of control steps in which any joint command hit the cap.
