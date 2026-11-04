# Spike01: can a BC-MLP student of a mismodeled classical teacher beat the teacher?

## Re-run
```
uv venv .venv && uv pip install --python .venv/bin/python mujoco numpy scipy matplotlib
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python spike/sanity.py                       # teacher sanity check (seconds)
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python spike/run_spike.py > results/spike01/run.log
.venv/bin/python spike/report.py                       # writes table.md, rmse_vs_error.png from raw.pkl
```
Files: spike/sim.py (env, refs, teacher, rollout), spike/run_spike.py, spike/report.py, results/spike01/{raw.pkl,run.log,raw_timing.json,table.md,rmse_vs_error.png}.

## Wall-clock (important for cost estimates)
Full run_spike.py (5 e x 10 seeds x {BC, DART} = 100 students + 5 teacher evals): **2974 s (~50 min)** on CPU, 2 torch threads, 1 process.
Breakdown: data collection 506 s, training 1878 s, evaluation (2 conditions x 50 trajs) 577 s, teacher eval 12 s.
Per student ~ 30 s (collect 5 s, train 19 s, eval 6 s). NOTE: machine was heavily loaded by other jobs (load avg 25-40 on 16 cores); with 4 torch threads training was ~10x slower (141 ms vs 13 ms/iter), so I used 2. Unloaded time is probably lower (inference).
An earlier aborted attempt (HID 256, 20 epochs) ran ~160 s/student under load; discarded.

## Setup (all fixed beforehand)
- Plant: MuJoCo 2-link planar arm in the vertical x-z plane, hinge axes y, gravity -9.81 z. Links 0.5 m/1.0 kg and 0.4 m/0.8 kg, joint damping 0.5/0.3, armature 0.01, true tip payload **1.5 kg**, RK4, timestep 2.5 ms, **control 200 Hz** (2 substeps, ZOH), torque limit ±80 Nm (applied to teacher and students). Episode 4 s = 800 steps.
- Reference: per joint center + 2 sines (amp sum 0.2-0.6 rad, 0.2-1.0 Hz, random phase); joint1 within [-1.2,1.2], joint2 within [0.3,2.3] rad. Random initial error: pos U(±0.15) rad, vel U(±0.3) rad/s.
- Metrics (fixed): RMSE = sqrt(mean over time and 2 joints of (q-qref)^2) per trajectory (includes the initial-error transient; step 0 counts); energy = sum |tau|^2 * dt (Nm^2 s) of the executed torque (dominated by gravity holding, ~1000).
- Teacher: computed torque using MuJoCo inverse dynamics (mj_inverse) on a **nominal model identical except payload mass = 1.5*(1+e)**, desired acc = qdd_ref + Kp*e + Kd*edot with per-joint double-integrator LQR (Q=diag(1.6e5, 800), R=1 -> Kp=400, Kd~40). e in {-30,0,+10,+30,+100}% (+100% added by me because mismatch effects were small). Teacher runs in the TRUE model.
- Student: MLP 10 -> 128 -> 128 -> 128 -> 2 (ReLU), inputs q, qd, qref, qdref, qddref (standardized), output torque (standardized). Adam lr 1e-3 cosine, batch 1024, 20 epochs, MSE. **200 training trajectories (160k samples) per student**, new training refs per seed (seeds 100000+1000*seed+i; disjoint from test).
- BC: teacher rollouts. BC+DART: Gaussian noise sigma=3 Nm (white, per joint) added to the executed torque, label = clean teacher action. DART sigma was not tuned. DAgger: not done (optional, skipped).
- Eval: **same 50 test trajectories (seeds 0-49), same initial errors, same disturbances** for teacher and every student. Disturbance: 3 torque pulses/episode, 0.1 s, 4-8 Nm, random joint/sign/time. 10 training seeds per (e, method). CI: t-interval over seeds for students; over the 50 test trajectories for the (deterministic) teacher, so the two CIs mean different things.

## Findings
- Sanity: e=0, zero initial error: teacher RMSE 1.7e-4 rad. With the random initial error: 0.0097 rad (this is the transient from the initial error, not tracking failure). OK.
- **The student never beat the teacher in mean RMSE in any (e, condition)**, 5 e x 2 conditions x 2 methods. Plain BC is far worse (2-4x clean; ~8-12x with pulses, high seed variance). BC+DART is close: +8-20% RMSE clean, +20-25% with pulses (e=+100% with pulses: noisy, 0.041 +- 0.032, a seed outlier). Student wins on individual trajectories only 0-23% of the time.
- Student energy is ~1% below the teacher's mean energy for clean runs, but the teacher's CI is across trajectories (+-145) so this is not a meaningful gap (paired comparison not computed for energy).
- The teacher is hardly hurt by payload error (RMSE 0.0097 -> 0.0148 at -30%, 0.0203 at +100%) because the LQR gain is high and the payload (<=1.5 kg) mismatch is modest relative to the loop gain. So there is little for a student to "fix". Student also only sees teacher actions (no reward), so by construction it can at best approximate the teacher; beating it would require something that is not just imitation.
- No warning-sign "student much better than teacher" observed. Checks anyway: test set identical (same seeds, asserted by construction in one script), train seeds disjoint from test, teacher implementation verified by the e=0 sanity check, rollout code shared between teacher and student.

## Anomalies / caveats
- Large RMSE variance across BC seeds; BC under pulses is fragile (covariate shift), DART fixes most of it.
- DART at e=+100% with pulses has one or more bad seeds (CI wide); not investigated.
- torch threads 2 instead of 4 (see above); epochs/width were reduced for time, so BC students might improve with longer training (not tested). The BC-vs-DART gap may partly be an undertraining effect (final normalized MSE ~0.01-0.04).
- Single plant/task; payload mismatch only (no friction/damping mismatch, no unmodeled dynamics beyond mass).
