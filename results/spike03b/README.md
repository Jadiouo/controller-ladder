# spike03b: ball juggling pilot for H6 (mirror-law teacher vs residual RL)

Full numbers and the H6 verdict: `table.md`. Clip: `juggle_restitution.mp4` / `.gif` (teacher left, teacher+residual RL right, same test episode, restitution e=0.4, first 30 s).

## Re-run (repo root, existing .venv, CPU only, 2 torch threads)
```
.venv/bin/python spike03b/sanity.py --write       # contact sanity check + zeta_map.json  (log: sanity.log; plain run does not overwrite the map)
.venv/bin/python spike03b/tune_teacher.py         # one-off teacher tuning on nominal (tune_teacher.log)
.venv/bin/python spike03b/calib.py                # round 1 (calib_round1.log); run again = round 2 for families with no hit (calib_round2.log) -> calib.json
spike03b/run_all.sh                               # BC+DART x10 seeds, residual PPO x3 seeds per condition, teacher-retune control, report (each heavy job wrapped in flock + systemd-run MemoryMax=6G)
.venv/bin/python spike03b/video.py restitution 0  # clip
.venv/bin/python spike03b/check_tilt.py           # racket tilt / offset at ball contact, teacher vs student (check_tilt.log)
```
Code: `env.py` (plant, teacher, metrics), `stu.py`, `run_bc.py`, `run_rl.py`, `ctrl_teacher.py`, `report.py`.

## Environment
- MuJoCo 3.17, dt 1 ms (implicitfast), control 200 Hz (5 substeps). Vertical-plane 3-DoF arm: link 0.5 m (1.0 kg), link 0.5 m (0.8 kg), racket plate = wrist body (box 0.24 m x 0.016 m, 0.3 kg, centre at the wrist, normal = plate normal, wrist angle = racket tilt). Joint damping 0.5/0.3/0.1, armature 0.01/0.01/0.002, torque limit 60 Nm per joint. One ball (radius 0.03, 0.05 kg) with two slide joints (x, z): no spin, planar.
- Contact (only ball-racket collide; arm links have contype 0): condim 1 (frictionless normal contact), solref = (timeconst 0.02 s, damping ratio zeta), default solimp, zeta set per desired restitution e via `zeta_map.json` (built by the drop test). Timeconst 0.005 s first tried: restitution was erratic (0-90x, solver stiffness vs 1 ms step), so 0.02 s.
- **Sanity check (sanity.log)**: ball dropped on a fixed racket from 0.1/0.25/0.5/1.0 m (impact 1.4-4.4 m/s); measured rebound velocity ratio vs set e: e=0.8 -> 0.83/0.85/0.78/0.88; e=0.7 -> 0.71/0.72/0.69/0.74; e=0.6 -> 0.61/0.62/0.60/0.65; e=0.5 -> 0.50/0.52/0.50/0.54; e=0.4 -> 0.38/0.42/0.40/0.45. Height ratio is its square (e.g. 0.60 for e=0.8 at 0.5 m). Within ~±0.05 of set e; at the operating impact speed (3.1 m/s, 0.5 m drop) the nominal plant has e=0.78 vs the teacher's 0.8 (a small unmodelled mismatch present in every condition).
- Episode: ball dropped 0.2-0.5 m above the racket, x offset U(-0.08,0.08) m, vx U(-0.3,0.3). Hit = ball vertical velocity flips from down to up within 0.12 m of the plate. Episode ends at: 100 hits (cap N=100), ball >0.3 m below racket ('dropped'), lateral |x-0.6|>0.5 m ('lateral'), ball lying on racket 0.3 s ('dead'), 100 s ('timeout'). Target: apex 0.5 m above the contact level at x*=0.6 m.
- Test set: seeds 0-49 (initial condition + sensor noise sequence), identical for all methods; calibration seeds 1000-1049; teacher tuning seeds 3000-3019; BC data seeds 100000+; RL training seeds 500000+.

## Teacher
Mirror law on the estimated ball state (alpha-beta filter, a=0.5, b=0.1, on noisy, delayed ball position): racket height z_r = -kappa * z_b', racket vertical velocity -kappa * v_z, kappa = clip((1-e_m)/(1+e_m) + kE (E*-E)/g, 0, 0.4) (energy feedback on ball specific energy E), racket x tracks the ball x, racket tilt chosen so the reflected ball gets vx = clip(Kx (x*-x), +-1.2) (model e_m=0.8 in the tilt formula). Executed through analytic IK + Jacobian velocity feedforward + computed-torque (inverse dynamics of the nominal model, Kp 900, Kd 60). Tuned once on nominal (tune_teacher.log; kE in {0.3,0.5,1.0,1.5} x Kx in {0.7,1.1,1.6}; chose kE=1.0, Kx=1.1 for lowest apex error, all had 100/100 hits). **Nominal result: median 100 hits (cap) on the test set, apex error 0.057 m, x-drift 1.5 mm (table.md, nominal block). >=50 requirement met.**

## Error conditions (true plant only; teacher model unchanged) and calibration
Rule: smallest grid intensity at which the teacher median hits on the calibration seeds <= 50% of nominal (100). Teacher only; no student involved; test set not used for choosing.
- restitution: e=0.8 -> **0.4** (e=0.7...0.44 all 100; 0.42 -> 54 (just misses the rule); 0.40 -> 26). Round 1 grid (0.7-0.5) had no hit, round 2 extended it (calib_round*.log).
- mass: all links+racket mass x **5.0** (x4 -> 100, x5 -> median 2, cliff). Large and not physically small; failure is gravity-compensation/torque-limit collapse. 
- sensing: ball position noise 4 mm + delay **150 ms** (120 ms -> 100, 150 -> 18). Noise alone (up to 15 mm) did not hurt the teacher. Delay (not noise) is what the rule picked.
- Test-set teacher medians: restitution 25.5 (-74%), mass 2.5 (-97.5%), sensing 19.5 (-80%).

## Students (pilot scale)
Both get the same 23-d observation: joint q, qd, teacher's references, filtered ball state, teacher torque, kappa, tracking error (the teacher's own estimator output; the student does not see raw sensor data otherwise).
- BC+DART: MLP 23-128x3-3 ReLU, Adam 1e-3 cosine, 30 epochs, batch 1024; 60 teacher rollouts (<=1500 steps each, ~34k samples) in the TRUE (error) plant with N(0, 2 Nm) torque noise, clean-teacher labels; 10 seeds (different data and init). No tuning.
- Residual PPO (own implementation): action = clip(tau_teacher + 12 Nm * clip(a,-1,1), +-60) (12 Nm = 20% of the 60 Nm limit; asserted on every evaluation, observed max exactly 12.00). Reward: +1 per hit, -3*|apex error| at each apex, -3 on failure, -2e-6*|residual|^2; no test information. 3 seeds, **3,000,000 env steps each**, 32 envs x 256 horizon, gamma .995, lambda .95, lr 3e-4, 8 epochs, minibatch 2048, clip .2, tanh 128x128, log-std init -2.0 (initial -1.0 wrecked the teacher with exploration noise, found in a 200k smoke run). Hyperparameter tuning: that one change after 2 smoke runs. BC gets ~34k samples vs RL 3M steps (real asymmetry).
- Evaluation: locked test set, paired with the teacher on the same episodes. Median over all (seed, episode) pairs; 95% CI by cluster bootstrap over the 50 episodes (4000 resamples), relative gain bootstrapped paired.

## Results (details table.md)
| condition | teacher median | BC+DART | residual PPO | control: re-tuned teacher |
|---|---|---|---|---|
| nominal | 100 | - | - | - |
| restitution e=0.4 | 26 [23,26] | 25 (-2%) | 100 (+292%, CI [+285,+335]), 3/3 seeds | 100 (kappa_max 0.4->0.8) |
| mass x5 | 2 [2,100] | 2 | 100 (2 of 3 seeds; one seed -20%); CI [0,+4900] | 100 (model mass x2) |
| sensing 150 ms | 20 [18,22] | 18 (-8%) | 8 (-59%) CI [-66,-50] | 100 (extrapolate ball state 50 ms) |
Medians of 100 are the cap (censored), so CIs like [100,100] and ratios such as +3900% are cap artefacts.
**H6 (pre-set rule: >=+30% median in at least one condition, plus paired CI excluding 0): met in the restitution condition (+292%) -> "supported" by the letter.** Mass: point estimate met but CI includes 0 and seed-dependent. Sensing: student worse than teacher.

## Must-read caveats (the headline is NOT "RL beats a good classical controller")
1. **Control (ctrl_teacher.json/log)**: a one-parameter classical re-tune chosen on the calibration seeds (kappa_max 0.4->0.8, or CTC model mass x2, or ball-state extrapolation by 50 ms) brings the teacher back to median 100 in ALL three conditions. The student's win is over a teacher whose racket-speed clamp / model parameter was left as-is by protocol; the same pattern as spike02's "key missing control". This control was added by me after seeing results; the re-tune grid was small and chosen per condition from the failure mechanism (so a hand-picked retune, but cheap).
2. Failures of the teacher are cliffs (restitution 0.44->0.40, mass x4->x5, delay 120->150 ms): kappa clamp 0.4 is a design parameter (racket speed limit) I fixed before calibration; with a larger clamp the restitution error would never have bitten at 0.4. The calibration rule therefore partly measures where my own clamp breaks.
3. >3x red flag checked: reward uses only hits/apex/failure; test seeds (0-49) disjoint from training (500000+); residual cap asserted (12.00 Nm); same test set for teacher and students (paired). `check_tilt.log`: at contact the student's racket tilt median 0.09 rad (90% 0.44; teacher 0.0005) and ball offset from plate centre <2.1 cm, so it hits with the flat plate, but the clip shows a transient pose (racket swung vertical between hits, ball high), and the student's mean hit x-drift is large (0.15 m vs 0.0075 m for the teacher): it keeps the ball alive while letting it wander laterally. Apex error improves (0.16 vs 0.44 m). I did not inspect each student episode.
4. Mass: RL seeds disagree (-20%, +3900%, +3900%); the condition kills the teacher in 2 hits (arm cannot hold the racket at x5 mass under the 60 Nm cap with the nominal CTC), RL training took up to 1056 s as episodes got long. Sensing: RL seeds -90/-59/+5% with 3M steps; probably under-trained (delay needs prediction, no recurrent state) - not evidence that RL cannot help.
5. Pilot limits: 3 RL seeds, 1 teacher, one plant; the teacher and students share the estimator (BC students are nearly identical copies of the teacher because the teacher torque is in their observation; this makes H1-style "BC does not beat the teacher" nearly trivial); calibration grids were extended once (round 2) after round 1 found nothing; mass error x5 is large; contact e fidelity ~±0.05. Reported hits cap at 100, so "median 100" hides differences above the cap (a higher cap might separate the re-tuned teacher from the student).
6. Machine load average was 20-60 (other jobs) so wall-clock numbers are inflated.

## Wall-clock
Env/teacher/contact development + calibration: ~1.5 h (including waiting for the shared lock behind another job, ~15 min). Final pipeline `run_all.sh` 00:24-02:20 (1 h 56 min): BC 10 seeds per condition 2-5 min; RL per seed 225-290 s (restitution), 459-1056 s (mass), 518-900 s (sensing). Control teacher sweep ~15 min; video ~10 min. Total ~4 h. RL was NOT reduced (all 3 conditions trained).
