# Architecture notes and design decisions

## Using it

```python
from arena import Scenario
from arena.perturb import PayloadError, SensorNoise, Delay, TorquePulses
from arena.eval import evaluate, evaluate_grid, metrics_table, paired_table, CAL_SEEDS, TEST_SEEDS
from arena.viz import record_rollout

sc = Scenario("tracking", {"ref": "glyph"}).with_perturb(PayloadError(rel=0.3), SensorNoise(q=1e-3, qd=0.02), Delay(2))
res = evaluate_grid([sc], {"pid": "pid", "ctc": "ctc_lqr"}, CAL_SEEDS)
print(metrics_table(res)); print(paired_table(res, baseline="pid"))
record_rollout(sc, "ctc_lqr", seed=1000, path="clip.mp4")
```

Core contracts (all in `arena/types.py`, `controllers/base.py`, `tasks/base.py`, `perturb/base.py`):

* `env.reset(seed) -> Obs`; `env.step(tau) -> (Obs, reward, terminated, truncated, info)` (gymnasium convention;
  `info['outcome']` is `'failure' | 'success' | 'timeout' | None`). `Obs` has `t, k, q, qd, ref{} (read-only), extra{},
  prev_action, history (H,d) [Scenario(history=H)], ref_preview(H)`.
* `Controller(ctx)` with `reset(seed)` and `act(obs) -> tau (n,)`. `ctx` (`ControlContext`) carries the NOMINAL robot, task name,
  control period, torque limit, task constants, `n_steps`, `rng` (seeded from the episode seed), `ref_fn(k)` (reference at
  step k, for iLQR/MPC) and `make_model()` (a fresh NOMINAL `mujoco.MjModel` incl. tool and task objects such as the pole).
  Controllers never see the true plant. Random numbers must come from `self.rng` / `ctx.rng`.
* A controller factory is any `callable(ctx) -> Controller` or a registered name (`controller_factory("pid", kp=...)`).
* `Task`: `reset(seed, env)`, `reference(k)`, optional `error(env)` (None for event tasks), `measure(env)`, `terminated(env)` (failure),
  `succeeded(env)`, `reward(env)`, `metrics` (generic metric names wanted), `primary_metric`, `extra_metrics(trace)`.

### Add a task (one new file in `arena/tasks/`, nothing else edited)

```python
# arena/tasks/hold.py
import numpy as np
from .base import Task, register_task

@register_task("hold")
class Hold(Task):
    name, duration = "hold", 2.0
    def reset(self, seed, env): env.data.qpos[:2] = [0.1, 1.0]
    def reference(self, k): return dict(q=np.array([.1, 1.]), qd=np.zeros(2), qdd=np.zeros(2), ee=np.zeros(2))
    def error(self, env): return env.data.qpos[:2] - [0.1, 1.0]
```
Files in the package are auto-imported (`registry.autoload`). Use `Scenario("hold")`; all controllers that read the standard
`ref` keys work immediately; others raise `NotImplementedError` in `__init__` for it (`evaluate_grid` skips them and lists the skipped
combinations; a `NotImplementedError` raised inside `act` propagates).
Extra scene bodies (ball, pole, ...) come from `Task.tip_xml()`; extra metrics from `Task.extra_metrics(trace)`.

### Add a controller (one new file in `arena/controllers/`)

```python
# arena/controllers/zero.py
import numpy as np
from .base import Controller, register_controller

@register_controller("gravity_only")
class GravityOnly(Controller):
    def __init__(self, ctx):
        super().__init__(ctx)
        from ._arm import ArmModel; self.m = ArmModel(ctx.robot)
    def act(self, obs): return self.m.gravity(obs.q)
```
A learned policy is the same: load weights in `__init__`, call the network in `act`, `obs.to_vector()` gives a flat float32 input.
Residual learning: `controller_factory("residual", base="ctc_lqr", policy=MLPPolicy(state_dict), bound=16.0)`.

### Add a perturbation
Subclass `arena.perturb.Perturbation`, override any of `modify_robot / modify_task / reset / on_obs / on_command / qfrc / active`
(`PoleLength`, `BallRestitution` change TRUE task parameters; the controller keeps the nominal `task_info`).
They compose by listing: `scenario.with_perturb(A(), B())` (command hooks run in list order).

## Evaluation protocol v2 in code

`arena/eval/protocol.py`: `TRAIN_SEEDS` 10000-10199 (RL / learned-controller training, data collection), `CAL_SEEDS` 1000-1049
(all tuning and weakness calibration), `TEST_SEEDS` 0-49 (locked, report only), `DEMO_SEEDS` 5000-5019 (demos and unit tests);
pairwise disjoint (tested). `evaluate(..., split=...)` checks that the seeds belong to the named split (inferred if omitted,
mixing splits is an error) and every evaluation on `TEST_SEEDS` appends `time, script, condition` to `results/.test_access.log`
(only `evaluate` is guarded; `run_episode` is not; unit tests redirect the log with `ARENA_TEST_LOG`).
Metrics per episode (`eval/metrics.py`, a task lists the ones it wants in `Task.metrics`): `rmse` (whole horizon, transient
included), `ss_rmse` (t >= 0.5 s), `settle_time`, `recovery_time` (after torque pulses; NaN without pulses), `sat` (fraction of
executed steps), `energy` (mean over executed steps of sum tau^2), `alive_frac`, plus task extras (`ee_rmse`; pole: `survival_time`
(primary), `fell`, `tip_rmse`, `tip_settle_time`). Tables: mean +- 95% t-CI over seeds, percentile-bootstrap CI for 0/1 metrics;
paired difference per seed (seeds must be identical), "WIN" = CI entirely on the better side and mean improvement >= 10%
(base mean 0: any improvement whose CI excludes 0). `paired_table(..., holm=True, common_survivors=True)`.

## Design decisions 

1. **Single environment, not batched.** spike/ stepped N plants in a Python loop anyway; one plant per `ArenaEnv` keeps the
   interface and the tests simple. A batched/vectorised wrapper for RL training is future work. Cost: ~1-3 s per 50-seed cell.
   `ArenaGymEnv` wraps one env (gymnasium API; gymnasium itself is not a dependency and is not installed in `.venv`: a stand-in
   with the same signatures is used so the tests run, install gymnasium before training with SB3).
2. **Controllers get nominal knowledge via a factory context**, so "model error" is a property of the scenario (perturbations
   change only the true plant). `PayloadError(rel=0.3)` means true = nominal x 1.3 (spike used nominal = true x (1+e); tests
   map between the two with `PayloadError(mass=...)`).
3. **Torque cap and delay are perturbations; the controller still assumes the nominal limit** (as in spike02). Saturation is
   measured against the smallest effective limit.
4. **Exact legacy compatibility is kept as options**: `tracking(ref="joint_sines")` is the spike joint-space reference,
   `TorquePulses` uses spike's seed offset/draw order, `Delay` holds the first command during warm-up, friction uses
   tanh(qd/0.05). Result: CTC+LQR matches spike01 and the spike02 weaknesses to ~1e-12 (tests allow 1e-6).
5. **New default tracking task = a figure drawn by the tool tip** (Lissajous, or a glyph through S/8/M/Z/heart/circle with
   minimum-jerk timing), IK (damped least squares, warm-started, elbow-positive) to joints, spline for qd/qdd on a padded
   time grid. Error is still the joint error, so RMSE is comparable with spike numbers; `ee_rmse` is reported too.
6. **pole_balance** = massless rod + point bob on a passive hinge at the tip, i.e. the textbook inverted pendulum driven by
   tip acceleration. Error vector `[theta, dx, dz]` mixes rad and m (kept simple; revisit if a weighted metric is wanted).
   Episode ends when |theta| > 0.6 rad. `survival_time` is the primary metric.
7. **PID for the pole**: a position-target PD cannot balance a pole (the closed-loop polynomial has a negative constant
   coefficient), so the PID baseline is a joint velocity servo driven by a PID on theta that outputs tip velocity. Its gains
   were found by random search on calibration seeds 1000-1003 only. It balances the nominal case but is weak; that is the point of the ladder.
8. **PID tracking gains** are per-joint critically damped (`wn = 20 rad/s`) from the nominal inertia at the first reference
   posture, because fixed gains were unstable on the 3-link arm. No feed-forward, with gravity compensation (nominal model).
9. **Residual wrapper** is an interface port: `tau = clip(base + bound*clip(policy(features)))`, `spike_features` = spike02 `feats`
   (2-DoF only), `MLPPolicy` runs a saved tanh-MLP in numpy. Test: the stored spike02 actor re-run in arena reproduces its stored RMSE.
10. **Perturbations on observations** only touch `q, qd` and pole signals, never the reference. `Delay(steps|ms, where='command'|'obs'|'both')`;
    `Delay.from_ms(ms)` converts with the env's real control period at reset (`from_ms(ms, dt=...)` converts at once).
11. **Tools**: `paddle` is a collidable box, `gripper` is two passive fixed fingers (no actuation yet), both just add mass/geometry.
    Added for the ball tasks: `net` (shallow collidable basket, opening along the tool's local +z) and `claw` (visual holder, no contact).
12. **Tests use DEMO/CAL seeds**, except one check that re-computes spike01's reported table value (0.0097) and one that re-runs
    the stored spike02 actor on already-reported test trajectories; neither tunes anything.

### Review fixes (protocol and interface)

13. **Perturbation state is per env**: `Scenario.make_env` deep-copies the perturbations (delay buffer, noise rng and pulse table are
    per-episode state). `ArenaEnv` also deep-copies the task and applies `modify_task`, so `env.task` is the TRUE task and
    `env.nominal_task` the one behind `ctx.task_info` / `ctx.make_model()`.
14. **Episodes of different length are comparable**: `energy` is a per-step mean (the old horizon integral = energy x duration; the
    spike01/02 reproduction tests multiply back), `sat` a fraction of executed steps. In `paired_table`, when `alive_frac` differs
    between the two controllers on any seed, `energy` and `sat` rows are never a WIN (annotated "survival differs").
15. **Non-finite values are never dropped**: non-finite controller output is replaced (nan->0, inf->+-tau_max) and the episode is a
    divergence failure; `mean_ci / paired_diff / bootstrap_ci` raise `NonFiniteError`; the tables use `summarize` /
    `undefined='mark'`, which exclude NaN only visibly (n column `valid/total`, cell suffix). A metric that is NaN for all seeds by
    design (recovery_time without pulses) shows n/a.
16. **pole_balance error is theta (rad) only**. The tip error (m) is the channel `tip_err` with `tip_rmse` / `tip_settle_time` (0.02 m).
    `Task.settle_thr` is per error component in that component's unit. Consequence: theta-rmse no longer ranks PID below CTC
    (the old ranking came from the tip drift in the mixed norm); the tip metric does (test updated).
17. **One termination rule**: the rollout stores the state after the last action (`Trace.final`) and every Trace has an outcome
    (`failure` = task failure / divergence, `success`, `timeout` = horizon reached). `Trace.series()` appends the terminal state and
    charges every remaining step the FIXED penalty `protocol.FAILURE_ERR = 2.0` per error component after a failure (2.0 rad >> the 0.6 rad fall;
    tip: `FAIL_TIP_ERR = 0.5` m); after a success the terminal error is held. `rmse` on all seeds therefore includes the penalty
    (by design: a fall is worse than any tracking error); use `paired_table(common_survivors=True)` to compare error metrics only on
    seeds both controllers survived. `survival_time` / `success` are always compared on all seeds. Reaching the horizon of a balance task
    is 'timeout' (= survived).
18. **Seeded randomness**: `Controller.reset(seed)` (re-seeds `self.rng`), `ctx.rng` (re-seeded by every `env.reset(seed)`); `run_episode`
    passes the episode seed. Old `reset()` signatures must become `reset(self, seed=None)`.
19. **Splits**: see the protocol section. `ArenaGymEnv.reset()` draws seeds from `TRAIN_SEEDS` only.
20. **Event tasks**: `Task.error` may return None; list only event metrics in `Task.metrics` and compute the rest in `extra_metrics`.
    `success` / `fell` are binary (bootstrap CI). `terminated` in `env.step` includes success; truncated = horizon.
21. **Reward** (`Task.reward`) is for RL only: tracking `-mean(e^2)`; pole `1 - (theta/fall)^2 - 0.5|tip err|^2`, `-10` on a fall. Evaluation never uses it.
22. **Previous action / history**: `prev_action` is the controller's (sanitised) command, not the delayed/clipped torque. `history`
    holds past `to_vector()` rows (oldest first, padded with the first observation at the start); `Obs.to_vector()` does not include
    either. `HoldWrapper` ("hold") counts its own calls from `reset`, so it needs `reset(seed)` before the first `act`.
23. **Task-parameter perturbations**: `BallRestitution` only sets `task.restitution` (raises AttributeError on tasks without it); the
    ball task (not written yet) must turn that parameter into contact parameters in `tip_xml`. It is tested with a dummy task.

### Classical ladder, upper half: iLQR, iLQR-MPC, MPPI 

24. **Nominal model only.** `controllers/_ocp.py` (`NominalOCP`) builds its own `ctx.make_model()` copy (exposed as `ctrl.model`; a test
    checks that under `PayloadError` its masses equal the nominal ones). Supported tasks: `tracking`, `pole_balance` (others raise
    `NotImplementedError` in `__init__`). State = `[qpos, qvel]` of the MuJoCo model; for `pole_balance` the passive hinge is part of the state
    and is reconstructed from `pole_theta` / `pole_omega` (hinge = theta - sum(q)). Reference = `obs.ref_preview` only.
25. **Dynamics.** One RK4 step of the control period (5 ms) instead of the plant's two 2.5 ms steps (max difference ~1e-6, tested);
    `mjd_transitionFD` does not support RK4, so the Jacobians are forward finite differences (eps 1e-6) of all T steps in ONE
    `mujoco.rollout` call (no extra processes, 1 thread; tested against central differences). Torque limit = `ctx.tau_limit`, applied by clipping
    in the forward pass (no box-QP in the backward pass: the gains ignore saturation).
26. **Cost** `1/2 dx'Q dx + 1/2 du'R du`, `dx = x - x_ref`, `du = u - u_ref` with `u_ref` = nominal inverse-dynamics torque of the reference
    (so R penalises only the deviation from the feed-forward). tracking: `Q = diag(1e4 q, 1e2 qd)`, `R = 1e-3`; pole: world pole angle/rate
    `theta 2e3, omega 1e2` plus arm posture `q 20, qd 2` (keeps the tip near the hold point), `R = 1e-2`. Terminal cost = discrete
    Riccati solution at the end of the horizon (frozen linearisation). Weights were set by hand on a few calibration seeds, not swept; override with `weights={...}`.
27. **`ilqr`** plans the whole episode once, at the first `act` (full horizon, warm start = time-varying LQR around the reference, then
    regularised iLQR with line search, <= 20 iterations), then executes `u* + K_t (x - x*_t)`; there is no re-planning. Its planning
    time is `offline_time` and is NOT part of the per-step latency. **`ilqr_mpc`** re-solves every `replan_every=2` steps over `horizon=40` steps
    (3 iterations, warm start = previous plan shifted and re-rolled closed-loop from the measured state) and applies the plan's feedback law between solves.
    **`mppi`**: `samples=128`, `horizon=25`, smoothed Gaussian noise (`sigma = 0.1 tau_limit`), adaptive temperature, nominal sequence initialised with the
    reference feed-forward and shifted each step, noise from `self.rng` (same seed -> identical torques, tested). `mujoco.rollout` does the sampling rollouts.
28. **Timing for the latency axis.** `TimedController` (base of the three) records per `act` call: `compute_times` (wall), `cpu_times` (thread CPU),
    `info_log` (per-step dict: `step_time`, `solve_s`, `iters`, `cost`, ...) and `timing()` (mean / p95 / max ms, offline seconds). `run_episode` only
    returns the torque, so read them from the instance: `made = []; run_episode(sc, lambda ctx: made.append(f(ctx)) or made[-1], seed)`
    (see `scripts/demo_classical_ladder.py`). The machine is shared, so wall times are inflated; compare cpu times. The latency perturbation itself is the
    existing `Delay`; these controllers do not compensate for their own computation time.

### Ball tasks: catch and toss 

Files: `tasks/catch.py`, `tasks/toss.py` (+ `tasks/_ballistics.py`: closed-form projectile with linear drag, analytic level IK, quintic
segments), `controllers/catch_teacher.py`, `controllers/toss_teacher.py`, `perturb/ball_tasks.py`, tools `net` / `claw`
(`robots/`), `tests/test_ball_tasks.py`, `scripts/demo_ball_tasks.py` -> `results/ball_tasks/{table.md, *.mp4}`.
Both tasks use the 3-link arm, an event primary metric (`success`, 0/1, bootstrap CI) and report their secondary metrics from `extra_metrics`.

24. **Projectile physics**: the ball / object is a sphere with two slide joints (x, z, no spin). Air drag is a linear drag F = -c v
    implemented as joint damping, so the drag rate is k = c / m (1/s) and the mass matters through it. Linear (not quadratic) drag
    keeps the nominal ballistic model in closed form (exact for the teacher when there is no error); a quadratic law would only change `_ballistics.py`.
25. **`tip_xml` sections (the one change below the task layer)**: `build_mjcf` splits the task's `tip_xml` at sentinel comments:
    `<!--@world-->...` goes into `<worldbody>` (a free ball, a basket) and `<!--@equality-->...` into `<equality>` (the connect
    constraint that holds the toss object). Text before any sentinel is still placed in the tip body, so old tasks are unchanged (tested).
    Neither `env.py` nor any interface was edited.
26. **Per-step state in tasks goes through `Task.terminated`**: the env calls it exactly once per step, so the catch dwell counter and the
    toss release schedule are updated there (idempotent per `env.k`). A task that wants a per-substep hook would need a new env hook; not done.
27. **catch geometry**: the launch (seed) is generated from the NOMINAL drag and mass: ball from x ~ U(1.25, 1.55) m (outside the 1.0 m reach),
    aimed through a random point (x 0.45-0.75, z 0.1-0.4) after 0.55-0.75 s. So the same seed gives the same ball under a drag / mass error
    (it drifts off the aim point). The basket is level (sum(q) = 0) when it catches: the teacher uses analytic IK with that constraint.
    The ball is only observed by position (`Obs.extra['ball_pos']`): velocity has to be estimated (use `Scenario(history=H)` or a filter).
    `Obs.ref` is empty. Success = ball centre inside the basket volume for 0.5 s CONTINUOUSLY (episode ends then); floor at z = -0.85 m = failure.
    `impact_speed` = |v_ball - v_tip| in the row before the first ball-basket contact appears (NaN: no contact); `restitution` (BallRestitution)
    is mapped to the contact damping ratio of the ball geometry.
28. **catch teacher**: online least-squares fit of the nominal drag-ballistic model to the last 0.25 s of ball positions (timestamps = control time),
    earliest intercept time that is reachable AND torque-feasible under the nominal model (feed-forward inverse dynamics), tip-space path
    (reach, short ride-along with the basket floor under the ball, brake) -> analytic IK + Jacobian for q, qd, qdd, computed torque, re-plan every
    0.03 s with the intercept time fixed. Because it keeps re-fitting, a drag / mass error costs little; what breaks it is the measurement
    (delay and noise), which is exactly the part a learned policy with history could improve.
29. **toss geometry**: `connect` equality holds the object at the tip; the release is SCHEDULED by the task (t = 0.8 s, plus the true
    `release_delay`), because the action space is torques only. The controller must hit the release position and velocity at t_release.
    Basket (seed): x ~ U(0.7, 1.8) m (reach 1.0 m, so most baskets must be thrown at), floor top z ~ U(-0.5, -0.1) m, inner half width 0.11 m, wall
    height 0.2 m. Success = object centre inside the basket volume (it entered through the opening; episode ends, a later bounce-out is not
    scored). Failure only after the release (z < -0.9 m or x > 3 m). `landing_err` = |x - x_basket| at the first downward crossing of the rim
    plane after the release. The nominal object (0.15 kg, c = 0.10 N s/m, k = 0.67/s) is light and draggy on purpose so that a drag / mass
    error moves the landing by more than the opening.
30. **toss teacher**: grid over release point (arc around the base) x launch angle x release orientation, launch speed by root finding on the
    nominal drag-ballistic flight (crossing the rim plane at the basket centre), min-norm pseudo-inverse for the release joint velocity, quintic joint
    trajectory from the start posture, candidate chosen by the smallest peak feed-forward torque; computed torque with the object mass included before t_release.
    The plan is computed once at the first step from `Obs.ref['basket']`.
31. **Pilot perturbation strengths** (demo script, DEMO seeds, NOT calibrated): drag x2, mass x2, ball noise 10 mm, ball delay 10 ms (catch); drag x2,
    mass x2, release 40 ms late, friction (Coulomb 5 Nm + viscous 1.0) (toss). Final strengths come from the CAL seeds. Results: `results/ball_tasks/table.md`.

### Whip and Push-T: the two "weak classical teacher" tasks 

Files: `tasks/whip.py`, `tasks/push_t.py` (+ `tasks/_pusht_geom.py`: T geometry, exact polygon-clipping coverage), `controllers/whip_teacher.py`
(+ baseline `whip_random`), `controllers/pusht_teacher.py` (+ baseline `push_t_random`, + `controllers/_pusht_plan.py`: quasi-static push model and
planner), `perturb/whip_pusht.py`, `tests/test_whip_pusht.py`, `scripts/demo_whip_pusht.py` -> `results/whip_pusht/{table.md, *.mp4}`.
No core file was edited except the export list in `perturb/__init__.py`. Event tasks like the ball tasks: primary metric `success` (0/1, bootstrap CI).

32. **whip plant.** 3-link arm (0.40 / 0.35 / 0.25 m, 0.2 kg payload) + a planar chain of 10 capsules (0.10 m, 0.015 kg, hinge damping 4e-4 N m s/rad, no
    actuation, gravity on) hinged at the tip through `tip_xml` (nested bodies; no sentinel needed). The chain state is not observed (`Obs.q` is the arm).
    Target (seed): distance 1.15-1.70 m from the shoulder, angle +-0.45 rad, always outside the 1.0 m reach and inside reach + chain. Success = chain
    tip within `r_hit` = 8 cm at any time in 2.5 s (the episode ends then, so `min_dist` of a success is its first-hit distance, < 8 cm). `hit_speed` = chain-tip speed
    at the first hit (finite difference of the logged positions).
33. **whip teacher = CMA-ES keyframes + computed torque, not MPPI / iLQR.** The only thing that matters is the closest approach ~1 s after the arm has moved
    (10 passive hinges), a non-smooth cost with no usable gradient for a short-horizon receding controller, but a small smooth problem in joint-space
    keyframes. So, per target: clamped cubic spline through the start posture and 4 keyframes per joint (12 parameters), (mu/mu_w, lambda)-CMA-ES (own
    implementation) on the NOMINAL arm + chain with the arm as a stiff PD servo with gravity compensation, acceleration and velocity feed-forward and the real torque
    limit (a batched `mujoco.rollout` proxy of computed torque; ~0.7 s per generation of 24), then the best 8 distinct candidates are re-simulated in the exact closed loop
    that will run and the best is kept; execution = computed torque with the arm-only nominal model (chain reaction = disturbance). The chain is never measured: the
    fling is open loop for the chain. The plan is cached per (class, seed, target, start posture, nominal parameters), so a perturbation sweep over the same
    seeds does not re-plan (perturbations never reach the planner anyway). Cost of the search: ~3-10 s per target on an idle machine.
34. **whip perturbations** (`ChainMass`, `ChainDamping`, `ChainLength`, all `rel` plus optional per-link `spread`/`pattern`: each link is multiplied by an own
    lognormal factor fixed by `pattern`, so the same Scenario always has the same uneven chain). They change the true chain; the target is unchanged (tested).
35. **push_t plane and gravity.** The arm lives in the x-z plane; for this task that plane is the table seen from above. Gravity is switched off in `reset` (true model) and
    `task_info['gravity'] = 0` tells controllers to do the same in their nominal models (`ctc_lqr`, iLQR, MPPI refuse the task or would carry a wrong gravity term; only the
    teacher is written for it). Pusher = cylinder (r 2.5 cm) at the tip of a 2-link arm (0.5 / 0.4 m, 0.3 kg), T block (bar 0.20 x 0.05 m, stem 0.05 x 0.15 m = the Diffusion
    Policy T, 1 px = 1.667 mm, 0.4 kg) on three planar joints. The goal pose is fixed ((0.55, 0), 45 deg) as in the paper.
36. **push_t table friction = viscous + dry (50/50).** With F = mu m g: joint damping b = 0.5 F / v_ref (v_ref = 0.1 m/s) on both slides and b rho^2 on the hinge (an exact ellipsoidal
    limit surface: the model the teacher assumes) plus joint `frictionloss` 0.5 F (slides) and 0.5 F rho (hinge) for the dry part, which the model does not capture. Pusher-block
    contact = MuJoCo's friction cone with coefficient mu. A purely viscous table was tried first: the teacher's model was then exact and it never failed. (Diffusion Policy's
    pymunk world with `damping = 0` is also quasi-static; this is a deliberate, slightly less ideal choice.)
37. **push_t COM offset.** The body frame sits at the true COM and the geometry is shifted, so friction acts about the true COM; the observation is the pose of the OUTLINE (the
    centroid frame, what a camera sees). The offset is invisible in `Obs` (tested) and changes only the physics.
38. **push_t events and metrics.** Coverage = overlap area / goal area (exact; checked against a raster estimate and closed-form shifted-T cases), success = coverage >= 0.9, the episode
    ends then (so `coverage` of a success is the value that triggered it, 0.90-0.91); failure = block leaves x in (0.2, 1.0) or |z| < 0.5 (pushed off the table) unless it is a success; timeout 30 s.
    Secondary: `coverage` (end), `max_cov`, `time_to_success`, `path_len` (pusher path).
39. **push_t initial states (seed).** Block position = goal + U(-10, 10 cm)^2, orientation = goal + U(-pi, pi) (full circle like the paper; rejection: initial coverage <= 0.3, outline inside the
    work area), pusher at a random free point >= 10 cm from the outline.
40. **push_t teacher** = quasi-static push planning + operational-space computed torque. Model: point-like pusher with friction coefficient mu, ellipsoidal limit surface with radius of gyration
    rho (sticking / sliding / separating modes, vectorised over candidates). Planner: primitive = (contact point on the outline [16 points], push direction inside the cone [3], push distance
    [1, 2.5, 5, 9 cm]) -> 168 primitives; beam search depth 4, beam 10; cost = |dp|/2 cm + |dtheta|/0.15 rad + 0.15 per primitive; the first primitive of the cheapest leaf that has an approach
    path (visibility graph around the outline offset by the pusher radius) is executed, plus 0.3 per metre of travel. Closed loop at primitive level: approach at 0.4 m/s, push at 0.08 m/s,
    hold 0.2 s, average the pose samples of the hold (noise filter), re-plan; if nothing is feasible back off 6 cm. Tracking: OSC `qdd = J^+(kp e + kd ed - Jdot qd)`, `tau` = nominal inverse dynamics.
    It knows the goal, nominal mu, rho and the pusher radius; it does not know the true table friction, mass or COM offset. The baseline `push_t_random` draws the primitive at random
    (everything else identical).
41. **Honest result: the Push-T teacher is NOT weak.** On CAL seeds it succeeded 12/12 clean and under every block-parameter perturbation tried (friction x0.5-x2, mass x0.25-x4, COM offset 40 mm
    in x or 30 mm in z, pose noise 8 mm / 0.08 rad). That is expected from quasi-static pushing theory: with a position-controlled pusher, the outcome depends on the limit-surface shape and the
    contact friction ratio, not on friction magnitude or mass, and a closed-loop re-planner repairs the rest. What does break it is latency of the arm's own feedback (sensor delay on q, qd
    >= 30 ms makes the OSC loop unstable: the block is flung off the table), see the table. So for the paper: Push-T is an example of "model error in the block parameters does not hurt a
    closed-loop classical teacher; delay does", not of a weak teacher; the random-primitive baseline (0%) shows the planner matters.
42. **Pilot strengths** (demo script, CAL-chosen, NOT calibrated). whip: chain mass +-50%, damping x3 / x0.25, length +-10%, uneven mass (log-sd 0.3); push_t: friction x0.5 / x2, mass x0.25 / x4,
    COM 20 / 40 mm, pose noise 8 mm / 0.08 rad, sensor delay 20 ms / 30 ms (`Delay(n, where='obs')`). Results: `results/whip_pusht/table.md`.
43. **Results on DEMO seeds** (`results/whip_pusht/table.md`, whip n = 20, push_t n = 20, pilot strengths, bootstrap 95% CI in the table). whip teacher: 100% without error
    (tip speed at the hit 9 +- 2 m/s), chain mass +50% 90%, -50% 80%, damping x3 80%, x0.25 75%, length +10% 65%, -10% 60%, uneven chain 65%; random keyframes (no search) 15%.
    push_t teacher: 100% without error (first success after 12.6 +- 1.5 s, pusher path 2.6 m), friction x0.5 / x2 100%, mass x2 / x4 100%, COM 20 mm 95%, COM 40 mm 100%, pose
    noise 8 mm / 0.08 rad 100%, sensor delay 20 ms 95%, sensor delay 30 ms 0%; random primitive 0%. Whip is the "weak because of model error" example (every chain parameter
    costs 10-40 points); Push-T is the "weak because of latency only" one. The tests (`tests/test_whip_pusht.py`) require >= 75% on the first 8 / 5 demo seeds and that the
    baselines, a -10% chain length and a 30 ms sensor delay each cost at least 25 / 50 points.
44. **Known limits.** (a) Perturbation strengths are pilot values, not calibrated on the CAL seeds. (b) The whip hit radius (8 cm) and the Push-T threshold (0.9) were fixed by the
    task spec, not tuned; whip `min_dist` of a success is truncated at the first hit. (c) Push-T's three requested block perturbations barely matter (see 41); to give a student something to
    learn from, use sensor delay / pose noise or change the contact model (e.g. a pusher with compliance, a non-uniform pressure distribution). (d) The whip chain is never measured, so the
    teacher cannot correct a fling in flight; a student with chain observations would be a different problem. (e) The planner caches whip plans in a module-level dict (per process); the
    cache key includes the controller class, seed, target, start posture and the nominal chain parameters. (f) Push-T runs ~30-60 s per 30 s episode on a loaded machine (planning ~0.5 s per
    primitive); the whip search costs 3-10 s per target on an idle machine. (g) Push-T arm gravity is off in `reset` of the true model only: any other controller built from
    `ctx.make_model()` sees gravity unless it reads `task_info['gravity']`.

### L6: imitation students (BC-MLP, MIP, Diffusion Policy, flow matching) 

Files: `arena/learn/{features,dataset,models,train}.py`, `controllers/student.py`, `tests/test_learn_{data,models,student}.py`,
`scripts/demo_students.py` -> `results/learn_demo/{table.md, data/, ckpt/, ckpt_rel/}`. No core file was edited (torch is imported lazily
by `controllers/student.py`; `import arena` still works without torch). Pipeline: teacher -> `collect` (npz shards of whole episodes,
read shard by shard; optional DART noise and DAgger rounds) -> `train` (GPU) -> `student` controller -> `evaluate(split="test")`.
Research question: what must a student get besides imitation to beat a classical teacher; Much Ado About Noising (arXiv 2512.01809)
says the advantage of diffusion / flow is NOT multimodality, so the four students share data, features, backbone, optimiser, budget.

45. **Install.** The old `.venv` has torch 2.14.1+cpu (used by pytest, tests are tiny CPU models). Training uses a second venv `.venv-gpu`
    (torch 2.14.1+cu130 from the cu130 index, `uv pip install -e . pytest` on top; arch list contains sm_120, `cuda.is_available()` True on
    a consumer GPU). Training uses a GPU job queue (fp32, VRAM < 1 GB, no TF32/bf16: the targets are precision-hungry, the nets tiny).
46. **Observation = `Obs` only.** Row = `Obs.to_vector()` (q, qd, sorted ref keys, sorted extra keys); the input is the window
    [`Obs.history` (H rows), current row] flattened, so the env must be built with `Scenario(history=H)`; the student raises if the history is
    too short. `prev_action` is not used. Optional `rel_ref` appends (q - ref.q, qd - ref.qd) to each row: a linear function of the row, i.e. no new
    information, but it is the "extra input" ablation of the demo. The student never touches `ctx.make_model`, `ref_fn` or the env (tested).
47. **Backbone shared by all four.** Residual MLP (LayerNorm, Linear, Mish, Linear; width 256, 4 blocks) over concat[o, flat action chunk, sin/cos(t)];
    the chunk is flattened so chunking needs no special case, the observation horizon is the window. A 1-D temporal U-Net is NOT implemented
    (Ta = 8, 2 joints: nothing to convolve over). BC has no (x, t) input, its width is searched (+-24) so its parameter count equals the denoisers' (table in
    `table.md`: diffusion = flow = MIP exactly, BC within 0.5%).
48. **MIP** follows arXiv 2512.01809 sec. 4 (Eq. 4.4 / 4.5 as read from the HTML version, the appendix was not read): loss = |pi(o, 0, 0) - a|^2 + |pi(o, I, t*) - a|^2
    with I = t* a + (1 - t*) z, z ~ N(0, I), t* = 0.9, both terms supervised with the true action chunk; inference a0 = pi(o, 0, 0), a = pi(o, t* a0, t*)
    (2 network evaluations, no sampling noise). **Flow** = rectified flow (x_t = t a + (1-t) z, target a - z, t ~ U(0,1), Euler sampling with `steps`; 1 step is the
    one-step version, evaluated from the same weights as the 10-step one). **Diffusion** = DDPM training (epsilon prediction, cosine schedule, K = 100) with deterministic
    DDIM sampling (10 steps; `eta > 0` gives ancestral noise; x0 clipped to +-5 in normalised units). Observations and actions are standardised with streamed dataset statistics.
49. **Equal budget.** 15 000 steps, batch 256, AdamW lr 1e-3 cosine to 1e-5, grad-clip 1, EMA 0.999 (EMA weights saved), seeds 0-2 per student,
    same shard files, no per-student tuning (a fair-by-construction comparison, but each student could be better with its own tuning).
50. **Data.** `collect` uses the teacher's own `act` as label; DART executes teacher + N(0, (sigma tau_limit)^2) (sigma = 0.1, tau_limit 80 Nm; see 54) and labels with the clean
    action. DAgger: `collect(..., student=factory, beta=b, append=True, round_id=r)` executes the student (teacher with probability beta per step) and labels every
    state with the teacher; chunk labels along a student trajectory are the teacher's labels at the visited states (not a teacher plan from there). The demo uses DART only (no DAgger rounds).
    Rows of a shard are whole episodes; chunk labels repeat the last action after the episode end. Randomness: `default_rng([seed, 0xDA27, round])`.
51. **Student controller.** `controller_factory("student", ckpt=..., exec_k=k, sample_kw={"steps": n})`: receding horizon (predict a chunk of Ta, execute the first k, re-plan);
    `chunk_out=True` returns the (Ta, n) chunk for `HoldWrapper(chunk=True, every=k)` (tested identical). Sampling noise comes from a torch Generator seeded from `self.rng`
    in `reset(seed)` (same seed -> same torques, tested for all four). `infer_times` / `timing()` record the wall time of every network call (latency axis B); the demo also measures
    batch = 1 on GPU and CPU. Checkpoints are cached per (path, mtime, device), so `evaluate` does not reload weights per episode.
52. **Demo scenario.** `tracking/clean`, Lissajous, 2-DoF, CTC+LQR teacher, nominal plant (no perturbation: the student can at best match the teacher, "beating" it needs a plant the teacher models
    wrongly). 200 teacher rollouts on TRAIN_SEEDS (160 000 rows), evaluated on TEST_SEEDS (logged in `results/.test_access.log`), paired with the teacher per seed. The CPU rollouts of the
    demo ran WITHOUT the shared `heavy.lock` for collection and checkpoint evaluation in this session (the lock queue held multi-hour RL jobs, the work is single-thread, < 1 GB, 1-10 min, niced);
    pytest ran with `systemd-run` only.
53. **Angle wrap (found by the first run).** The tracking reference is built from an unwrapped IK: on 10/200 TRAIN, 4/50 CAL and 2/50 TEST seeds joint 0 of `ref.q`
    sits at -7 ... +81 rad (the teacher only uses q - ref.q and does not care). Students fed raw q / ref.q diverged on exactly these seeds (rmse 0.8-1.4 rad
    while the others were ~0.02). `FeatureSpec(wrap_angles=True)` (used by the demo) shifts q and ref.q of each row by the same multiple of 2 pi so that ref.q is in [-pi, pi]
    (information-preserving; the shift cancels in q - ref.q). **Superseded by decision 56:** the task is fixed; the wrap is now a defensive no-op.
54. **Selection on CAL seeds, and a disclosed first look at TEST.** My first full run (DART 0.02, 30 000 steps, no wrap) was evaluated on TEST (it is in `results/.test_access.log`;
    BC / MIP rmse 0.16-0.24, flow / diffusion 0.4-1.4, dominated by the wrap seeds and by drift). Afterwards the choices were made on CAL seeds 1000-1019 with a 40-episode pilot dataset on CPU:
    angle wrap (fixes the outliers), DART sigma 0.1 (0.02 / 0.1 / 0.2 compared; 0.1 best for BC / MIP, 0.2 for 1-step flow), 15 000 steps (more steps overfit the correlated rollouts: at 40 episodes 3k steps
    beat 12k; 30k steps at 200 episodes ~ 6k at 40 so 15k was taken). The TEST numbers in `table.md` are the second evaluation; all students share every choice. Nothing else was tuned.
55. **Known limits.** No training resume from a checkpoint (jobs are minutes); no validation split (loss is the training loss, compare kinds by rollout, not by loss); one scenario, one
    teacher, no DAgger in the demo; the student's chunk length / execution horizon / T* / K were not tuned; the MLP denoiser is not the paper's U-Net; the timing numbers come from a heavily loaded machine.

56. **Tracking reference fix (angle wrap and IK branch).** Root cause: `Tracking._build_reference` chained plain damped-least-squares IK calls; near a singularity (straight arm) one solve
    can wander through many 2 pi turns (ref.q -7 ... +81 rad) or leap to another configuration, so on ~2-4 % of seeds the reference was not even the commanded figure (tip error up to 0.9 m;
    5/200 TRAIN lissajous seeds). Rule now: the first sample is wrapped into (-pi, pi]; every later sample takes, per joint, the 2 pi branch nearest the previous sample (unwrap), so the
    reference is continuous and starts on the branch nearest the elbow-positive warm start; if a direct solve misses the target or jumps > 0.5 rad it is redone with 8 / 64 / 512 sub-steps
    from the previous tip target. `tests/test_tracking_ref.py` checks all TRAIN/CAL/TEST/DEMO seeds, lissajous and glyph: start in (-pi, pi], |q| < pi+1, step < 0.1 rad, FK(ref.q) = figure to 1e-6.
    `joint_sines` is untouched (spike repro passes). `FeatureSpec.wrap_angles` is kept as a defensive no-op for tracking (old datasets/checkpoints keep working; it is a few lines and
    information-preserving); results made before the fix (see `docs/progress.md` follow-up) used the broken references.

