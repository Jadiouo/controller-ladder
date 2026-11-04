# MJX (JAX/GPU) port of the 2-DoF tracking task + residual PPO

Code: `arena_mjx/` (config, sampling, model, env, ctc, rollout, ppo), `scripts/mjx_*.py`, `tests/test_mjx_parity.py`.
`arena/`, `spike*/`, `docs/` untouched. Raw numbers: `parity_gpu_x64_{0,1}.json`, `bench_*.json`, `ppo_table.md`, `eval_*.pkl`, `policies/*.npz`.

## Setup
`uv pip install --python .venv/bin/python "jax[cuda12]" "mujoco-mjx==3.17.0" optax` -> jax/jaxlib 0.10.2, CUDA 12.9 wheels (>= 12.8),
mujoco 3.17.0 (= mujoco-mjx). **sm_120 works**: `jax.devices() = [CudaDevice(id=0)]`, "NVIDIA GeForce RTX 5070 Ti", compute capability 12.0, matmul run OK
(GPU queue job #80, `scripts/mjx_gpu_probe.py`). MuJoCo Warp was not needed. Every GPU run: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, `a GPU job queue systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 ...`.
`arena_mjx.setup()` sets `jax_default_matmul_precision=highest` (TF32 off; this model is not matmul-bound, the setting only removes the risk).
`pyproject.toml` was not changed (jax is an optional extra for this package; the test uses `pytest.importorskip`).

## What is ported (and what is not)
Joint-space tracking, `ref="joint_sines"` (the spike01-03 reference; exact legacy), 2-link arm, built from the SAME MJCF generator
(`arena.robots.build_mjcf`), RK4, dt_sim 2.5 ms, 2 substeps. Perturbations in the true plant: payload error (per env), friction
(`tanh(qd/0.05)` Coulomb + viscous), torque cap, torque pulses (table from the arena draw order), actuator delay (ring buffer, first command held, up to 8 steps),
non-finite command sanitising. Controller: CTC+LQR (`mjx.inverse` on the NOMINAL model). Reset is decided by the seed: reference, initial error and
pulses are numpy float64 tables built with arena's RNG draw order (`arena_mjx/sampling.py`; unit test: bit-identical to `Tracking`/`TorquePulses`).
**Not ported / limits: no contact tasks (juggling, catching, Push-T ...); this port is only valid for contact-free tracking. Contact tasks would need a
separate contact-equivalence validation (MJX contact solver/collision differ from CPU MuJoCo) before any use.** Also not ported: sensor noise, obs delay,
lissajous/glyph references (IK on CPU; could be tabulated the same way), 3-link, pole_balance, iLQR/MPPI, history in obs.
MJX model caveat: per-env payload is done by interpolating `body_mass/body_inertia/body_subtreemass` (affine in the payload); other model fields that change with the payload
(`actuator_acc0` only) are not used by the dynamics. Verified by the payload+30% parity case.

## 1. Numerical parity (gate for everything below) - all on GPU (GPU queue job #83/#90 fp32, #84 fp64), 20 DEMO seeds (5000-5019), never TEST seeds
`scripts/mjx_parity.py`. Part A: arena `ArenaEnv` (CPU MuJoCo) vs MJX, same MJCF, same initial state, same fixed torque sequence (sinusoids +-18 Nm + noise),
**1000 control steps (2000 physics steps)**, 7 plant variants (clean, +pulses, cap, cap15 saturating, delay 6, friction W3, W4 all, payload +30% + pulses).
Part B: CTC+LQR episode metrics per seed, arena evaluator vs MJX rollout.

| | float64 (x64) | float32 (training precision) |
|---|---|---|
| A: max abs joint-angle error over 1000 steps | **4.7e-8 rad** worst case (clean wild sequence, which is chaotic: arm spins ~25 rad); 5.6e-12 (friction), 9e-12 (W4), <=1.3e-9 other | worst case 5.2 rad (chaotic open-loop sequence, see below); **4.4e-4 rad** with friction (W3), 2.4e-2 W4+pulses |
| B: RMSE per seed, max abs / max rel | 1.6e-15 / 1.1e-14 | **1.2e-6 / 1.5e-5** (rmse); ss_rmse abs 1.2e-6 |
| B: energy, sat, max rel | 7e-15, 1e-16 | 7.6e-6, 6e-8 |

Tolerances stated: float64 must agree to 1e-9 (state) / 1e-8 (relative metrics); float32 closed-loop metrics to rel 1e-3 or abs 1e-6 (observed 1.5e-5 / 1.2e-6).
float32 open-loop (Part A): my fixed torque sequence is NOT stabilised, so the arm tumbles and rounding noise grows exponentially
(median-over-trajectories error in the W0 case: 1.6e-7 at step 25, 7.6e-7 at 100, 4.4e-6 at 200, 5.9e-5 at 500, 1.6e-3 at 1000; 20% of trajectories stay < 1e-3, W3 100%).
That is a property of float32 on an unstable system (the float64 run, which executes the same algorithm, matches to 1e-8 even there), not an implementation difference;
the closed-loop metric (Part B) is the one that matters for RL, and it agrees to 1.5e-5 relative. In the fast unit test (`tests/test_mjx_parity.py`, JAX CPU backend, subprocess,
no GPU, 2 seeds, 100 steps) the tolerances are 1e-9 (fp64) and 5e-3 (fp32); ~1 min.
Policy-level check (`scripts/mjx_policy_parity.py`, CPU backend, DEMO seeds, a trained actor with the output layer scaled x30 so the residual is large): MJX vs arena `Residual`
RMSE max abs diff 1.1e-7, saturation 5.6e-11.

## 2. Vectorised env
`MjxTracking` (`arena_mjx/env.py`): `jax.vmap` over E envs, `lax.scan` over the 800 control steps (`rollout.py`) / 200-step PPO horizon. Per-env parameters
(payload, torque cap, Coulomb/viscous friction, delay, pulse table, reference tables). All envs run in lock-step.

## 3. Measurements (loaded machine: load average 70-100 on 16 cores during all runs; GPU runs shared the card sequentially through a GPU job queue)
CTC+LQR episodes, W3 + pulses, 800 control steps per episode (`scripts/mjx_bench.py`, GPU queue job #88 fp32, #89 fp64; CPU: arena single env, no a GPU job queue, `bench_cpu.json`).

| backend | envs | control steps / s (1 step = 2 physics steps) | speed-up vs CPU arena | CPU use of the python process | JAX peak VRAM |
|---|---|---|---|---|---|
| CPU MuJoCo, arena env + python CTC | 1 | 5.3e3 (raw `mj_step` alone: 9.3e4 physics/s = 4.7e4 control/s) | 1x | 0.95 core | - |
| MJX GPU fp32 | 64 | 1.1e5 | 22x | 0.80 core | |
| MJX GPU fp32 | 1024 | 1.5e6 | 280x | 0.93 core | |
| MJX GPU fp32 | 4096 | **4.7e6** | 900x | 0.87 core | |
| MJX GPU fp32 | 16384 | 9.4e6 | 1800x | 0.84 core | **0.70 GB** (peak_bytes_in_use, incl. 4x(E,800,2) tables) |
| MJX GPU fp64 | 1024 / 4096 | 6.6e5 / 2.3e6 | | 0.8 core | 0.36 GB (4096) |

Compile (first call) 20-27 s. nvidia-smi sees more than JAX's number (CUDA context, ~0.3-0.5 GB); well below the 10 GB limit.
CPU burden: the python process still spins ~1 core while the GPU works (JAX host thread), so the saving is not "zero CPU" but **CPU-seconds per simulated step**:
the same core simulates 900x (4096 envs) more steps. PPO 2M steps (64 envs): 20-21 CPU-s / 22-23 s wall per policy (first policy of a process adds 4-8 min of XLA compile on the loaded CPU, 350-470 s CPU)
vs spike02 torch CPU ~150-170 s wall per policy with 2 torch threads (spike03a table), i.e. roughly 300 CPU-s (inference from wall x 2 threads; spike02 did not log CPU time).
So ~7x less wall and ~15x fewer CPU-seconds for the same 2M-step budget at the same 64-env batch structure; at 4096+ envs the ratio is far larger but PPO hyper-parameters would have to be re-tuned.

## 4. Residual PPO on MJX (hyper-parameters = spike02: 64 envs, horizon 200, bound 16 Nm, lr 3e-4, 10 epochs, mb 2048, tanh 128x128, gamma .99, lambda .95, clip .2, obs 16-d `spike_features`, reward as spike02; JAX reimplementation, `arena_mjx/ppo.py`)
Training: W0 (nominal) and W3 (Coulomb 1.5 + viscous 0.3), TRAIN_SEEDS 10000-10199 (protocol rule; half of each batch gets pulses), progress/checks on CAL seeds only.
Evaluation: **CPU arena evaluator** (`arena.eval.evaluate`, `split="test"`, TEST_SEEDS 0-49, logged in `results/.test_access.log`), `residual` controller with the saved actor as numpy MLP, mean action, residual bound
asserted <= 16 Nm on every run (held). Full table with paired CIs: `ppo_table.md`. RMSE in rad, +- 95% CI over the 50 trajectories.

| W | cond | spike02 RL (torch CPU, 2M, 5 seeds) | **MJX PPO 2M (5 seeds)** | **MJX PPO 20M (3 seeds)** | MJX 2M, fresh seeds (3 seeds) | spike03a C1 (tuned classical) | C0 teacher |
|---|---|---|---|---|---|---|---|
| W0 | clean | 0.0081 | **0.0081** | **0.0080** | 0.0081 | 0.0076 | 0.0097 |
| W0 | dist | 0.0082 | **0.0082** | **0.0081** | 0.0082 | 0.0078 | 0.0149 |
| W3 | clean | 0.0081 | **0.0081** | **0.0080** | 0.0081 | 0.0076 | 0.0219 |
| W3 | dist | 0.0082 | **0.0082** | **0.0081** | 0.0082 | 0.0077 | 0.0237 |

(all +-0.0010-0.0011.) Paired over the 50 trajectories: MJX 2M vs spike02 RL: difference +0.00001 to +0.00004 rad (0.1-0.4%), CIs touch 0 / are < 1e-4 wide: indistinguishable
(the 5 MJX seeds span 0.0079-0.0082, spike02's 5 seeds ~1%). MJX 2M vs untuned teacher C0: W0 -17% clean / -45% dist, W3 -63% / -65%, all WIN (same as spike02's 17/45/63/65%).
MJX vs C1: C1 is better by 5.4-6.8% in every cell, CI includes 0 (e.g. W3 clean -0.00048 [-0.00112, +0.00016]) - same conclusion as spike03a (tuned classical ties/slightly beats residual PPO).
**More budget (20M, 10x) does not help**: 0.0080 vs 0.0081 (spike02-vs-MJX20M paired diff +0.00008..0.00012, CI excludes 0 but is ~1%); calibration RMSE 0.00865 (2M) vs 0.00865-0.00882 (20M).
The residual PPO has plateaued at the ~0.008 floor by 2M steps (inference: the floor is set by the 16 Nm residual bound / 200 Hz loop / the initial-error transient, as spike02/03a already argued, not by sample count).
Training seed pool (TRAIN_SEEDS only vs spike02's 500000+ seeds) makes no difference (column 5).
Training cost per policy: 2M steps 22-23 s (+compile), 20M 228 s (~88k steps/s with 64 envs); logs in `policies/*_summary_*.json`.

## Re-run
```
uv pip install --python .venv/bin/python "jax[cuda12]" "mujoco-mjx==3.17.0" optax
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
G="a GPU job queue systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 $PWD/.venv/bin/python"
$G scripts/mjx_parity.py --x64 0 --out results/mjx/parity_gpu_x64_0.json     # and --x64 1
$G scripts/mjx_bench.py --n 64,1024,4096,16384 --out results/mjx/bench_gpu_fp32.json
$G scripts/mjx_train_ppo.py --w W3 --seeds 0,1,2,3,4 --steps 2000000 --tag b2m --nenv 64   # also W0; --steps 20000000 --tag b20m (3 seeds)
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/mjx_eval_arena.py --tag b2m --w W3          # CPU arena evaluator on TEST_SEEDS
.venv/bin/python scripts/mjx_report.py b2m b20m b2m_fresh                                  # -> ppo_table.md
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/test_mjx_parity.py               # CPU only, ~1 min
```
Use `CUDA_VISIBLE_DEVICES=` (or `JAX_PLATFORMS=cpu`) for any jax code that must not touch the GPU.

## a GPU job queue ids
#80 probe; #83 parity fp32 (first run), #84 parity fp64, #90 parity fp32 re-run with error-growth columns (overwrote `parity_gpu_x64_0.json`); #88 / #89 bench fp32 / fp64;
#91 / #92 PPO 2M W3 / W0 (5 seeds); #99 / #100 PPO 20M W3 / W0 (3 seeds); #103 / #104 PPO 2M fresh seeds W3 / W0 (3 seeds);
#101 / #102 20M with 1024 envs: crashed at start (`rs.choice(TRAIN_SEEDS, 1024, replace=False)` with only 200 seeds; fixed to sample with replacement), re-queued as #112 / #113 and then CANCELLED (queue was blocked for >1 h by other sessions' jobs; this variant was optional, so it was not run - no 1024-env PPO result exists).

## Deviations / decisions (made without asking)
1. Reference is `joint_sines` (spike legacy), not the new Lissajous/glyph default (IK on CPU); needed for comparability with spike02/03a.
2. PPO is a JAX reimplementation, not a bit-copy of the torch code: same hyper-parameters, but different RNG streams / init draws, so numbers match in distribution (seed-to-seed spread ~1%), not per seed.
   Small known differences: `adv.std(ddof=1)` like torch; last minibatch of an epoch has 512 samples (as in spike02); Adam/optax defaults.
3. Main run uses TRAIN_SEEDS (200 seeds, protocol rule); spike02 used 500000+ seeds. Both give the same result (column 5), so the protocol-compliant one is kept as main.
4. The 1000-step fixed-torque check is open-loop on an unstable system; float32 cannot be held to a tight state tolerance there (see section 1). The decisive float32 checks are closed-loop metrics.
5. The CPU-benchmark and all timings were measured on a machine with load average 70-100; absolute numbers are noisy, ratios are large enough not to matter.
6. 20M budget used 64 envs (same batch structure as spike02, only more iterations, 1562) so that "more budget" is the only change.
7. Delay is implemented and parity-tested (6 steps) but PPO was only run on W0/W3 (as asked); W1/W2/W4 PPO runs are possible with a one-line change (`WK` in `mjx_train_ppo.py`).
