# L6 imitation students on tracking (2-DoF, CTC+LQR teacher)

Scenario `tracking/clean` (Lissajous reference, 4 s, nominal plant), TEST_SEEDS 0-49 (n = 50, split="test"), 3 training seeds per student. Data: 200 teacher rollouts on TRAIN_SEEDS (DART sigma = 0.1 tau_limit), same file for every student. Observation: window of 3 past rows + current row of `Obs.to_vector()` (q, qd, ref.ee, ref.q, ref.qd, ref.qdd). Chunk 8, executes 4 then re-plans. Steps 15000, batch 256, AdamW + cosine, EMA 0.999, fp32.

## Parameters and training

| student | params | final train loss (mean of 3 seeds) | train device | train time / run (s) |
|---|---|---|---|---|
| bc | 553,942 | 0.29812 (loss is not comparable across kinds) | cuda | 24 |
| mip | 553,744 | 0.30480 (loss is not comparable across kinds) | cuda | 40 |
| diffusion | 553,744 | 0.22143 (loss is not comparable across kinds) | cuda | 26 |
| flow | 553,744 | 0.58834 (loss is not comparable across kinds) | cuda | 25 |

### Raw features (as specified)

| student | rmse per training seed (rad) | rmse mean +- sd over training seeds | ss_rmse | sat | paired diff vs teacher (seed-averaged student - teacher), 95% CI | verdict |
|---|---|---|---|---|---|---|
| **teacher (ctc_lqr)** | - | 0.01050 | 0.00080 | 0.0006 | - | - |
| bc | 0.01117, 0.01122, 0.01128 | 0.01123 +- 0.00004 | 0.00250 | 0.0000 | +0.00073 [-0.00004, +0.00150] (-7% improvement; negative = worse) | no difference |
| mip | 0.01117, 0.01108, 0.01137 | 0.01121 +- 0.00012 | 0.00230 | 0.0001 | +0.00071 [-0.00007, +0.00149] (-7% improvement; negative = worse) | no difference |
| diffusion (DDIM-10) | 0.01225, 0.02514, 0.02816 | 0.02185 +- 0.00690 | 0.01581 | 0.0000 | +0.01136 [-0.00917, +0.03188] (-108% improvement; negative = worse) | no difference |
| flow (10 steps) | 0.01156, 0.01163, 0.01148 | 0.01156 +- 0.00006 | 0.00454 | 0.0003 | +0.00106 [+0.00045, +0.00167] (-10% improvement; negative = worse) | worse |
| flow (1 step) | 0.01078, 0.01082, 0.01081 | 0.01080 +- 0.00002 | 0.00212 | 0.0001 | +0.00031 [-0.00014, +0.00075] (-3% improvement; negative = worse) | no difference |

### Extra input: tracking errors (q - ref.q, qd - ref.qd) appended to every row (`--rel-ref`)

| student | rmse per training seed (rad) | rmse mean +- sd over training seeds | ss_rmse | sat | paired diff vs teacher (seed-averaged student - teacher), 95% CI | verdict |
|---|---|---|---|---|---|---|
| **teacher (ctc_lqr)** | - | 0.01050 | 0.00080 | 0.0006 | - | - |
| bc | 0.01098, 0.01110, 0.01092 | 0.01100 +- 0.00007 | 0.00239 | 0.0001 | +0.00050 [+0.00007, +0.00094] (-5% improvement; negative = worse) | worse |
| mip | 0.01100, 0.01086, 0.01091 | 0.01093 +- 0.00006 | 0.00206 | 0.0002 | +0.00043 [-0.00003, +0.00089] (-4% improvement; negative = worse) | no difference |
| diffusion (DDIM-10) | 0.03551, 0.04167, 0.03257 | 0.03658 +- 0.00379 | 0.03150 | 0.0000 | +0.02609 [-0.02415, +0.07632] (-249% improvement; negative = worse) | no difference |
| flow (10 steps) | 0.01155, 0.01160, 0.01155 | 0.01157 +- 0.00002 | 0.00454 | 0.0002 | +0.00107 [+0.00042, +0.00172] (-10% improvement; negative = worse) | worse |
| flow (1 step) | 0.01082, 0.01089, 0.01077 | 0.01082 +- 0.00005 | 0.00217 | 0.0001 | +0.00033 [-0.00021, +0.00087] (-3% improvement; negative = worse) | no difference |

## Inference time (latency axis B), batch = 1, one chunk of 8 x 2 torques, 1 CPU thread, fp32

| student | network evaluations | GPU (RTX 5070 Ti) mean / p95 (ms) | CPU mean / p95 (ms) |
|---|---|---|---|
| bc | 1 | 0.22 / 0.23 | 0.12 / 0.12 |
| mip | 2 | 0.50 / 0.51 | 0.26 / 0.27 |
| diffusion (DDIM-10) | 10 | 3.38 / 3.46 | 1.60 / 1.63 |
| flow (10 steps) | 10 | 2.24 / 2.30 | 1.29 / 1.32 |
| flow (1 step) | 1 | 0.29 / 0.30 | 0.15 / 0.16 |

The controller calls the network once per 4 control steps (period 5 ms): per-step cost = mean / 4. The machine is shared and heavily loaded, wall times are inflated and noisy.

## Reading

* Every student is close to the teacher but none beats it: the paired CI of BC, MIP and 1-step flow includes 0 (cost +3 to +7 %), 10-step flow is worse by 10 %, DDIM-10 diffusion is 2-3x worse and unstable across training seeds (0.012 to 0.028). The teacher is exact on the clean nominal plant, so a student can at best match it (README 52).
* rmse is dominated by the start transient (teacher ss_rmse 0.0008 vs rmse 0.0105); students lose the steady-state precision (ss_rmse 0.002-0.016), i.e. what separates them from the teacher is the last digits of the feedback law, not the gross behaviour.
* Same backbone, data and budget: regression (BC), two-step regression (MIP) and the one-step flow (which is regression to the conditional mean) are indistinguishable; the stochastic multi-step samplers (10-step flow, DDIM-10) add variance and no accuracy here, consistent with arXiv 2512.01809 (no multimodality in this task; the data are a deterministic teacher + DART noise).
* Appending (q - ref.q, qd - ref.qd) changes nothing material (BC -2 %, MIP -2 %).
* Inference (per call, batch 1): BC 0.12 ms CPU, MIP 0.26, 1-step flow 0.15, 10-step flow 1.29, DDIM-10 1.60; on the GPU every row is about 1.7-2x slower (kernel-launch latency, result copy to host): CPU wins at batch 1 for these tiny nets.
* First run, disclosed (README 54): DART 0.02, 30 000 steps and no angle wrap gave rmse 0.16-0.24 (BC / MIP) and 0.4-1.4 (flow / diffusion) on TEST. The seeds whose reference is stored unwrapped (README 53) broke every student.
