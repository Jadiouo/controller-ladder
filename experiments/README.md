# Experiments

One folder per study. Folder names are historical ("spike" = a time-boxed experiment) and are kept unchanged so that they match
`results/<folder>/` (outputs), `docs/experiments.tsv` (per-run commands) and the hypothesis cards in `docs/03_hypotheses.md`.
The first study, `spike/`, writes to `results/spike01/` (it was called spike01 when it ran).

Run everything from the repository root, e.g. `.venv/bin/python experiments/spike13a/report13.py`.
Each study's pre-specification, deviations and "Re-run" commands are in `results/<result folder>/README.md`.
Scripts find the repository root from their own location, so the working directory only matters for the few that write to `.scratch/`.

| Folder | Research question (one sentence) | Results | Paper setting |
|---|---|---|---|
| `spike/` | Can a BC-MLP student of a mismodeled classical teacher beat the teacher? (H1) | `results/spike01/` | pilot, not a setting |
| `spike02/` | Does residual RL on a teacher with un-modelled weaknesses beat the teacher? (H2-H4) | `results/spike02/` | pilot (legacy; S1 uses spike03a) |
| `spike03a/` | Do tuned classical opponents tie residual PPO on 2-DoF tracking? (H5) | `results/spike03a/` | S1 |
| `spike03b/` | Does the mirror-law teacher beat residual RL on ball juggling with one known error? (H6) | `results/spike03b/` | S4 |
| `spike04/` | Does the tuned classical controller still tie residual RL under sensor noise and delay? (H7) | `results/spike04/` | S2 |
| `spike05/` | Juggling with an unknown, per-episode random plant error (H8) | `results/spike05/` | excluded (RL recipe defective, classical saturated) |
| `spike06/` | Juggling with a wide, non-saturating unknown error (H8b) | `results/spike06/` | S5 (legacy rows) |
| `spike07/` | Oracle headroom and fair RL on juggling, distribution C (H9, H10) | `results/spike07/` | S5 |
| `spike08_diag/` | Diagnostics: headroom inside error cells, teacher sensitivity to torque noise | `results/spike08_diag/` | support |
| `spike09a/` | Whip task: error distribution, robust and adaptive classical tuning on CAL seeds (no TEST) | `results/spike09a/` | S6 (setup) |
| `spike09b/` | Whip, idealised: learned estimator and residual RL vs best classical (H9w, H10w) | `results/spike09b/` | S6 |
| `spike10a/` | Whip, realistic-condition layer (sensor noise, friction, mass) and classical baselines on CAL and holdout | `results/spike10a/` | S7 (setup) |
| `spike10b/` | Whip, realistic: learned vs classical identification, residual RL (H11, H12) | `results/spike10b/` | S7 |
| `spike11a/` | Privileged-teacher distillation on the realistic whip (feasibility) | `results/spike11a/` | S8 |
| `spike12a/` | Do generative policies beat MLP regression on a multimodal teacher? | `results/spike12a/` | S8 |
| `spike13a/` | Is residual RL "not learned" because of the pipeline, or because nothing is learnable? | `results/spike13a/` | S7 (PPO) |
| `spike13b/` | Headroom decomposition and fragility sweep on the realistic whip | `results/spike13b/` | S7 |
| `spike14a/` | Estimator duel: learned vs strengthened classical parameter estimation | `results/spike14a/` | S7 |
| `spike15a/` | Locked confirmation of K3 on TEST-C | `results/spike15a/` | S7 (TEST-C) |
| `spike15b/` | Re-run the learned side of S5, S6 and S3 with a gate-validated exploration scale | `results/spike15b/` | S3, S5, S6 |
| `spike15c/` | Whip "fair learning side" trial: can any learned method eat part of the oracle headroom? | `results/spike15c/` | S7 |
| `spike15d/` | Classical control for spike15c approach B (robust plan selection with a classical estimator) | `results/spike15d/` | S7 |
| `spike15e/` | Locked TEST-D confirmation (v9) | `results/spike15e/` | S7 (TEST-D) |
| `spike15f/` | Source of the MuJoCo QACC warnings and divergence in executed whip episodes | `results/spike15f/` | support |
| `spike16a/` | Fair-protocol mini tables for catch, toss and Push-T (appendix) | `results/spike16a/` | appendix, tasks without a learned comparison |

Settings S1-S8 are defined in the paper (Table 1) and enumerated in `paper/support/claims_evidence.md`.
Some folders import their predecessors read-only (for example `spike04` -> `spike03a` -> `spike02` -> `spike`), so move them together.
