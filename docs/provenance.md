# Provenance of the pre-specification (hypothesis cards)

Paper contribution K1 states that the hypothesis cards were written and frozen BEFORE the confirmatory runs. The evidence is the commit order of the private development repository. This public repository is a fresh single-snapshot export (no history, neutral author), so the order is documented here instead. Only the abbreviated commit hash, the author-date (ISO 8601, local time zone as recorded) and the commit subject are listed. No names, e-mails or trailers are included.

**Verification.** The commit hashes below belong to the private repository. The authors can provide the private repository (or a signed `git log` / bundle) to reviewers on request, so that the hash chain and the ordering "card commit < result commit" can be checked independently. Until then these tables are an assertion by the authors, not an independently verifiable proof.

Subjects are copied verbatim from the private history (some are about the version of the card, some about the spike added in the same commit). Card contents (`docs/03_hypotheses.md`) are included in this repository in their final form (v9); earlier versions v1-v4 are folded into the snapshot commit `b4185d3`.

## 1. Commits touching `docs/03_hypotheses.md` (newest first)

| commit | author date | subject |
|---|---|---|
| `800de6e` | 2026-11-02T20:32:43+08:00 | Hypothesis card v9: locked TEST-D confirmation of robust plan selection (H15, H15b, H16a, H16b) |
| `67d3acf` | 2026-11-02T11:04:55+08:00 | Hypothesis card v8.1: L-dec4 retrain acceptance rule before TEST-C |
| `d167ed6` | 2026-11-02T10:58:17+08:00 | Hypothesis card v8: locked TEST-C confirmation of K3 (H13, H13b, H14) |
| `9c982c0` | 2026-10-31T11:15:39+08:00 | Hypothesis card v7.1 after white-team review (dose curve, OOD, TEST-B) |
| `ea8dffe` | 2026-10-31T11:14:23+08:00 | Add spike10a (whip realism headroom) and hypothesis card v7 |
| `b4be1ec` | 2026-10-30T22:42:44+08:00 | Add spike09a (whip headroom diagnostic) and hypothesis card v6 |
| `c84cb0a` | 2026-10-30T16:20:22+08:00 | Hypothesis card v5: framing decision, deviation table, H9/H10 for spike07 |
| `983c5b9` | 2026-10-30T10:49:30+08:00 | Fix tracking reference IK: unwrap angles and avoid branch jumps |
| `81def21` | 2026-10-25T03:12:10+08:00 | Add iLQR, iLQR-MPC, MPPI controllers on nominal model |
| `b4185d3` | 2026-10-25T00:47:56+08:00 | Snapshot: plan, hypotheses, spikes 01-03a |

Card version to first commit containing its section (found with `git log -S"<heading>"` on the private repo):

| card version | first commit | author date | frozen before |
|---|---|---|---|
| v1 / v2 (v2 revision after white-team round 2) | `b4185d3` | 2026-10-25T00:47:56+08:00 | spike03 results |
| v3 | `81def21` | 2026-10-25T03:12:10+08:00 | spike05 results |
| v4 | `983c5b9` | 2026-10-30T10:49:30+08:00 | spike06 results |
| v5 | `c84cb0a` | 2026-10-30T16:20:22+08:00 | any spike07 result |
| v6 | `b4be1ec` | 2026-10-30T22:42:44+08:00 | spike09b RL / TEST results |
| v7 | `ea8dffe` | 2026-10-31T11:14:23+08:00 | spike10b learned-method / TEST results |
| v7.1 | `9c982c0` | 2026-10-31T11:15:39+08:00 | spike10b start |
| v8 | `d167ed6` | 2026-11-02T10:58:17+08:00 | spike15a (TEST-C) |
| v8.1 | `67d3acf` | 2026-11-02T11:04:55+08:00 | TEST-C |
| v9 | `800de6e` | 2026-11-02T20:32:43+08:00 | spike15e (TEST-D) |

Caveats the authors disclose: (a) v6 and v7 were committed in the same commit as the diagnostic spike (09a / 10a) that motivated them, so the order inside that commit is not provable from git alone; (b) the "frozen before" column is taken from the card headings, and is checked against the result commits in section 2 (card commit earlier than the first confirmatory result commit: v7.1 `9c982c0` < spike10b `db07b4d`; v8.1 `67d3acf` < spike15a `248bce0`; v9 `800de6e` < spike15e `83abec6`).

## 2. Result commits per spike (first commit that added `results/<spike>/`)

Read together with `docs/experiments.tsv` (column `commit` is the latest commit touching the result directory). A confirmatory spike is pre-specified if its card commit in section 1 is earlier than the first result commit here.

| spike | first result commit | author date | subject |
|---|---|---|---|
| spike01 | `b4185d3` | 2026-10-25T00:47:56+08:00 | Snapshot: plan, hypotheses, spikes 01-03a |
| spike02 | `b4185d3` | 2026-10-25T00:47:56+08:00 | Snapshot: plan, hypotheses, spikes 01-03a |
| spike03a | `b4185d3` | 2026-10-25T00:47:56+08:00 | Snapshot: plan, hypotheses, spikes 01-03a |
| spike03b | `b4185d3` | 2026-10-25T00:47:56+08:00 | Snapshot: plan, hypotheses, spikes 01-03a |
| spike04 | `918ad69` | 2026-10-25T01:36:00+08:00 | Add arena skeleton: tasks, perturbations, controllers, eval, viz, tests |
| spike05 | `7294707` | 2026-10-25T05:35:24+08:00 | Add whip and Push-T tasks with classical teachers |
| spike06 | `63cf540` | 2026-10-30T14:36:40+08:00 | Add spike06 results (H8b refuted) and spike05 diagnosis |
| spike07 | `f852c82` | 2026-10-30T20:24:47+08:00 | Add spike07 results (H9 not supported, H10 refuted; RL still not learning) |
| spike08_diag | `a434157` | 2026-10-30T21:00:39+08:00 | Add spike08_diag (noise sensitivity, per-cell headroom): both negative |
| spike09a | `b4be1ec` | 2026-10-30T22:42:44+08:00 | Add spike09a (whip headroom diagnostic) and hypothesis card v6 |
| spike09b | `c01e279` | 2026-10-31T00:27:40+08:00 | Add spike09b results (whip: classical system ID 98%, H9w/H10w not supported) |
| spike10a | `ea8dffe` | 2026-10-31T11:14:23+08:00 | Add spike10a (whip realism headroom) and hypothesis card v7 |
| spike10b | `db07b4d` | 2026-10-31T19:02:08+08:00 | Add spike10b results (whip realism: H11/H12 not supported) |
| spike11a | `584ed2d` | 2026-10-31T22:02:06+08:00 | Add spike11a (privileged distillation pilot: not worth full run; teacher plans are multimodal) |
| spike12a | `3e1ec31` | 2026-11-01T00:47:36+08:00 | Add spike12a (generative vs MLP students on multimodal teacher: primary rule not met) |
| spike13a | `ccc10dc` | 2026-11-01T16:54:53+08:00 | Add spike13a (RL fairness: pipeline passes sanity gate; whip realism still not learned) |
| spike13b | `78b6253` | 2026-11-01T19:40:20+08:00 | Add spike13b (whip headroom is parameter estimation; classical ID fragile to actuator noise) |
| spike14a | `7141438` | 2026-11-02T02:39:06+08:00 | Add spike14a (estimator duel: no winner; learned halves parameter error without success gain) |
| spike15a | `248bce0` | 2026-11-02T15:04:08+08:00 | Add spike15a (v8 locked TEST-C: H13 supported, H13b not, H14 not confirmed after Holm) |
| spike15b | `f82c94e` | 2026-11-02T16:57:58+08:00 | Add spike15b (fixed-exploration PPO on S5/S6 and IL rerun on S3: no learned win) |
| spike15c | `cea2e35` | 2026-11-02T18:07:09+08:00 | Add spike15c (ensemble-robust plan selection eats whip headroom; asymmetric-critic PPO gate failed) |
| spike15d | `2d028f7` | 2026-11-02T20:31:54+08:00 | Add spike15d (classical ensemble control for robust plan selection: verdict uncertain) |
| spike15e | `83abec6` | 2026-11-03T00:03:37+08:00 | Add spike15e (v9 locked TEST-D: H15, H15b, H16a supported; H16b not) |
| spike15f | `a8b4895` | 2026-11-03T00:27:33+08:00 | Add spike15f (MuJoCo instability warnings come from planner model rollouts; no executed-plant divergence) |
| spike16a | `9fe5682` | 2026-11-03T03:58:32+08:00 | Add spike16a (catch/toss/Push-T fair-protocol mini tables on second machine; fill appendix E2) |
| learn_demo | `81a1fd7` | 2026-10-25T06:41:07+08:00 | Add MJX vectorized tracking env and JAX residual PPO |
| mjx | `81def21` | 2026-10-25T03:12:10+08:00 | Add iLQR, iLQR-MPC, MPPI controllers on nominal model |
| classical_ladder | `81def21` | 2026-10-25T03:12:10+08:00 | Add iLQR, iLQR-MPC, MPPI controllers on nominal model |
| arena_demo | `918ad69` | 2026-10-25T01:36:00+08:00 | Add arena skeleton: tasks, perturbations, controllers, eval, viz, tests |
| ball_tasks | `c94e57f` | 2026-10-25T03:30:21+08:00 | Add catch and toss tasks with classical teachers |
| whip_pusht | `7294707` | 2026-10-25T05:35:24+08:00 | Add whip and Push-T tasks with classical teachers |

## 3. Order of the confirmatory ("locked") evaluations

| locked set | card | card commit | spike | first result commit |
|---|---|---|---|---|
| TEST-B (20000-20499) | v7/v7.1 | `ea8dffe` / `9c982c0` | spike10b | see section 2 |
| TEST-C (30000-30499) | v8 / v8.1 | `d167ed6` / `67d3acf` | spike15a | see section 2 |
| TEST-D (40000-40499) | v9 | `800de6e` | spike15e | see section 2 |

## 4. Locked-seed access log (`results/.test_access.log`)

Every real evaluation on the locked TEST seed sets appends a line (timestamp, script, scenario, controller, seeds). Unit tests never write it. The log is included verbatim below the table as `results/.test_access.log` (path-rewritten where needed; see `RELEASE_NOTES.md`).

Note on time stamps: author dates in this document are those recorded by the committer's local clock; they are not notarised. A stronger proof (e.g. an external timestamp of the hash `800de6e`) was not made at the time.
