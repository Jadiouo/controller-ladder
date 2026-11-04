# Pre-specification audit (review r1, M2; round-2 update 2026-11-03 adds spike15a-15e)

Method: `git log --format='%h %ci %s' -- <file>` plus `git show <commit>:docs/03_hypotheses.md` to confirm the card text really contains the rule at that commit. Timestamps are local commit dates (author clock, no external timestamping; this repo has no remote). Verdict meanings: **YES** = rule text is in a commit strictly earlier than the first commit containing the result; **UNPROVABLE** = rule and result arrive in the same commit (or the rule only exists in a task prompt), so git cannot show the order, only the README's own statement; **NO** = rule changed after results (found none; one amendment is disclosed, see spike04).

| spike | rule location (file @ commit, time) | result commit (time) | order holds? |
|---|---|---|---|
| spike01 / spike02 (H1-H4) | `docs/03_hypotheses.md` v1/v2 @ b4185d3 (2026-10-25 00:47) | b4185d3 (same snapshot commit) | UNPROVABLE (same commit). Known deviation: W1/W2 strengths set after a teacher-only look at TEST (spike02) |
| spike03a (H5) | docs/03 v2 @ b4185d3 (10-25 00:47) | b4185d3 (same) | UNPROVABLE (same commit) |
| spike03b (H6, juggling pilot) | docs/03 v2 H6 @ b4185d3 (10-25 00:47) | 4f7b9dd (10-25 02:47) | YES (2 h earlier). Pilot; PPO recipe pre-lesson |
| spike04 (H7) | `results/spike04/H7.md` @ c1c12bc (10-25 07:21); no H7 heading in docs/03 at any commit | c1c12bc (same) | UNPROVABLE, and H7.md contains an "Amendment 1" written after seeing the N1 delay break old C1 (disclosed in the file) -> **flag in skeleton: rule amended after results** |
| spike05 (H8) | docs/03 v3 @ 81def21 (10-25 03:12) | 434309a (10-25 07:29) | YES. Test-log entry for spike05 was added retroactively (README.md:115-117) |
| spike06 (H8b) | docs/03 v4 @ 983c5b9 (10-30 10:49) | 63cf540 (10-30 14:36) | YES. Deviations: checkpoint selection added after round 1 (s06 deviation 1) |
| spike07 (H9, H10) | docs/03 v5 @ c84cb0a (10-30 16:20) | f852c82 (10-30 20:24) | YES. Deviation: PPO not fixed before TEST |
| spike08_diag | no rule (diagnostic) | a434157 | not applicable |
| spike09b (H9w, H10w) | docs/03 v6 @ b4be1ec (10-30 22:42) | c01e279 (10-31 00:27) | YES (1 h 45 min earlier) |
| spike10b (H11, H12) | docs/03 v7 @ ea8dffe (10-31 11:14), v7.1 @ 9c982c0 (10-31 11:15) | db07b4d (10-31 19:02) | YES. Dose curve run after the locked TEST-B (deviation 3); plant step changed after CAL tuning (deviation 1) |
| spike11a | rule "best student >= T-adapt3-full + 10 pp on CAL" stated "in the task" (README:5), i.e. a task prompt; progress.md decision log cf8d529 (10-31 19:47) has the decision but not the numeric rule | 584ed2d (10-31 22:02) | UNPROVABLE (rule text only in README of the result commit) |
| spike12a | progress.md log 25fa8bc (10-31 22:10) states the primary rule (D-multi: best generative >= MLP +10 pp, CI excludes 0) | 3e1ec31 (11-01 00:47) | YES for the primary rule only; secondary comparisons are not pre-specified in git |
| spike13a | progress.md log d99f5bf (11-01 11:12) states the learned rule (>= +5 pp over init); README section 0 holds the full list | ccc10dc (11-01 16:54) | YES for the +5 pp rule; the rest of section 0 UNPROVABLE (same commit). Gate bias fell back to b=6, gate judged on n=500 instead of n=200 (disclosed) |
| spike13b | README section 0 ("written BEFORE any holdout / new-seed result", incl. the "group matters if CI excludes 0" rule); no earlier log entry with the rule | 78b6253 (11-01 19:40) | UNPROVABLE (same commit) |
| spike14a | progress.md log 2051e2d (11-01 19:51) states the primary rule (>= 10 pp and CI excludes 0); README section 0 holds details | 7141438 (11-02 02:39) | YES for the primary rule; details UNPROVABLE (same commit) |
| learn_demo (S3) | no card; hyper-parameters chosen after one TEST look (README:369) | 5ff9fb8 (10-25 08:33) | NO / not pre-specified (already flagged stale) |
| spike15a (H13, H13b, H14, TEST-C) | `docs/03_hypotheses.md` v8 @ d167ed6 (2026-11-02 10:58), v8.1 @ 67d3acf (11:04) | run logged 13:04:07 (`.test_access.log`); result commit 248bce0 (15:04) | **YES** (card committed on its own, 2 h before the run). v8.1 deviation: L-dec4 re-trained, accepted by the pre-written rule (bit-identical) |
| spike15b (S5/S6 gates, PPO rerun, S3 rerun) | README section 0 'written BEFORE any run' (author statement) | f82c94e (2026-11-02 16:57) | UNPROVABLE (rule and result in one commit). X1 (T-nom base) added after the results (labelled descriptive); S5/S6 CAL widened to n=100/200 (disclosed) |
| spike15c (approach B, A gate) | README section 0 (author time ~15:20 on 11-02) | cea2e35 (18:07) | UNPROVABLE (same commit). The 9-network extension M9 was added after seeing the 3-network result (labelled post hoc) |
| spike15d (classical control for B) | README section 0 (author time ~18:20) | 2d028f7 (20:31) | UNPROVABLE (same commit). No deviation from section 0 reported |
| spike15e (H15, H15b, H16a, H16b, TEST-D) | `docs/03_hypotheses.md` v9 @ 800de6e (2026-11-02 20:32) | run logged 2026-11-02T21:54:34; result commit 83abec6 (2026-11-03 00:03) | **YES** for the card (committed 1 h 22 min before the run). The code in `spike15e/` was uncommitted at run time and committed with the results. The card itself is internally inconsistent for H16b (two-sided 95% rule vs one-sided Holm p); the rule text was applied |

Result: no case where git shows a rule committed after its result. But six spikes (01, 02, 03a, 04, 11a, 13b) cannot be shown pre-specified by git because rule and result share a commit or the rule lives in a task prompt, and for 13a/14a only the headline rule (not the full section 0) is provable. spike04's H7 was amended after seeing results, and learn_demo was never pre-specified. File modification times (13a-14a READMEs are newer than their result arrays) do not help: the READMEs were appended after the runs.

Consequence for the text: "pre-registered" must be replaced by "pre-specified in time-stamped, version-controlled hypothesis cards", with the 7 spikes of the table above not backed by an earlier commit marked. An inexpensive fix for the future (v8 onwards): commit the card on its own before any run.

Round-2 result (2026-11-03): the cards v8, v8.1 and v9 were each committed on their own before the run that they govern (spike15a, spike15e), which is the fix proposed above for v8 onwards. Spike15b, 15c and 15d still carry their rules only in the README that arrives in the same commit as the results, so they stay 'PRE-README' (not git-provable) in `claims_evidence.md`. Still no case where git shows a rule committed after its result. spike16a states its design in `results/spike16a/README.md` section 0; the spike is committed (9fe5682, 2026-11-03 03:58) and the README (rule) and the results arrived in that one commit, and the spike16a code was uncommitted when the run started, so the order is UNPROVABLE from git ([PRE-README]). spike15f (a8b4895) is a diagnostic with no rule to order. Update v4 (R2-M4, R2-M6): the two rows below are added.

| spike | rule location (file @ commit, time) | result commit (time) | order holds? |
|---|---|---|---|
| spike15f (warning source, diagnostic) | none (instrumentation only; no hypothesis) | a8b4895 | n/a (no rule); n=100 holdout only, result: 92-98% of warnings from planner rollouts, 0 from executed plants |
| spike16a (catch, toss, Push-T mini tables) | `results/spike16a/README.md` section 0 @ 9fe5682 | 9fe5682 (2026-11-03 03:58) | UNPROVABLE (same commit; code uncommitted at run start). Breadth only, not used for K2 |
