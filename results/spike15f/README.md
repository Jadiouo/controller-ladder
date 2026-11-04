# spike15f: source of the MuJoCo "Nan, Inf or huge value in QACC" warnings and divergence of actually executed whip episodes

Code: spike15f/instab.py (instrumentation, no existing file modified), spike15f/report.py. Data: holdout seeds 1100-1199 (n=100) only; test access log still 111 lines. Plan/estimator caches were redirected to a fresh dir so the planner really re-ran. Raw: .scratch/s15f/{R0,B,LAP}.pkl, numbers: summary.json.
Method: mujoco user-warning callback counts "Nan, Inf or huge value" warnings by phase: plan (CMA-ES stage-1 and member plans, Tuned3S.plan/get_picked), model-other (replay, closed-loop candidate scoring, R0 estimator fit), plant (inside arena.env mj_step). The plant shim also counts MuJoCo auto-resets (time jumps back) and non-finite qpos/qvel; run_episode is wrapped for per-execution counters. B / C-ens executed episode = the candidate actually selected (c11/M3c/max; LAP/c11/M3c/hit06).

| method | plan-phase warn (a) | other model warn (a) | plant warn (b) | executed episodes with reset/NaN/diverged | % | <0.5% | success % |
|---|---|---|---|---|---|---|---|
| R0 | 3250 | 276 | 0 | 0/100 | 0.0 | yes | 69.0 |
| L-ens | 6442 | 141 | 0 | 0/100 | 0.0 | yes | 71.0 |
| B | 6442 (shared pipeline with L-ens) | 141 | 0 | 0/100 | 0.0 | yes | 86.0 |
| C-ens | 9924 | 181 | 0 | 0/100 | 0.0 | yes | 79.0 |

All min_dist values of executed episodes are finite. Success matches the holdout reproduction (R0 69, L-ens 71, B 86, C-ens 79).

Offline-only executions (all 11 candidates per episode, used for selection evaluation, not a method's output): B/L-ens pipeline 1069 executions, 1 had a plant warning plus an auto-reset (seed 1132, candidate index 1, not selected by B or C-ens); C-ens pipeline 1087 executions, 0.

## Conclusion
- 92-98% of the warnings (R0 92%, others 98%) come from model rollouts inside the planner (stage-1 CMA-ES and member plans), the rest from model scoring/estimation. Inference: CMA-ES probes extreme knots/parameters that blow up the simulated chain and the optimiser discards them; they never reach the plant.
- The true plant produced 0 warnings in all four methods' executed episodes; diverged fraction is 0/100 each, satisfying <0.5% (resolution at n=100 is 1%; the rule is checked as zero events, the full-n check for holdout 500 and TEST-D 500 was not re-run).
- Inference: since no executed episode diverged, success rates are unaffected by divergence (excluding or counting them as failures gives identical numbers). The single plant reset occurred in a non-selected offline candidate execution; it could only matter if that candidate were chosen by some other selection rule.
- Caveat: warnings counted in this re-run (fresh planner), not the original 125,555-line log; the proportion by source is the finding, absolute counts differ with n and cache state.
