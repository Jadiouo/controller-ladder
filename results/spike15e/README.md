# spike15e: v9 locked TEST-D confirmation (seeds 40000-40499, n=500, run once)

## 0. Specification
Rules: docs/03_hypotheses.md section v9 (commit 800de6e, written before this run). Code: experiments/spike15e/ (eval15e.py, infer15e.py, stats15e.py). Frozen methods: B = c11/M3c/max (spike15c), C-ens = LAP/c11/M3c/hit06 (spike15d), R0, R1, L-ens; descriptive B9, R0+c11/centre LAP. Weights: results/spike15c/models/.
Pre-run commit: 800de6e95a5048de0f44f4a60b5861379d55c051 (code in experiments/spike15e/ was uncommitted at run time; committed together with these results).

## 1. Reproduction check (holdout 1100-1599, before TEST-D)
repro_hold.json: B 86.6 (ref 86.6), C-ens 79.6 (ref 79.6), R0 69.2, R1 86.0, L-ens 71.0; per-episode agreement with spike15c/15d 100 %. PASS. Holdout stage-1 plans partly hit the .scratch/s10b_plans cache (input-hash keyed), so holdout CPU seconds are low; TEST-D had no cache.

## 2. TEST-D
Access log: one line added (110 -> 111): `2026-11-02T21:54:34	experiments/spike15e/eval15e.py	v9 single locked TEST-D run, seeds=40000-40499`. Full numbers: table.md; per-episode: testd_episodes.json.

| | success % [95% CI] | vs R0 pp |
|---|---|---|
| R0 | 66.4 [62.2, 70.6] | - |
| R1 | 84.4 [81.2, 87.6] | +18.0 [13.0, 23.2] |
| L-ens | 70.8 [66.8, 74.6] | +4.4 [-1.4, 10.0] |
| B (learned) | 82.6 [79.2, 85.8] | +16.2 [11.0, 21.4] |
| C-ens (classical) | 77.0 [73.2, 80.6] | +10.6 [5.8, 15.4] |

Verdicts (v9 rule text applied literally; Holm on one-sided bootstrap p):
- H15 SUPPORTED (Holm p 0.0002). H15b SUPPORTED (0.0002). H16a SUPPORTED: B - C-ens +5.6 [0.6, 10.6], Holm p 0.032.
- H16b NOT supported: rule is "95 % CI upper < +10", upper = 10.6. Pre-written sentence: 「學習式優勢可能 ≥10 pp，論文照實改寫 K2」.

## 3. Notes and deviations (written by the commander after the run)
- **Inconsistency in the v9 card itself:** H16b's rule uses the two-sided 95 % CI upper bound, while the Holm step uses a one-sided p (H0: diff >= 10), which corresponds to the 90 % CI (upper 9.8, p 0.043). By the rule text H16b fails; by the one-sided test alone it would pass Holm (0.043 <= 0.05). We report the rule text as binding and disclose the conflict. Reading: the learned advantage is real (H16a) and its point estimate is +5.6 pp; a value of ~10 pp cannot be excluded at 95 %.
- The original agent was interrupted after the holdout run started; the commander ran the gated TEST-D step and statistics with the agent's code unchanged.
- MuJoCo "simulation unstable" warnings: 125,555 lines in the TEST-D log (4,621 in holdout). No non-finite outcome in any executed episode record; whether warnings come from planner model rollouts or executed plants is not separated in this run (follow-up on holdout only).
