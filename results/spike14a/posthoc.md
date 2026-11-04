### hold: episode-level overlap with R0 (C0) and R1 (n=500)

| method | success pct | gained vs R0 (R0 fail -> success) | lost vs R0 (R0 success -> fail) | converted of the R1-only episodes (R1 success and R0 fail, n=125) | success pct when R1 fails (n=70) |
|---|---|---|---|---|---|
| dec4 (mean of 3 seeds, rounded to majority) | 76.2 | 106 | 71 | 87/125 = 70% | 69% |
| dec4 seed 0 | 71.2 | 104 | 94 | 89/125 = 71% | 64% |
| ens_dec4 | 74.0 | 115 | 91 | 93/125 = 74% | 71% |
| lmlp1 s0 | 67.0 | 97 | 108 | 77/125 = 62% | 69% |
| gru4 s0 | 72.8 | 111 | 93 | 90/125 = 72% | 69% |
| C_map1_ms | 67.2 | 101 | 111 | 84/125 = 67% | 66% |
| HYB | 63.6 | 84 | 112 | 70/125 = 56% | 59% |

### new: episode-level overlap with R0 (C0) and R1 (n=200)

| method | success pct | gained vs R0 (R0 fail -> success) | lost vs R0 (R0 success -> fail) | converted of the R1-only episodes (R1 success and R0 fail, n=49) | success pct when R1 fails (n=40) |
|---|---|---|---|---|---|
| dec4 (mean of 3 seeds, rounded to majority) | 72.0 | 41 | 32 | 32/49 = 65% | 57% |
| dec4 seed 0 | 65.5 | 40 | 44 | 28/49 = 57% | 55% |
| ens_dec4 | 71.0 | 42 | 35 | 33/49 = 67% | 50% |
| lmlp1 s0 | 63.0 | 37 | 46 | 30/49 = 61% | 48% |
| gru4 s0 | 65.0 | 39 | 44 | 31/49 = 63% | 60% |
| C_map1_ms | 68.0 | 43 | 42 | 34/49 = 69% | 57% |
| HYB | 67.5 | 45 | 45 | 34/49 = 69% | 65% |

### R1 + noise diagnostic (holdout, n=500; post hoc): TRUE theta plus iid Gaussian noise of the size of the selected learned estimator's holdout error (ls 0.016, per-link log-mass 0.094, log-damping 0.320, friction 0.50e-3 Nm; noise x1 and x0.5)

| plant parameters given to the planner | success % |
|---|---|
| true theta | 86.0 |
| true theta + noise x0.5 | 74.0 |
| true theta + noise x1 | 65.0 |
| R0 fit | 69.2 |
| learned dec4 (seed 0) | 71.2 |
