# spike12a: do generative policies beat MLP regression on a multimodal teacher? (feasibility, no TEST seed)

Numbers: `table.md` (from `experiments/spike12a/report12.py`). Code: `experiments/spike12a/` (c12 rollout, run12 driver, train12 trainer, disp12 dispersion/latency, sel12 selection). Seeds used: training >= 500000, CAL 1000-1099 (selection), final 700200-700399 (n=200). `results/.test_access.log` untouched.

Setup: whip, abc x1, SUBF=4; student obs/output as spike11a "canon_dagger3" (prior-residual on the canonical grid plan, CTC inner loop) but outputs a 16-step chunk, executes 8 (action chunking, fresh noise per chunk; fixed-noise-per-episode tried on CAL: worse, not used). BC only (no DAgger).
Data (equal size): D-multi = 400 scenarios x 4 CMA-ES streams (standard teacher), 1600 episodes, 432,608 samples; D-canon = same 400 scenarios x 4 canonical-teacher plans, 1600 episodes, 470,650 samples. All ~0.62 M params (BC 621,039; others 622,480). 3 training seeds per cell, checkpoint (10k/20k/40k steps) chosen on CAL per seed.
Training on CPU (not GPU): a GPU job queue queue was behind ~5 x 50-min jobs of another project when checked 25 min after submission; fell back to CPU (deviation from the "1 h" rule, decided on the estimated wait).

Verdict: primary rule NOT met (best generative on D-multi: Flow-1 +5.0 pp vs MLP, CI [-0.3, +10.2]; Flow-10 +3.8 [-1.0, +8.7]; Diffusion -3.2). Secondary: canonicalising the teacher (MLP on D-canon 51.3 %) beats every generative student on D-multi (<= 30.7 %) by >= 20 pp. See table.md for all CIs.
Caveats: single data scale, BC only, student variance across seeds 5-15 pp (Diffusion D-multi 28.5/13.5/25.5), CAL selection noisy (n=100).
