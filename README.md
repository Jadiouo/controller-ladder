# Controller Ladder arena

A MuJoCo benchmark and shared skeleton for asking one question under a fair, pre-specified protocol: **what extra ingredient does a learned controller need before it beats its classical teacher?** Robots, tasks, perturbations ("weaknesses"), controllers (PID, computed-torque + LQR, iLQR, iLQR-MPC, MPPI, residual RL, imitation students such as BC, MIP, diffusion and flow matching) and the evaluation protocol plug into the same interface, so every comparison is recorded and tabulated the same way.

Paper: `docs/paper/arxiv/main.pdf` (LaTeX sources and figure script in `docs/paper/arxiv/`; arXiv link to be added). Portfolio site and in-browser MuJoCo controller arena: live at https://jadiouo.github.io/controller-ladder/ (source `web/`; serve locally with `python -m http.server -d web`). Status: research code accompanying a paper; numbers in the paper are indexed in `docs/canonical_numbers.md`.

## Install

Python >= 3.10. Dependencies: numpy, scipy, mujoco, pillow (ffmpeg binary optional, for mp4).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest
```

Some experiments (`spike*/`, `arena/learn`, `arena_mjx`, `scripts/mjx_*`) additionally need torch and/or JAX/MJX (and a GPU for the larger runs); they are not needed for the tests.

## Test

```bash
python -m pytest -q                        # all tests, serial
python -m pytest -q tests/test_spike_repro.py   # numerical reproduction of spike01/02 (needs spike/ and spike02/ unchanged)
```

Unit tests never write `results/.test_access.log`; real evaluations on the locked seed sets do (see `docs/provenance.md`).

## Reproduce the paper tables

* Every experiment is one directory `spikeNN/` (code) with outputs in `results/spikeNN/` (`README.md` with the pre-specification and deviations, `table.md`, JSON summaries, per-episode `.npy` arrays where small).
* `docs/experiments.tsv` lists, per run, the hypothesis ids, seeds, command and result file; `docs/canonical_numbers.md` maps every number used in the paper to a run and command; `docs/paper/claims_evidence.md` (section 5) gives the "regenerate table" command per result.
* Cheap path (no simulation): re-build a table from the stored per-episode data, e.g. `python spike15e/stats15e.py testd`, `python spike10b/report10b.py`, `python spike13a/report13.py`, `python spike15b/report15.py`, `python docs/paper/stats/equivalence.py`.
* Full re-runs are listed in each `results/spikeNN/README.md` ("Re-run"). Some are long (hours; RL training) and a few use locked test seeds that must be evaluated **once**: do not repeat TEST-B/C/D runs (see below).
* The frozen whip estimator weights used on TEST-D are included (`results/spike15c/models/`, 15 MB). Other weights, training-data shards and large arrays are **not** included (size); see `RELEASE_NOTES.md` for the list. Tables derived from them are included.

Demo (about a minute, writes tables and mp4 to `results/arena_demo/`):

```bash
python scripts/demo_ladder.py
```

Video backend: `MUJOCO_GL=egl` (set automatically by `arena.viz`).

## Evaluation protocol in one paragraph

Seeds are split into disjoint sets: CAL (tuning), HOLD / NEW (model selection and fresh checks) and locked TEST sets (0-49, 20000-20499, 30000-30499, 40000-40499) evaluated once per pre-specified hypothesis. Hypothesis cards (`docs/03_hypotheses.md`) were frozen before the corresponding confirmatory runs; the commit order evidence is documented in `docs/provenance.md`. Confidence intervals are paired; margins are stated in the cards.

## Repository layout

```
arena/        robots, tasks, perturbations, controllers, evaluation, viz, learning (BC/MIP/diffusion/flow)
arena_mjx/    JAX/MJX vectorised environment and PPO (GPU)
spike*/       experiment code, one folder per study (see docs/experiments.tsv)
scripts/      demos and MJX benchmarks
tests/        unit tests and numerical-reproduction tests
docs/         hypotheses, rulers, experiments table, canonical numbers, paper support files, ARCHITECTURE.md
results/      small result files (tables, JSON, logs)
```

Architecture, how to add a task / controller / perturbation, and the design decisions: `docs/ARCHITECTURE.md`.

## License and citation

MIT: see `LICENSE`. Citation metadata: `CITATION.cff`.
