#!/bin/bash
# step 2: CMA-ES retune of T-robust and T-adapt on the new distribution, CAL only. 1 worker each (2 total), one lock hold, equal budget (pop 10, 8 generations + start = 81 evals each).
cd "$(dirname "$0")/.."
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 bash -c '
.venv/bin/python spike06/calib6.py robust --gens 8 --workers 1 --warm results/spike05/robust.json > results/spike06/calib_robust.log 2>&1 &
.venv/bin/python spike06/calib6.py adapt --gens 8 --workers 1 --warm results/spike05/adapt.json > results/spike06/calib_adapt.log 2>&1 &
wait'
