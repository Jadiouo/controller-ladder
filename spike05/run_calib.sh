#!/bin/bash
# T-robust and T-adapt CMA-ES in parallel (2 workers each), calibration seeds only. One lock hold.
cd "$(dirname "$0")/.."
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 bash -c '
.venv/bin/python spike05/calib_classical.py robust --cap 120 --pop 10 --gens 20 --workers 2 > results/spike05/calib_robust.log 2>&1 &
.venv/bin/python spike05/calib_classical.py adapt --cap 120 --pop 10 --gens 20 --workers 2 > results/spike05/calib_adapt.log 2>&1 &
wait'
