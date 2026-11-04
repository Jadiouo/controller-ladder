#!/bin/bash
cd "$(dirname "$0")/.."
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 .venv/bin/python experiments/spike07/calib7.py robust --gens 8 --workers 4 --warm results/spike06/robust.json > results/spike07/calib_robust.log 2>&1
