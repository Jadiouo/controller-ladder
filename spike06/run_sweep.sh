#!/bin/bash
cd "$(dirname "$0")/.."
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python"
$H spike06/sweep_dist.py "$@"
