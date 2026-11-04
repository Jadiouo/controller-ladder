#!/bin/bash
cd "$(dirname "$0")/.."
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python"
$H spike05/calib_final.py > results/spike05/calib_final.log 2>&1
for v in dr hist; do $H spike05/run_rl5.py --variant $v --seeds 0,1,2 --steps 3000000 > results/spike05/run_rl_$v.log 2>&1; done
