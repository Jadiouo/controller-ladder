#!/bin/bash
# one heavy job at a time (shared flock), 6G memory cap, 2 torch threads, CPU only. Run from repo root.
cd "$(dirname "$0")/.."
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python"
[ -s results/spike03b/calib.json ] || $H spike03b/calib.py > results/spike03b/calib.log 2>&1
$H spike03b/run_bc.py --cond nominal --seeds 0 > results/spike03b/run_bc_nominal.log 2>&1
for c in restitution mass sensing; do
  $H spike03b/run_bc.py --cond $c > results/spike03b/run_bc_$c.log 2>&1
  $H spike03b/run_rl.py --cond $c --seeds 0,1,2 --steps 3000000 > results/spike03b/run_rl_$c.log 2>&1
done
$H spike03b/ctrl_teacher.py > results/spike03b/ctrl_teacher.log 2>&1
.venv/bin/python spike03b/report.py > /dev/null
