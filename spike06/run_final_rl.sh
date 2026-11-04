#!/bin/bash
# step 3 final: configs selected on CAL by the round-2 sweep (per variant): hist lr3e-5 logstd0 -4.5 ; dr lr1e-4 logstd0 -4.5. 3 training seeds (0,1,2), 3M steps each, CAL checkpoint selection every 60 iterations.
cd "$(dirname "$0")/.."
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python"
$H spike06/rl6.py --variant hist --classical results/spike06/best.json --seeds 0,1,2 --steps 3000000 --lr 3e-5 --logstd0 -4.5 --out results/spike06/final_hist.pkl > results/spike06/final_hist.log 2>&1
$H spike06/rl6.py --variant dr --classical results/spike06/best.json --seeds 0,1,2 --steps 3000000 --lr 1e-4 --logstd0 -4.5 --out results/spike06/final_dr.pkl > results/spike06/final_dr.log 2>&1
