#!/bin/bash
# step 3 (round 2, with CAL checkpoint selection every 60 iterations; round 1 = last iterate, in sweep_rl_round1/). small RL sweep, CAL only. 2 variants x lr{3e-5,1e-4} x logstd0{-3.5,-4.5}, 1 seed (100), 3M steps each.
cd "$(dirname "$0")/.."
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python"
for v in hist dr; do for lr in 1e-4 3e-5; do for ls in -3.5 -4.5; do
  $H spike06/rl6.py --variant $v --classical results/spike06/best.json --seeds 100 --steps 3000000 --lr $lr --logstd0 $ls --out results/spike06/sweep2_${v}_lr${lr}_ls${ls}.pkl > results/spike06/sweep2_${v}_lr${lr}_ls${ls}.log 2>&1
done; done; done
