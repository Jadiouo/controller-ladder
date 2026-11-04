#!/bin/bash
# FINAL R-SAC (hist), 3 training seeds in parallel, 7 equally spaced CAL checkpoints. Config from CAL pilots S1-S5 (all pilots damaged the teacher; this is the most conservative one: lr 1e-5, BC 1, fixed alpha 0.001, critic-only warm-up 150k steps).
cd "$(dirname "$0")/.."
C=results/spike07/best.json; S=".venv/bin/python spike07/rl7.py --algo sac --variant hist --classical $C --steps 1200000 --nck 7 --c_apex 0.3 --alpha_fixed 0.001 --bc 1 --actor_delay 150000 --warm 50000 --buf 200000 --lr 1e-5 --logstd0 -3.5 --sac_ls_max -3.0"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
for s in 0 1 2; do
$S --seeds \$s --out results/spike07/final_sac_s\$s.pkl > results/spike07/final_sac_s\$s.log 2>&1 &
done
wait"
echo done > .scratch/final2.done
