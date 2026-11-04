#!/bin/bash
# R-SAC final, second attempt. The first SAC pilots S1-S5 and the first final attempt (aborted/) were confounded: the actor's initial residual was ~0.8-1 Nm (gain-0.01 output init) and CAL success was 0-4% already at the first checkpoint. Now the mean head is zero-initialised.
# CAL-only sweep of 2 configs x 3 training seeds (budget counted): A lr 3e-5 BC 1 ; B lr 1e-4 BC 3. The config with the higher mean CAL success of the selected checkpoints becomes R-SAC (decided by merge_final.py from CAL numbers only).
cd "$(dirname "$0")/.."; export MALLOC_ARENA_MAX=2
C=results/spike07/best.json
S=".venv/bin/python experiments/spike07/rl7.py --algo sac --variant hist --classical $C --steps 1200000 --nck 7 --c_apex 0.3 --alpha_fixed 0.001 --actor_delay 150000 --warm 50000 --buf 200000 --logstd0 -3.5 --sac_ls_max -3.0"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
for s in 0 1 2; do
$S --lr 3e-5 --bc 1 --seeds \$s --out results/spike07/final_sacA_s\$s.pkl > results/spike07/final_sacA_s\$s.log 2>&1 &
$S --lr 1e-4 --bc 3 --seeds \$s --out results/spike07/final_sacB_s\$s.pkl > results/spike07/final_sacB_s\$s.log 2>&1 &
done
wait"
echo C2 > .scratch/finalC2.done
