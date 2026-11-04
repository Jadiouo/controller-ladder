#!/bin/bash
# round 3: round 2 showed (single seed) that only the 4x batch kept CAL near the init; KL stop, apex-coefficient alone, BC and cap bootstrap did not. Test more envs (effective sample size), lower lr, and the spec reward (c_apex 3) at the larger batch.
cd "$(dirname "$0")/.."
C=results/spike07/best.json; R=".venv/bin/python experiments/spike07/rl7.py --variant hist --classical $C --steps 2500000 --nck 4 --logstd0 -3.5 --seeds 100"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
$R --lr 1e-4 --c_apex 0.3 --nenv 256 --horizon 128 --mb 8192 --out results/spike07/diag/E1_env256.pkl > results/spike07/diag/E1_env256.log 2>&1 &
$R --lr 3e-5 --c_apex 0.3 --nenv 128 --mb 8192 --out results/spike07/diag/E2_batch4x_lr3e-5.pkl > results/spike07/diag/E2_batch4x_lr3e-5.log 2>&1 &
$R --lr 1e-4 --c_apex 3.0 --nenv 128 --mb 8192 --seeds 101 --out results/spike07/diag/E4_batch4x_apex3_s101.pkl > results/spike07/diag/E4_batch4x_apex3_s101.log 2>&1 &
$R --lr 1e-4 --c_apex 0.3 --nenv 128 --mb 8192 --seeds 101 --out results/spike07/diag/E5_D9rep_s101.pkl > results/spike07/diag/E5_D9rep_s101.log 2>&1 &
wait"
echo done > .scratch/diag3.done
