#!/bin/bash
# PPO collapse diagnosis (CAL + training seeds only). 6 single-factor / combined runs, hist variant, 1 training seed (100), 2.5M steps, 4 equally spaced CAL checkpoints. best-classical = results/spike07/best.json
cd "$(dirname "$0")/.."; mkdir -p results/spike07/diag
C=results/spike07/best.json; R=".venv/bin/python spike07/rl7.py --variant hist --classical $C --seeds 100 --steps 2500000 --nck 4 --lr 1e-4 --logstd0 -3.5"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
$R --out results/spike07/diag/D0_base.pkl > results/spike07/diag/D0_base.log 2>&1 &
$R --cap_bootstrap 1 --out results/spike07/diag/D2_capboot.pkl > results/spike07/diag/D2_capboot.log 2>&1 &
$R --nenv 128 --mb 8192 --out results/spike07/diag/D3_batch4x.pkl > results/spike07/diag/D3_batch4x.log 2>&1 &
$R --target_kl 0.01 --out results/spike07/diag/D4_kl.pkl > results/spike07/diag/D4_kl.log 2>&1 &
$R --bc 10 --out results/spike07/diag/D5_bc.pkl > results/spike07/diag/D5_bc.log 2>&1 &
$R --cap_bootstrap 1 --nenv 128 --mb 8192 --target_kl 0.01 --out results/spike07/diag/D6_combo.pkl > results/spike07/diag/D6_combo.log 2>&1 &
wait"
