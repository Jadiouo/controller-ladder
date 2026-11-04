#!/bin/bash
# round 4 (PPO): hypothesis 2 = reward under-weights failure (gamma .995, fail -3 ~ 3 hits) -> drift to risky high-hit-rate play. Plus 2 SAC pilot runs (CAL-only sweep).
cd "$(dirname "$0")/.."
C=results/spike07/best.json; P=".venv/bin/python spike07/rl7.py --variant hist --classical $C --steps 2500000 --nck 4 --logstd0 -3.5 --lr 1e-4 --c_apex 0.3 --nenv 128 --mb 8192"
S=".venv/bin/python spike07/rl7.py --algo sac --variant hist --classical $C --steps 1500000 --nck 3 --logstd0 -3.5 --lr 3e-5 --c_apex 0.3 --seeds 100"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
$P --gamma 0.999 --r_fail 20 --seeds 100 --out results/spike07/diag/F1_g999_f20_s100.pkl > results/spike07/diag/F1_g999_f20_s100.log 2>&1 &
$P --gamma 0.999 --r_fail 20 --seeds 101 --out results/spike07/diag/F2_g999_f20_s101.pkl > results/spike07/diag/F2_g999_f20_s101.log 2>&1 &
$P --gamma 0.995 --r_fail 20 --seeds 100 --out results/spike07/diag/F3_g995_f20_s100.pkl > results/spike07/diag/F3_g995_f20_s100.log 2>&1 &
$S --bc 0 --out results/spike07/diag/S1_sac_bc0.pkl > results/spike07/diag/S1_sac_bc0.log 2>&1 &
$S --bc 1 --out results/spike07/diag/S2_sac_bc1.pkl > results/spike07/diag/S2_sac_bc1.log 2>&1 &
wait"
echo done > .scratch/diag4.done
