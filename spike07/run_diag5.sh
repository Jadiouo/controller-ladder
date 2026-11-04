#!/bin/bash
# round 5: (PPO) more exploration noise for signal-to-noise (log-std0 -2.5 = ~1 Nm) ; (SAC) pilots S1 (bc 0) and S2 (bc 1) destroyed the teacher (CAL 0% / 26% at 0.5M, residual 9.6 / 0.7 Nm): try noisier policy so the critic sees a real action slope, plus BC, lower actor lr, longer critic-only warmup.
cd "$(dirname "$0")/.."
C=results/spike07/best.json; P=".venv/bin/python spike07/rl7.py --variant hist --classical $C --steps 2500000 --nck 4 --lr 1e-4 --c_apex 0.3 --nenv 128 --mb 8192 --seeds 100 --logstd0 -2.5"
S=".venv/bin/python spike07/rl7.py --algo sac --variant hist --classical $C --steps 1200000 --nck 3 --c_apex 0.3 --seeds 100 --logstd0 -2.5 --sac_ls_max -2.0 --tgt_ent -3.5"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
$P --out results/spike07/diag/G1_ls-2.5.pkl > results/spike07/diag/G1_ls-2.5.log 2>&1 &
$P --gamma 0.999 --r_fail 20 --out results/spike07/diag/G2_ls-2.5_g999_f20.pkl > results/spike07/diag/G2_ls-2.5_g999_f20.log 2>&1 &
$S --lr 3e-5 --bc 1 --out results/spike07/diag/S3_sac_noisy_bc1.pkl > results/spike07/diag/S3_sac_noisy_bc1.log 2>&1 &
$S --lr 1e-5 --bc 0.2 --actor_delay 150000 --warm 50000 --out results/spike07/diag/S4_sac_noisy_lr1e-5.pkl > results/spike07/diag/S4_sac_noisy_lr1e-5.log 2>&1 &
wait"
echo done > .scratch/diag5.done
