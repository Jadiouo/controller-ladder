#!/bin/bash
# FINAL PPO-fixed (hist) and R-oracle, 3 training seeds each, 7 equally spaced CAL checkpoints, config chosen from the CAL-only diagnosis (D9/E5): 128 envs x 256, minibatch 8192, lr 1e-4, log-std0 -3.5, c_apex 0.3, gamma .995, fail -3, 2.5M steps.
# (first attempt was OOM-killed at 8 procs / 10G at 18:37 with 6 PPO + 2 SAC; this is the rerun with 6 PPO + 1 SAC pilot) plus 1 conservative SAC pilot (CAL only; fixed alpha because auto-alpha diverged to 337 in S3).
cd "$(dirname "$0")/.."
C=results/spike07/best.json; P=".venv/bin/python spike07/rl7.py --classical $C --steps 2500000 --nck 7 --lr 1e-4 --c_apex 0.3 --nenv 128 --mb 8192 --logstd0 -3.5"
S=".venv/bin/python spike07/rl7.py --algo sac --variant hist --classical $C --steps 1200000 --nck 3 --c_apex 0.3 --seeds 100 --alpha_fixed 0.001 --bc 1 --actor_delay 150000 --warm 50000 --buf 200000"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
for s in 0 1 2; do
$P --variant hist --seeds \$s --out results/spike07/final_ppo_s\$s.pkl > results/spike07/final_ppo_s\$s.log 2>&1 &
$P --variant oracle --seeds \$s --out results/spike07/final_oracle_s\$s.pkl > results/spike07/final_oracle_s\$s.log 2>&1 &
done
$S --lr 1e-5 --logstd0 -3.5 --sac_ls_max -3.0 --out results/spike07/diag/S5_sac_fixa_cons.pkl > results/spike07/diag/S5_sac_fixa_cons.log 2>&1 &
wait"
echo done > .scratch/final1.done
