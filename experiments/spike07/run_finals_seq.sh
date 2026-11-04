#!/bin/bash
# FINAL runs, 3 processes per lock hold (two earlier attempts with 8 and 6 procs were OOM-killed at the 10 GB cap, losing ~1.5 h; one process grows to ~1.7 GB late in training).
#  A: R-PPO-fixed x3 seeds   B: R-oracle x3 seeds   C: R-SAC x3 seeds
# PPO/oracle config (D9/E5 from the CAL-only diagnosis): 128 envs x 256, minibatch 8192, lr 1e-4, log-std0 -3.5, apex weight 0.3, gamma .995, fail -3, 2.5M steps, 7 equally spaced CAL checkpoints.
# SAC config: lr 1e-5, BC 1, fixed alpha 0.001, critic-only warm-up 150k steps, log-std in [-5,-3], 1.2M steps, 7 CAL checkpoints.
cd "$(dirname "$0")/.."; export MALLOC_ARENA_MAX=2
C=results/spike07/best.json
P=".venv/bin/python experiments/spike07/rl7.py --classical $C --steps 2500000 --nck 7 --lr 1e-4 --c_apex 0.3 --nenv 128 --mb 8192 --logstd0 -3.5"
S=".venv/bin/python experiments/spike07/rl7.py --algo sac --variant hist --classical $C --steps 1200000 --nck 7 --c_apex 0.3 --alpha_fixed 0.001 --bc 1 --actor_delay 150000 --warm 50000 --buf 200000 --lr 1e-5 --logstd0 -3.5 --sac_ls_max -3.0"
H="flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c"
$H "for s in 0 1 2; do $P --variant hist --seeds \$s --out results/spike07/final_ppo_s\$s.pkl > results/spike07/final_ppo_s\$s.log 2>&1 & done; wait"; echo A > .scratch/finalA.done
$H "for s in 0 1 2; do $P --variant oracle --seeds \$s --out results/spike07/final_oracle_s\$s.pkl > results/spike07/final_oracle_s\$s.log 2>&1 & done; wait"; echo B > .scratch/finalB.done
$H "for s in 0 1 2; do $S --seeds \$s --out results/spike07/final_sac_s\$s.pkl > results/spike07/final_sac_s\$s.log 2>&1 & done; wait"; echo C > .scratch/finalC.done
