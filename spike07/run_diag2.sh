#!/bin/bash
# round 2 of the PPO diagnosis, after round 1 (D0 base, D2 cap bootstrap, D3 batch x4, D4 target-KL, D5 BC, D6 combo) all still decayed and the per-hit reward arithmetic (1 - 3*0.38 < 0) pointed at the reward.
cd "$(dirname "$0")/.."
C=results/spike07/best.json; R=".venv/bin/python spike07/rl7.py --variant hist --classical $C --seeds 100 --steps 2500000 --nck 4 --lr 1e-4 --logstd0 -3.5"
flock .scratch/heavy.lock systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 bash -c "
$R --c_apex 0.3 --out results/spike07/diag/D7_apex03.pkl > results/spike07/diag/D7_apex03.log 2>&1 &
$R --c_apex 0.3 --target_kl 0.01 --out results/spike07/diag/D8_apex03_kl.pkl > results/spike07/diag/D8_apex03_kl.log 2>&1 &
$R --c_apex 0.3 --nenv 128 --mb 8192 --out results/spike07/diag/D9_apex03_batch4x.pkl > results/spike07/diag/D9_apex03_batch4x.log 2>&1 &
$R --c_apex 0.0 --out results/spike07/diag/D10_apex0.pkl > results/spike07/diag/D10_apex0.log 2>&1 &
wait"
echo done > .scratch/diag2.done
