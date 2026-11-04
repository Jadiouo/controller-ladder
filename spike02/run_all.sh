#!/bin/bash
# sequential driver (one heavy process at a time). Run from repo root.
cd ./
LIM="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
for W in W0 W1 W2 W3 W4; do
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do echo "waiting for RAM"; sleep 60; done
  $LIM .venv/bin/python spike02/run_bc.py --w $W --seeds 10 >> results/spike02/run_bc_$W.log 2>&1
done
for W in W0 W1 W2 W3 W4; do
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 8 ]; do echo "waiting for RAM"; sleep 60; done
  $LIM .venv/bin/python spike02/run_rl.py --w $W --seeds 0,1,2,3,4 --steps 2000000 >> results/spike02/run_rl_$W.log 2>&1
done
echo ALLDONE > results/spike02/ALLDONE
