#!/usr/bin/env bash
# spike16a queue (second machine). Resumable: every stage skips finished outputs.
cd ~/robot_arm_control
PY=.venv-gpu/bin/python
for t in catch toss push_t; do $PY experiments/spike16a/s16.py cal $t || echo "FAIL cal $t"; done
for t in catch toss push_t; do $PY experiments/spike16a/s16.py gate $t || echo "FAIL gate $t"; $PY experiments/spike16a/s16.py real $t || echo "FAIL real $t"; done
echo QUEUE_DONE
