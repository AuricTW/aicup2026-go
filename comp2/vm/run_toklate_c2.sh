#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# comp2: late-game tokens (plies 181-300, frozen comp1 worker via a read-only copy of comp1's toklate_20261118_v1.py) for the
# OPPONENT side of the comp2 test / dev / OOF question games.  Waits until the 4090 has at most one job (GPU rule: max two).
PY=$PY1
mkdir -p $KQ/toklate; echo "waiting $(date)" > $KQ/toklate/run.log
while true; do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); if [ "$n" -le 1 ]; then break; fi; sleep 120; done
echo "start $(date)" >> $KQ/toklate/run.log
cd $C1RUNS
for b in testdev oof; do
  $PY $SRC/toklate_20261118_v1.py $KQ/tokpay/$b $KQ/tokpay/$b/games.jsonl $C1RUNS/fisher_policy_pilot_20261105_v2/assets/HumanSL_state_only.pt $KQ/toklate/$b --lo 181 --hi 300 --k 60 --workers 8 > $KQ/toklate/$b.log 2>&1
  echo "$b rc $? $(date)" >> $KQ/toklate/run.log
done
