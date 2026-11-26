#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# Wait for the HumanSL pilot to finish, then run stage 2 (rest of dev, all test games, 10 games for remaining train players).
cd $KQ
while [ ! -f $KQ/hpilot/DONE.json ]; do
  if ! pgrep -f kata_human.py > /dev/null; then echo "pilot died" >> $KQ/hstage2.log; exit 1; fi
  sleep 20
done
sleep 5
PY=$PY
$PY $SRC/kata_human.py $KQ/gamelist.jsonl $KQ/hstage2 --select $KQ/select_h_stage2.json --profiles rank_10k,rank_3k,rank_2d,rank_7d >> $KQ/hstage2.log 2>&1
