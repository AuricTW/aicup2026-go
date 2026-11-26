#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# comp2 11-19: opponent-side tokens of the 43,389 test / dev / OOF games re-extracted with comp1's FINE-TUNED HumanSL trunks.
# step 1 (this script): CPU cache of the network inputs (copy of comp1's tokcache_20261118_v1.py, frozen worker of the payload).
# SSD staging rule: outputs on $SCRATCH (deleted after the results are copied back).
PY=$PY1; S=$SCRATCH/ft
mkdir -p $S $KQ/ft
cd $KQ/ft
for b in "$@"; do
  [ -f $S/cache_$b/RECEIPT.json ] && continue
  rm -rf $S/cache_$b; echo "$b start $(date)" >> $S/cache.log
  $PY $COMP1_SCRIPTS/tokcache_20261118_v1.py $KQ/tokpay/$b $KQ/tokpay/$b/games.jsonl $S/cache_$b --workers 15 > $S/cache_$b.log 2>&1
  echo "$b end rc $? $(date)" >> $S/cache.log
done