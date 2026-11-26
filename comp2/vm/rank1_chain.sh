#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
B=$KQ
T=$C1RUNS/tokfull_20261116
cd $B
DIRS=""; for d in $T/bigbank*; do if [ -f "$d/TOKENS.json" ]; then DIRS="$DIRS $d/results"; fi; done
echo "dirs:$DIRS" > rank1.log
while [ ! -f $B/tokpay/oof_own.rc ]; do sleep 60; done
$PY1 $SRC/tok_ranknet_vm.py rank1 tokres_rank_labels.npz $DIRS --epochs 8 --seed 0 >> rank1.log 2>&1
echo rank1 rc $? > rank1.rc