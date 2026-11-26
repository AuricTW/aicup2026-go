#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# comp2 11-19 step 3: read the tuned-trunk EARLY tokens (plies 1-180) with comp1's eight frozen early heads (tok_apply.py).
# outputs $S/out/{td,oof}_<head>_<trunk>.npz     log $S/early.log
PY=$PY1; S=$SCRATCH/ft
cd $KQ/ft; echo "start $(date) $@" >> $S/early.log
for T in ${@:-b3 t6}; do for b in testdev oof; do
  o=td; [ $b = oof ] && o=oof
  for h in vm_t1 vm_t6 full_s1 full_big_s3 full6a_s12 vm_t4 vm_t10 vm_t2; do
    [ -f $S/out/${o}_${h}_$T.npz ] && continue
    $PY $SRC/tok_apply.py $C1RUNS/tokhead_20261116_v1/$h/model.pt tok_classmap.npz $S/out/${o}_${h}_$T.npz "$S/tok_${T}_$b|$KQ/tokpay/$b/games.jsonl|$KQ/ft/${b}_keys.json" --k 1024 >> $S/early_apply.log 2>&1
    echo "apply $h $T $b rc $? $(date)" >> $S/early.log
  done
done; done
echo "end $(date)" >> $S/early.log
