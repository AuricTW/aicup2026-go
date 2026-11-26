#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# comp2 11-19 step 2: re-extract the opponent-side tokens with ONE fine-tuned HumanSL trunk of comp1 (GPU), then read them with
# comp1's frozen late heads (tok_apply_late.py).  usage: ft_run.sh <tag> <trunk_ft.pt>     outputs $S/out/{td,oof}_<head>_<tag>.npz
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$PY1; S=$SCRATCH/ft
T=$1; TR=$2; L=$S/run_$T.log; mkdir -p $S/out; cd $KQ/ft
echo "start $(date) trunk $TR" >> $L
for b in testdev oof; do
  while [ ! -f $S/cache_$b/RECEIPT.json ]; do sleep 20; done
  if [ ! -f $S/tok_${T}_$b/DONE ]; then
    rm -rf $S/tok_${T}_$b $S/tok_${T}_${b}_late; echo "extract $b start $(date)" >> $L
    $PY $SRC/tokextract_ft_20261118_v1.py $KQ/tokpay/$b $C1RUNS/fisher_policy_pilot_20261105_v2/assets/HumanSL_state_only.pt $TR $S/cache_$b $S/tok_${T}_$b --mem-frac 0.45 > $S/extract_${T}_$b.log 2>&1
    rc=$?; echo "extract $b end rc $rc $(date)" >> $L; [ $rc = 0 ] && touch $S/tok_${T}_$b/DONE
  fi
  o=td; [ $b = oof ] && o=oof
  for h in "t17late|$C1RUNS/tokfull_20261116/t17late/model.pt|" "s24|$C1RUNS/tokfull_20261116/full_late_s24_model.pt|" "xf10|$C1RUNS/tokfull_20261116/xf10_s23/model.pt|--ext"; do
    IFS='|' read hn hm hx <<< "$h"
    [ -f $S/out/${o}_${hn}_$T.npz ] && continue
    $PY $SRC/tok_apply_late.py $hm $KQ/ft/tok_classmap.npz $S/out/${o}_${hn}_$T.npz "$S/tok_${T}_$b|$KQ/tokpay/$b/games.jsonl|$KQ/ft/${b}_keys.json|$S/tok_${T}_${b}_late" --k 1024 $hx >> $S/apply_$T.log 2>&1
    echo "apply $hn $b rc $? $(date)" >> $L
  done
done
echo "end $(date)" >> $L