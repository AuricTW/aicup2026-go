#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: second self-training round on the TEST track (gallery = train + dev + lockbox accounts, 7,969 classes).
# Validated on the x track (bx2_p9c64, FINDINGS 21:40): 4-network ensemble OOF fused 0.7736 -> 0.7769.
# 1) pseudo-labels of the three lockbox-gallery networks on the opponent side of EVERY training game of this track
#    (split codes 0,1,3,4; the x track labelled all of its training games too)        -> idnet/pseudo_v4lb.npz
# 2) ba3lb_p9c64: board-patch student with these labels (recipe of bx2_p9c64)
# 3) evidence of the 4-network ensemble                                                -> idnet/ba3lb_p9c64/acc_4.npz
# Lockbox accounts stay gallery classes only (never scored, no validation or calibration question).
cd $KQ
D=$KQ/idnet; M=$KQ/tm/moves_all3.npz; N=ba3lb_p9c64
if [ ! -f $D/pseudo_v4lb.npz ]; then
  $PY $SRC/idnet_pseudo.py $M $D/id_v4lb,$D/a2lb_ps05,$D/ba2lb_p9c64 $D/pseudo_v4lb.npz --splits 0,1,3,4 > $D/pseudo_v4lb.log 2>&1
fi
for try in 1 2 3 4 5 6; do
  $PY $SRC/idnet.py $M $D/$N --L 80 --epochs 16 --seed 2 --eval-every 8000 --extra-split 1,3,4 --pseudo $D/pseudo_v4lb.npz --pseudo-thr 0.5 --board 64 --board-patch 9 > $D/$N.log 2>&1
  if [ -f $D/$N/idnet_pred.npz ]; then break; fi
  cp $D/$N.log $D/$N.fail$try.log; sleep 300
done
$PY $SRC/idens.py $M $D/id_v4lb,$D/a2lb_ps05,$D/ba2lb_p9c64,$D/$N $D/$N/acc_4.npz --temp 1.0 --acc --extra-split 1,3,4 > $D/$N.ens_4.log 2>&1
$PY $SRC/idens.py $M $D/$N $D/$N/acc_b.npz --temp 1.0 --acc --extra-split 1,3,4 > $D/$N.ens_b.log 2>&1
echo done > $D/lockbox2_DONE.txt
