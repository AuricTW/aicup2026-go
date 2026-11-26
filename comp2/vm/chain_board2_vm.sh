#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: second self-training round on the validation (x) track with the board net as an extra teacher.
# 1) pseudo-labels from old teacher + old student + board net (OOF players held out)   -> idnet/pseudo_v3x.npz
# 2) bx2_p9c64: same recipe as bx1_p9c64 but trained with the new pseudo-labels
# 3) evidence files: acc_b (bx2 alone), acc_2 (bx1 + bx2), acc_4 (old teacher + old student + bx1 + bx2)
cd $KQ
D=$KQ/idnet; M=$KQ/tm/moves_all2.npz
if [ ! -f $D/pseudo_v3x.npz ]; then
  $PY $SRC/idnet_pseudo.py $KQ/tm/moves_all.npz $D/id_v2x,$D/e3_ps05,$D/bx1_p9c64 $D/pseudo_v3x.npz --holdout-players $KQ/oof_games.csv > $D/pseudo_v3x.log 2>&1
fi
bash $KQ/run_idnet_vm.sh bx2_p9c64 --L 80 --epochs 16 --seed 2 --eval-every 8000 --pseudo-thr 0.5 --board 64 --board-patch 9 --holdout-players $KQ/oof_games.csv --pseudo $D/pseudo_v3x.npz
N=bx2_p9c64
$PY $SRC/idens.py $M $D/$N $D/$N/acc_b.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv > $D/$N.ens_b.log 2>&1
$PY $SRC/idens.py $M $D/bx1_p9c64,$D/$N $D/$N/acc_2.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv > $D/$N.ens_2.log 2>&1
$PY $SRC/idens.py $M $D/id_v2x,$D/e3_ps05,$D/bx1_p9c64,$D/$N $D/$N/acc_4.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv > $D/$N.ens_4.log 2>&1
echo done > $D/$N.ens_DONE
