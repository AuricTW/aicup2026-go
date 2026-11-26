#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: identity evidence files with the board-patch nets, per track, as soon as each net is finished.
# Combinations: b = board net alone, 2 = old 30-epoch teacher + board net, 3 = old teacher + old student + board net.
cd $KQ
D=$KQ/idnet; M=$KQ/tm/moves_all2.npz
ens() {   # NAME TEACHER STUDENT [extra idens args]
  N=$1; T=$2; S=$3; shift; shift; shift
  while [ ! -f $D/$N/idnet_pred.npz ]; do sleep 20; done
  $PY $SRC/idens.py $M $D/$N $D/$N/acc_b.npz --temp 1.0 --acc "$@" > $D/$N.ens_b.log 2>&1
  $PY $SRC/idens.py $M $D/$T,$D/$N $D/$N/acc_2.npz --temp 1.0 --acc "$@" > $D/$N.ens_2.log 2>&1
  $PY $SRC/idens.py $M $D/$T,$D/$S,$D/$N $D/$N/acc_3.npz --temp 1.0 --acc "$@" > $D/$N.ens_3.log 2>&1
  echo done > $D/$N.ens_DONE
}
ens bx1_p9c64 id_v2x e3_ps05 --holdout-players $KQ/oof_games.csv
ens bf1_p9c64 id_v2full f1_ps05
ens ba1_p9c64 id_v3all a1_ps05 --extra-split 1,3
echo ALLDONE > $D/board_ens_DONE.txt
