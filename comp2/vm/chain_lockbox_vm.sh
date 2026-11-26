#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: FINAL test-track gallery = train-split + dev-split + LOCKBOX accounts (user decision 2026-11-15; lockbox accounts are classes
# of the opponent gallery only: never scored, no validation or calibration question is built from them).
# Dataset tm/moves_all3.npz (build_moves_lockbox.py: lockbox games appended with split code 4), classes via --extra-split 1,3,4.
# Three networks = the v3 recipe of the test track (id_v3all / a1_ps05 / ba1_p9c64) with the larger class set:
#   id_v4lb       plain net, 80 moves, 30 epochs
#   ba2lb_p9c64   board-patch student, 16 epochs, pseudo-labelled opponent sides of id_v2full (> 0.5)
#   a2lb_ps05     plain student, same
# then the evidence file of the three-net ensemble: idnet/ba2lb_p9c64/acc_3.npz
# The 4090 is shared with comp1 (about 11 GB there): at most TWO of these jobs at a time (teacher + one student), about 10 GB.
cd $KQ
D=$KQ/idnet; M=$KQ/tm/moves_all3.npz
run() {   # NAME args ; retried after a pause if the card was full at the start
  N=$1; shift
  for try in 1 2 3 4 5 6; do
    $PY $SRC/idnet.py $M $D/$N "$@" > $D/$N.log 2>&1
    if [ -f $D/$N/idnet_pred.npz ]; then return 0; fi
    cp $D/$N.log $D/$N.fail$try.log; sleep 300
  done
}
while pgrep -f [o]ppnet.py > /dev/null; do sleep 20; done
run id_v4lb --L 80 --epochs 30 --eval-every 8000 --extra-split 1,3,4 &
sleep 120
run ba2lb_p9c64 --L 80 --epochs 16 --seed 1 --eval-every 8000 --extra-split 1,3,4 --pseudo $D/pseudo_v2full.npz --pseudo-thr 0.5 --board 64 --board-patch 9
run a2lb_ps05 --L 80 --epochs 16 --seed 1 --eval-every 8000 --extra-split 1,3,4 --pseudo $D/pseudo_v2full.npz --pseudo-thr 0.5
wait
$PY $SRC/idens.py $M $D/id_v4lb,$D/a2lb_ps05,$D/ba2lb_p9c64 $D/ba2lb_p9c64/acc_3.npz --temp 1.0 --acc --extra-split 1,3,4 > $D/ba2lb_p9c64.ens_3.log 2>&1
echo done > $D/lockbox_DONE.txt
