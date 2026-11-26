#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: TEST track (7,969 accounts incl. lockbox as gallery classes only) with trunk-embedding identity nets, recipe validated on the x track
# (tx_full, tx2_full): ta1lb (round 1) -> rank-aware pseudo-labels of 4 moves-only nets + ta1lb -> ta2lb (round 2) -> 6-net evidence acc_6m.npz
cd $KQ
D=$KQ/idnet; M=$KQ/tm/moves_all3.npz; T=$KQ/trunk/abt,$KQ/trunk/oof,$KQ/trunk/id
O=$D/id_v4lb,$D/a2lb_ps05,$D/ba2lb_p9c64,$D/ba3lb_p9c64
run() { N=$1; shift; for try in 1 2 3 4; do $PY $SRC/idnet.py $M $D/$N --L 80 --epochs 16 --eval-every 4000 --board 64 --board-patch 9 --extra-split 1,3,4 --trunk $T "$@" > $D/$N.log 2>&1; if [ -f $D/$N/idnet_pred.npz ]; then return 0; fi; sleep 200; done; }
run ta1lb --seed 1
if [ ! -f $D/pseudo_v5lbt.npz ]; then $PY $SRC/idnet_pseudo.py $M $O,$D/ta1lb $D/pseudo_v5lbt.npz --splits 0,1,3,4 --rank-prior --trunk $T > $D/pseudo_v5lbt.log 2>&1; fi
run ta2lb --seed 2 --pseudo $D/pseudo_v5lbt.npz --pseudo-thr 0.5
$PY $SRC/idens.py $M $O,$D/ta1lb,$D/ta2lb $D/ta2lb/acc_6m.npz --temp 1.0 --acc --extra-split 1,3,4 --trunk $T > $D/ta2lb.ens_6m.log 2>&1
$PY $SRC/idens.py $M $D/ta2lb $D/ta2lb/acc_b.npz --temp 1.0 --acc --extra-split 1,3,4 --trunk $T > $D/ta2lb.ens_b.log 2>&1
echo done > $D/ta2lb.DONE
