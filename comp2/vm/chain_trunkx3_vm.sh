#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: round-2 trunk student on the validation (x) track: pseudo-labels (rank-aware) from the mixed 5-net ensemble
# (id_v2x + e3_ps05 + bx1 + bx2 + tx_full) on the embedded games -> tx2_full -> acc_b (alone), acc_6m (5 + tx2), acc_3t (bx2 + tx_full + tx2)
cd $KQ
D=$KQ/idnet; N=tx2_full; T=$KQ/trunk/abt,$KQ/trunk/oof,$KQ/trunk/id
O=$D/id_v2x,$D/e3_ps05,$D/bx1_p9c64,$D/bx2_p9c64
if [ ! -f $D/pseudo_v5xt.npz ]; then $PY $SRC/idnet_pseudo.py $KQ/tm/moves_all.npz $O,$D/tx_full $D/pseudo_v5xt.npz --holdout-players $KQ/oof_games.csv --rank-prior --trunk $T > $D/pseudo_v5xt.log 2>&1; fi
for try in 1 2 3 4; do
  $PY $SRC/idnet.py $KQ/tm/moves_all.npz $D/$N --L 80 --epochs 16 --seed 2 --eval-every 4000 --board 64 --board-patch 9 --holdout-players $KQ/oof_games.csv --trunk $T --pseudo $D/pseudo_v5xt.npz --pseudo-thr 0.5 > $D/$N.log 2>&1
  if [ -f $D/$N/idnet_pred.npz ]; then break; fi; sleep 200
done
$PY $SRC/idens.py $KQ/tm/moves_all2.npz $D/$N $D/$N/acc_b.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.ens_b.log 2>&1
$PY $SRC/idens.py $KQ/tm/moves_all2.npz $O,$D/tx_full,$D/$N $D/$N/acc_6m.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.ens_6m.log 2>&1
$PY $SRC/idens.py $KQ/tm/moves_all2.npz $D/bx2_p9c64,$D/tx_full,$D/$N $D/$N/acc_3t.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.ens_3t.log 2>&1
echo done > $D/$N.DONE
