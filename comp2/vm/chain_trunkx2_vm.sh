#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: decisive test of the trunk input with ALL embedded games (up to 400 per account), validation (x) track, after the extraction ends.
# tx_full alone (acc_b), with the four moves-only nets (acc_5m), and the four moves-only nets on the same game set (acc_4ctl).
cd $KQ
D=$KQ/idnet; N=tx_full; T=$KQ/trunk/abt,$KQ/trunk/oof,$KQ/trunk/id
while [ ! -f trunk/id/DONE.json ]; do sleep 60; done
for try in 1 2 3 4; do
  $PY $SRC/idnet.py $KQ/tm/moves_all.npz $D/$N --L 80 --epochs 16 --seed 1 --eval-every 4000 --board 64 --board-patch 9 --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.log 2>&1
  if [ -f $D/$N/idnet_pred.npz ]; then break; fi; sleep 200
done
O=$D/id_v2x,$D/e3_ps05,$D/bx1_p9c64,$D/bx2_p9c64
$PY $SRC/idens.py $KQ/tm/moves_all2.npz $D/$N $D/$N/acc_b.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.ens_b.log 2>&1
$PY $SRC/idens.py $KQ/tm/moves_all2.npz $O,$D/$N $D/$N/acc_5m.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.ens_5m.log 2>&1
$PY $SRC/idens.py $KQ/tm/moves_all2.npz $O $D/$N/acc_4ctl.npz --temp 1.0 --acc --holdout-players $KQ/oof_games.csv --trunk $T > $D/$N.ens_4ctl.log 2>&1
echo done > $D/$N.DONE
