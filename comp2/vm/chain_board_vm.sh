#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: board-patch identity networks (idnet.py --board 64 --board-patch 9) for the three tracks, student recipe
# (80 moves, 16 epochs, seed 1, pseudo-labels of the old teachers > 0.5), i.e. e3_ps05 / f1_ps05 / a1_ps05 with the board CNN.
cd $KQ
COMMON="--L 80 --epochs 16 --seed 1 --eval-every 8000 --pseudo-thr 0.5 --board 64 --board-patch 9"
bash $KQ/run_idnet_vm.sh bx1_p9c64 $COMMON --holdout-players $KQ/oof_games.csv --pseudo $KQ/idnet/pseudo_v2x.npz &
bash $KQ/run_idnet_vm.sh bf1_p9c64 $COMMON --pseudo $KQ/idnet/pseudo_v2full.npz &
bash $KQ/run_idnet2_vm.sh ba1_p9c64 $COMMON --extra-split 1,3 --pseudo $KQ/idnet/pseudo_v2full.npz &
wait
echo ALLDONE > $KQ/idnet/board_DONE.txt
