#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: identity network on all train-split games.  usage: run_idnet_vm.sh NAME [extra args for idnet.py]
cd $KQ
NAME=$1; shift
mkdir -p $KQ/idnet
$PY $SRC/idnet.py $KQ/tm/moves_all.npz $KQ/idnet/$NAME "$@" > $KQ/idnet/$NAME.log 2>&1
