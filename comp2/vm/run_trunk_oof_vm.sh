#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: trunk embeddings for the OOF-validation players' games (same PCA as trunk/abt)
cd $KQ
$PY $SRC/make_gamelist_keys.py oof_trunk_todo.csv gamelist_oof_trunk.jsonl O > trunk/oof.log 2>&1
$PY $SRC/extract_trunk.py gamelist_oof_trunk.jsonl trunk/oof --pca trunk/pca64.npz --workers 13 --chunk 500 >> trunk/oof.log 2>&1
