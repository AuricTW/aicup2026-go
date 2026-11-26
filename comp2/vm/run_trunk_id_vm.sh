#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
# VM: trunk embeddings (first 80 plies) for the identity gallery, round-robin list gamelist_id.jsonl; resumable (chunks), retried after out-of-memory
cd $KQ
export TRUNK_MAX_MOVES=80
for try in $(seq 1 200); do
  $PY $SRC/extract_trunk_id.py gamelist_id.jsonl trunk/id --pca trunk/pca64.npz --workers 12 --chunk 2000 --bs 512 >> trunk/id.log 2>&1
  if [ -f trunk/id/DONE.json ] || [ -f trunk/id_STOP ]; then break; fi
  sleep 120
done
