#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
PY=$PY1
cd $KQ/tokpay/testdev && $PY worker.py > worker.log 2>&1; echo testdev rc $? > ../testdev.rc
cd $KQ/tokpay/oof && $PY worker.py > worker.log 2>&1; echo oof rc $? > ../oof.rc
