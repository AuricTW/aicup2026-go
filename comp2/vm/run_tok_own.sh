#!/bin/bash
source "$(dirname "$(readlink -f "$0")")/env.sh"      # KQ, SRC, PY, PY1, C1RUNS, SCRATCH, CUDA_LIB_DIR (vm/env.example.sh)
PY=$PY1
cd $KQ/tokpay/testdev_own && $PY worker.py > worker.log 2>&1; echo testdev_own rc $? > ../testdev_own.rc
cd $KQ/tokpay/oof_own && $PY worker.py > worker.log 2>&1; echo oof_own rc $? > ../oof_own.rc
