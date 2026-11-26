# Settings of the GPU-host shell scripts in this folder. Copy to env.sh and edit; every script sources it.
HERE_VM="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
export SRC="$HERE_VM/../src"                       # comp2 Python code
export KQ=/data/comp2_gpu                          # working folder: tm/ (moves datasets), idnet/, trunk/, tokpay/, toklate/, ft/
export PY=/opt/venv_comp2/bin/python               # Python 3.12 + torch 2.10 (cu128): KataGo features, identity nets
export PY1=/opt/venv_comp1/bin/python              # Python 3.12 + torch 2.11 (cu128): comp1 token worker, heads, rank1
export C1RUNS=/data/comp1_weights/runs             # comp1 release weights (runs/... layout of comp1's weights MANIFEST)
export COMP1_SCRIPTS="$HERE_VM/../../comp1/scripts"
export SCRATCH=/ssd/comp2_scratch                  # fast local disk for the token caches of ft_*.sh
export CUDA_LIB_DIR=                               # optional: extra CUDA / driver library folder
if [ -n "$CUDA_LIB_DIR" ]; then export LD_LIBRARY_PATH="$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
