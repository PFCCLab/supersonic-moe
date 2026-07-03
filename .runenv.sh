#!/bin/bash
# source .venv/bin/activate
# export PYTHONPATH="${SONIC_MOE_EXTRA_PYTHONPATH:-}:$PYTHONPATH"
export SONIC_MOE_FP8_MODE=perf
export USE_QUACK_GEMM=1
# CUDA 13 ptxas supports SM100 target architecture; triton-bundled ptxas does not.
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export NNODES=1
export PADDLE_TRAINERS_NUM=1
export CUDA_VISIBLE_DEVICES=0
