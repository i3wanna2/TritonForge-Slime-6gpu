#!/bin/bash
# CuMem + NCCL prewarm unit test on 2 free GPUs. Does NOT launch full SLIME train.
#
# Usage:
#   sudo docker exec tritonforge bash \
#     /data/liuxiaoyan/docker-tritonforge/TritonForge/SLIME/tests/run_nccl_prewarm_unit.sh
#
# Env:
#   UNIT_GPUS=6,7
#   PRESSURE_FREE_MIB=64     leave this much free after fake activation fill
#   NCCL_TEST_BIG_MIB=8      extra all_reduce payload (MiB)
#   FAKE_MODEL_MIB=1024      fake weight blob in CuMem model pool

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/liuxiaoyan/docker-tritonforge/TritonForge}"
UNIT_GPUS="${UNIT_GPUS:-6,7}"
PRESSURE_FREE_MIB="${PRESSURE_FREE_MIB:-8}"
NCCL_TEST_BIG_MIB="${NCCL_TEST_BIG_MIB:-8}"
FAKE_MODEL_MIB="${FAKE_MODEL_MIB:-2048}"
TEST_PY="${PROJECT_ROOT}/SLIME/tests/test_nccl_prewarm_under_pressure.py"

export CUDA_VISIBLE_DEVICES="${UNIT_GPUS}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export PRESSURE_FREE_MIB NCCL_TEST_BIG_MIB FAKE_MODEL_MIB
export PYTHONUNBUFFERED=1

echo "=== NCCL + CuMem prewarm unit test ==="
echo "GPUs=${UNIT_GPUS} PRESSURE_FREE_MIB=${PRESSURE_FREE_MIB} BIG=${NCCL_TEST_BIG_MIB}MiB MODEL=${FAKE_MODEL_MIB}MiB"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | head -20

run_mode() {
  local mode="$1"
  echo
  echo "---------- mode=${mode} ----------"
  NCCL_PREWARM_TEST_MODE="${mode}" torchrun --standalone --nproc_per_node=2 "${TEST_PY}"
}

# Separate process groups so "first use" is real for each mode.
run_mode no_prewarm
run_mode prewarm

echo
echo "=== done ==="
