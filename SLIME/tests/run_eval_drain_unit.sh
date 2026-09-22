#!/bin/bash
# Eval drain immediate-kill unit test (no GPU required).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/liuxiaoyan/docker-tritonforge/TritonForge}"
VENV="${PROJECT_ROOT}/KBenchEval/.venv/bin/python"
TEST_PY="${PROJECT_ROOT}/KBenchEval/scripts/test_eval_drain_immediate.py"
RACE_PY="${PROJECT_ROOT}/KBenchEval/scripts/test_borrow_acquire_race.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,7}"
export EVAL_PHYSICAL_IDS=1
export EVAL_RESERVED_DEVICES=7
export EVAL_BORROWABLE_DEVICES=2,3,4,5

echo "=== eval drain immediate-kill unit test ==="
"${VENV}" "${TEST_PY}"

echo
echo "=== borrow acquire race (regression) ==="
"${VENV}" "${RACE_PY}"

echo
echo "=== done ==="
