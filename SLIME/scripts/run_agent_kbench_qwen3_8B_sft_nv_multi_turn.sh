#!/bin/bash
# tmux launcher for NV multi-turn (6-GPU + 蹭卡). Default TRAIN_MODE=cold (from SFT).
# GPUs: train 2-5 | infer 6 | eval reserved 7 | untouched 0,1
# Usage:
#   bash run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh
#   TRAIN_MODE=resume bash run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh
#   bash run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh cold

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/data/liuxiaoyan/docker-tritonforge/TritonForge}"
TF_LOG_DIR="${TF_LOG_DIR:-/data/liuxiaoyan/docker-tritonforge/logs}"
export TF_LOG_DIR
export TF_ROLLOUT_DATA_DIR="${TF_ROLLOUT_DATA_DIR:-${TF_LOG_DIR}/rollout_data}"
export TF_ROLLOUT_PAUSE_FILE="${TF_ROLLOUT_PAUSE_FILE:-/tmp/tf_rollout_pause}"
export EVAL_SERVER_URL="${EVAL_SERVER_URL:-http://127.0.0.1:18188}"
export PROJECT_ROOT

if [ "${1:-}" = "resume" ] || [ "${1:-}" = "cold" ] || [ "${1:-}" = "resume_weights" ]; then
  export TRAIN_MODE="$1"
fi
export TRAIN_MODE="${TRAIN_MODE:-cold}"

mkdir -p "${TF_LOG_DIR}/train" "${TF_LOG_DIR}/multi_turn" "${TF_ROLLOUT_DATA_DIR}"

SESSION_NAME="slime_qwen3_sft_multi_turn_run"
WINDOW_1="slime"
WINDOW_2="buffer"
WINDOW_3="eval_server"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
fi

sleep 2

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export NO_PROXY="localhost,127.0.0.1,::1,172.17.0.2,172.17.0.1,0.0.0.0"
export no_proxy="$NO_PROXY"
rm -f "${TF_ROLLOUT_PAUSE_FILE}"

tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "cd ${PROJECT_ROOT}" C-m
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export TRAIN_MODE=${TRAIN_MODE} TF_LOG_DIR=${TF_LOG_DIR} TF_ROLLOUT_DATA_DIR=${TF_ROLLOUT_DATA_DIR} TF_ROLLOUT_PAUSE_FILE=${TF_ROLLOUT_PAUSE_FILE} EVAL_SERVER_URL=${EVAL_SERVER_URL} PROJECT_ROOT=${PROJECT_ROOT} NO_PROXY='${NO_PROXY}' no_proxy='${no_proxy}'; bash ./SLIME/scripts/agent-example-kbench-qwen3-8B-sft-nv-multi-turn.sh |& tee ${TF_LOG_DIR}/train/slime_qwen3_sft_multi_turn_train.log" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30 && cd ${PROJECT_ROOT}/SLIME/slime_plugins/rollout_buffer && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export TF_LOG_DIR=${TF_LOG_DIR} TF_ROLLOUT_DATA_DIR=${TF_ROLLOUT_DATA_DIR} TF_ROLLOUT_PAUSE_FILE=${TF_ROLLOUT_PAUSE_FILE} PROJECT_ROOT=${PROJECT_ROOT} EVAL_WORKER_GPUS=${EVAL_WORKER_GPUS:-2,3,4,5,7} EVAL_RESERVED_DEVICES=${EVAL_RESERVED_DEVICES:-7} EVAL_BORROWABLE_DEVICES=${EVAL_BORROWABLE_DEVICES:-2,3,4,5} EVAL_LARGE_BYTES=${EVAL_LARGE_BYTES:-268435456} PYTHONPATH=${PROJECT_ROOT}/SLIME:/root/Megatron-LM:\${PYTHONPATH:-} NO_PROXY='${NO_PROXY}' no_proxy='${no_proxy}' && python buffer.py |& tee ${TF_LOG_DIR}/train/buffer_qwen3_sft_multi_turn.log" C-m

tmux new-window -t $SESSION_NAME -n $WINDOW_3
# Eval CVD excludes infer GPU 6. Reserved=7, borrowable=2-5. 蹭卡 during Megatron offload.
tmux send-keys -t ${SESSION_NAME}:${WINDOW_3} "sleep 20 && cd ${PROJECT_ROOT}/KBenchEval && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export NO_PROXY='${NO_PROXY}' no_proxy='${no_proxy}' EVAL_RESERVED_DEVICES=7 EVAL_BORROWABLE_DEVICES=2,3,4,5 EVAL_PHYSICAL_IDS=1 EVAL_LARGE_BYTES=${EVAL_LARGE_BYTES:-268435456} EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-600} EVAL_DRAIN_VRAM_MARGIN_MIB=${EVAL_DRAIN_VRAM_MARGIN_MIB:-512} EVAL_BORROW_POST_KILL_SETTLE_S=${EVAL_BORROW_POST_KILL_SETTLE_S:-2} EVAL_BORROW_KILL_WAIT_S=${EVAL_BORROW_KILL_WAIT_S:-15} && CUDA_VISIBLE_DEVICES=2,3,4,5,7 ${PROJECT_ROOT}/KBenchEval/.venv/bin/python scripts/eval_server_subprocess.py |& tee ${TF_LOG_DIR}/train/eval_server_qwen3_sft_multi_turn.log" C-m

echo "TRAIN_MODE=${TRAIN_MODE}"
echo "GPUs (physical): actor=2,3,4,5 | rollout=6 | eval=2,3,4,5,7 (蹭卡) | untouched=0,1"
echo "Train order: pause → borrow disable (immediate kill stale eval) → Megatron"
echo "Logs under: ${TF_LOG_DIR}/train/"
if [ "${ATTACH:-0}" = "1" ]; then
  tmux attach-session -t $SESSION_NAME
else
  echo "tmux session ${SESSION_NAME} running detached. Attach with: tmux attach -t ${SESSION_NAME}"
fi
