#!/bin/bash

# Kernel Code Generation Agent Training — Qwen3-8B-SFT (6-GPU quality layout)
# Physical GPUs (do NOT touch 0,1). CVD ascending = role order:
#   actor  (Megatron TP=2 CP=2) : 2,3,4,5  (CVD slots 0-3)
#   rollout (SGLang)            : 6        (CVD slot 4)
#   eval reserved               : 7        (+ borrow 2-6 when actor offloaded)
#
# Uses sync train.py + --offload (not train_async) so generate/train do not overlap.

# Hard cleanup of OUR leftover train processes — kill children too, free ports, wait for GPU drain.
# Never touch unrelated jobs on GPU 0/1.
pkill -TERM -f "sglang::" 2>/dev/null || true
pkill -TERM -f "SLIME/train_async.py" 2>/dev/null || true
pkill -TERM -f "SLIME/train.py" 2>/dev/null || true
pkill -TERM -f "slime.rollout|RolloutRayActor|HttpServerEngineAdapter" 2>/dev/null || true
pkill -TERM -f "slime_plugins/rollout_buffer/buffer.py|python buffer.py" 2>/dev/null || true
pkill -TERM -f "eval_server_subprocess" 2>/dev/null || true
pkill -TERM -f "raysubmit_|ray::TrainRayActor|ray::RolloutRayActor|ray::Buffer" 2>/dev/null || true
sleep 2
pkill -KILL -f "sglang::" 2>/dev/null || true
pkill -KILL -f "SLIME/train_async.py" 2>/dev/null || true
pkill -KILL -f "SLIME/train.py" 2>/dev/null || true
pkill -KILL -f "slime.rollout|RolloutRayActor|HttpServerEngineAdapter" 2>/dev/null || true
pkill -KILL -f "slime_plugins/rollout_buffer/buffer.py|python buffer.py" 2>/dev/null || true
pkill -KILL -f "eval_server_subprocess" 2>/dev/null || true
pkill -KILL -f "raysubmit_|ray::TrainRayActor|ray::RolloutRayActor|ray::Buffer" 2>/dev/null || true
# Kill anything still holding our service ports (buffer/eval/ray dashboard)
python3 - <<'PY'
import os, signal, time
PORTS = (8889, 8265, 18188, 10000, 10001)
def kill_port_holders(sig):
    for port in PORTS:
        hexport = f"{port:04X}"
        inodes = set()
        try:
            for line in open("/proc/net/tcp"):
                parts = line.split()
                if len(parts) < 10:
                    continue
                if parts[1].endswith(":" + hexport) and parts[3] in ("0A", "01"):
                    inodes.add(parts[9])
        except FileNotFoundError:
            continue
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            fd = f"/proc/{pid}/fd"
            try:
                for e in os.listdir(fd):
                    try:
                        t = os.readlink(f"{fd}/{e}")
                    except OSError:
                        continue
                    if t.startswith("socket:[") and t[8:-1] in inodes:
                        os.kill(int(pid), sig)
            except OSError:
                pass
kill_port_holders(signal.SIGTERM)
time.sleep(1)
kill_port_holders(signal.SIGKILL)
PY
sleep 2
ray stop --force 2>/dev/null || true
pkill -KILL -f "ray::|raylet|gcs_server|dashboard.py|runtime_env_agent" 2>/dev/null || true
sleep 3
# Wait until our GPUs (2-7) are empty of leftover contexts (GPU0/1 may be busy elsewhere)
python3 - <<'PY'
import subprocess, time
def used(idx):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in out.strip().splitlines():
        i, m = line.split(",")
        if int(i.strip()) == idx:
            return int(float(m.strip()))
    return -1
deadline = time.time() + 60
while time.time() < deadline:
    bad = [i for i in range(2, 8) if used(i) > 500]
    if not bad:
        print("[cleanup] GPUs 2-7 memory clear")
        break
    print(f"[cleanup] waiting GPUs still using mem: {bad}")
    time.sleep(2)
else:
    print("[cleanup] WARNING: GPUs 2-7 not fully clear; continuing")
PY
rm -f "${TF_ROLLOUT_PAUSE_FILE:-/tmp/tf_rollout_pause}"

set -ex

export PYTHONBUFFERED=16
export WANDB_KEY=${WANDB_KEY:-"0db9fd073cc9e49c8bcec2b0a6929792ecb64e4e"}

# TP*PP*CP must equal actor GPUs: 2*1*2 = 4
export TP_SIZE=2
export PP_SIZE=1
export CP_SIZE=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data/liuxiaoyan/docker-tritonforge/TritonForge}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
export MCORE_MODEL_PATH="${MCORE_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B-Kernelbook-SFT-filtered}"
export PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/SLIME/data/kernel_bench/kernel_bench_triton_level_1_2.jsonl}"
export MCORE_MODEL_PATH_SAVE="${MCORE_MODEL_PATH_SAVE:-${PROJECT_ROOT}/models/Qwen3-8B-Kernelbook-SFT-filtered_save}"
# TP layout changed from 1→2: default resume weights without optimizer shards.
export TRAIN_MODE="${TRAIN_MODE:-resume_weights}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export EVAL_SERVER_URL="${EVAL_SERVER_URL:-http://127.0.0.1:18188}"
export TF_ROLLOUT_PAUSE_FILE="${TF_ROLLOUT_PAUSE_FILE:-/tmp/tf_rollout_pause}"
# shellcheck source=lib_train_mode.sh
source "${SCRIPT_DIR}/lib_train_mode.sh"

MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

# Restore quality-affecting hyperparams (align multi-turn NV recipe)
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type kernelbench
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --num-rollout 1000
   --rollout-batch-size 4
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-shuffle
   --n-samples-per-prompt 8
   --global-batch-size 32
   --balance-data
   --max-turns 3
   --gamma 0.4
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project TF-NV-singleturn-qwen3-8B-sft
   --wandb-group TF-Qwen3-8B-SFT-KBench-SingleTurn-6gpu
   --wandb-key ${WANDB_KEY}
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}
# CVD ascending: slots 0-3 actor, slot 4 rollout. GPU 7 reserved for eval (not in CVD).
export CUDA_VISIBLE_DEVICES=2,3,4,5,6
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# CuMemAllocator (--offload) is incompatible with expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF || true
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
# Workers: gen pool (ROLLOUT_NUM_PROCESS) + one eval worker per GPU (EVAL_WORKER_GPUS).
# Infer GPU 6 is never borrowed. OOM/large → reserved_queue → GPU 7 only.
# Train waits for in-flight /eval to return before /borrow/disable.
export ROLLOUT_NUM_PROCESS="${ROLLOUT_NUM_PROCESS:-20}"
export EVAL_WORKER_GPUS="${EVAL_WORKER_GPUS:-2,3,4,5,7}"
export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-5}"
export EVAL_RESERVED_DEVICES="${EVAL_RESERVED_DEVICES:-7}"
export EVAL_BORROWABLE_DEVICES="${EVAL_BORROWABLE_DEVICES:-2,3,4,5}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 5 --disable-usage-stats

sleep 5
echo "Checking Ray cluster status..."
ray status

echo "==== GPU layout (physical) ===="
echo "CVD=${CUDA_VISIBLE_DEVICES}  actor=2-5 (TP=${TP_SIZE} CP=${CP_SIZE})  rollout=6  eval=2,3,4,5,7"
echo "Scheme A: gen=${ROLLOUT_NUM_PROCESS} eval_workers=${EVAL_WORKER_GPUS} reserved=${EVAL_RESERVED_DEVICES} borrowable=${EVAL_BORROWABLE_DEVICES}"
echo "hyperparams: batch=4 gbs=32 max_resp=8192 max_tokens_per_gpu=4096 | sync train.py + --offload"
echo "===================="

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${PROJECT_ROOT}/SLIME:/root/Megatron-LM/\",
        \"CUDA_VISIBLE_DEVICES\": \"2,3,4,5,6\",
        \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_CUMEM_ENABLE\": \"0\",
        \"WANDB_MODE\": \"${WANDB_MODE:-offline}\",
        \"SGLANG_DISABLE_CUDA_GRAPH\": \"1\",
        \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:512\",
        \"TF_LOG_DIR\": \"${TF_LOG_DIR}\",
        \"TF_ROLLOUT_DATA_DIR\": \"${TF_ROLLOUT_DATA_DIR}\",
        \"TF_ROLLOUT_PAUSE_FILE\": \"${TF_ROLLOUT_PAUSE_FILE}\",
        \"EVAL_SERVER_URL\": \"${EVAL_SERVER_URL}\",
        \"EVAL_WORKER_GPUS\": \"${EVAL_WORKER_GPUS}\",
        \"EVAL_RESERVED_DEVICES\": \"${EVAL_RESERVED_DEVICES}\",
        \"EVAL_BORROWABLE_DEVICES\": \"${EVAL_BORROWABLE_DEVICES}\",
        \"PROJECT_ROOT\": \"${PROJECT_ROOT}\",
        \"http_proxy\": \"\",
        \"https_proxy\": \"\",
        \"HTTP_PROXY\": \"\",
        \"HTTPS_PROXY\": \"\",
        \"NO_PROXY\": \"localhost,127.0.0.1,::1,172.17.0.2,172.17.0.1,0.0.0.0\",
        \"no_proxy\": \"localhost,127.0.0.1,::1,172.17.0.2,172.17.0.1,0.0.0.0\"
     }
   }" \
   -- python3 SLIME/train.py \
   --num-epoch 1000 \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 1 \
   --rollout-num-gpus-per-engine 1 \
   --offload \
   --sglang-mem-fraction-static 0.7 \
   --sglang-disable-cuda-graph \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   --agent-rollout-buffer-url http://${MASTER_ADDR}:8889 \
   --rollout-num-process ${ROLLOUT_NUM_PROCESS} \
   --disable-rewards-normalization \
   --offload-old-actor \
   --offload-ref \
   --loss-mask-type qwen \
   --sglang-log-level error \
   --input-key prompt \
   --log-passrate \
   --rollout-task-type kernelbench
