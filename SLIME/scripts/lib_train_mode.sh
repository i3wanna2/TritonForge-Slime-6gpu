#!/bin/bash
# Shared resume / cold-start checkpoint wiring for TritonForge SLIME jobs.
#
# TRAIN_MODE:
#   resume (default) — load from SAVE if present, else fall back to SFT (first run)
#   cold             — always start from SFT at rollout 0; archives existing SAVE first
#
# Required env before sourcing:
#   MCORE_MODEL_PATH       SFT megatron checkpoint (ref)
#   MCORE_MODEL_PATH_SAVE  RL run directory (load+save for resume)
#   HF_MODEL_PATH          HF weights for tokenizer / sglang shape
#
# Do not `set -e` here — this file is sourced by other scripts.

TRAIN_MODE="${TRAIN_MODE:-resume}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

case "${TRAIN_MODE}" in
  resume|continue|RESUME|CONTINUE)
    TRAIN_MODE=resume
    ;;
  resume_weights|RESUME_WEIGHTS|reshard|RESHARD)
    # Load model weights from SAVE but skip optimizer (needed when TP/PP changes).
    TRAIN_MODE=resume_weights
    ;;
  cold|cold_start|COLD|COLD_START)
    TRAIN_MODE=cold
    ;;
  *)
    echo "Unknown TRAIN_MODE=${TRAIN_MODE}. Use: resume | resume_weights | cold" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

if [ -z "${MCORE_MODEL_PATH:-}" ] || [ -z "${MCORE_MODEL_PATH_SAVE:-}" ] || [ -z "${HF_MODEL_PATH:-}" ]; then
  echo "lib_train_mode.sh needs MCORE_MODEL_PATH, MCORE_MODEL_PATH_SAVE, HF_MODEL_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "${MCORE_MODEL_PATH_SAVE}"

archive_save_dir() {
  local src="$1"
  if [ ! -e "$src/latest_checkpointed_iteration.txt" ] && [ -z "$(ls -A "$src" 2>/dev/null || true)" ]; then
    return 0
  fi
  local bak="${src}.prev.$(date +%Y%m%d_%H%M%S)"
  echo "[train-mode] cold start: archiving existing SAVE -> ${bak}"
  mv "$src" "$bak"
  mkdir -p "$src"
}

CKPT_ARGS=()
case "${TRAIN_MODE}" in
  resume)
    # load==save: Megatron resumes iteration; empty SAVE -> slime falls back to ref_load @ 0
    CKPT_ARGS+=(
      --ref-load "${MCORE_MODEL_PATH}"
      --load "${MCORE_MODEL_PATH_SAVE}"
      --save "${MCORE_MODEL_PATH_SAVE}"
      --save-interval "${SAVE_INTERVAL}"
      --hf-checkpoint "${HF_MODEL_PATH}"
    )
    if [ -f "${MCORE_MODEL_PATH_SAVE}/latest_checkpointed_iteration.txt" ]; then
      echo "[train-mode] RESUME from ${MCORE_MODEL_PATH_SAVE} (iter=$(cat "${MCORE_MODEL_PATH_SAVE}/latest_checkpointed_iteration.txt"))"
    else
      echo "[train-mode] RESUME requested but SAVE empty -> first-run fallback to SFT ${MCORE_MODEL_PATH}"
    fi
    ;;
  resume_weights)
    # Keep RL/SFT weights from SAVE across TP change; re-init Adam (optimizer shard incompatible).
    if [ ! -f "${MCORE_MODEL_PATH_SAVE}/latest_checkpointed_iteration.txt" ]; then
      echo "[train-mode] resume_weights needs an existing SAVE ckpt at ${MCORE_MODEL_PATH_SAVE}" >&2
      return 1 2>/dev/null || exit 1
    fi
    _iter="$(cat "${MCORE_MODEL_PATH_SAVE}/latest_checkpointed_iteration.txt")"
    CKPT_ARGS+=(
      --ref-load "${MCORE_MODEL_PATH}"
      --load "${MCORE_MODEL_PATH_SAVE}"
      --save "${MCORE_MODEL_PATH_SAVE}"
      --save-interval "${SAVE_INTERVAL}"
      --hf-checkpoint "${HF_MODEL_PATH}"
      --no-load-optim
      --no-load-rng
    )
    echo "[train-mode] RESUME_WEIGHTS from ${MCORE_MODEL_PATH_SAVE} (iter=${_iter}); skip optim/rng (TP/PP may differ)"
    ;;
  cold)
    archive_save_dir "${MCORE_MODEL_PATH_SAVE}"
    CKPT_ARGS+=(
      --ref-load "${MCORE_MODEL_PATH}"
      --load "${MCORE_MODEL_PATH}"
      --save "${MCORE_MODEL_PATH_SAVE}"
      --save-interval "${SAVE_INTERVAL}"
      --hf-checkpoint "${HF_MODEL_PATH}"
      --no-load-optim
      --no-load-rng
      --start-rollout-id 0
    )
    echo "[train-mode] COLD start from SFT ${MCORE_MODEL_PATH}; writing RL ckpts to ${MCORE_MODEL_PATH_SAVE}"
    ;;
esac

export TRAIN_MODE
export CKPT_ARGS
