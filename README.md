# TritonForge-Slime-6gpu

**Fork of [RLsys-Foundation/TritonForge](https://github.com/RLsys-Foundation/TritonForge)**  
(upstream SLIME is based on [THUDM/slime](https://github.com/THUDM/slime)).

This fork adapts TritonForge + bundled SLIME for **GPU-scarce / shared-cluster** training: keep algorithm hyperparams close to the original NV multi-turn recipe, but **time-share** train / rollout / eval GPUs via Megatron offload + eval **蹭卡 (borrow)**.

> Upstream docs, Docker setup, and SFT/RL recipes still apply. Start from the [upstream README](https://github.com/RLsys-Foundation/TritonForge) for environment install and model download. This README only covers the **limited-GPU** changes.

---

## Why this fork

Upstream TritonForge assumes enough dedicated GPUs for train + SGLang rollout + KernelBench eval at once.

When cards are limited (example layout below), you need:

| Problem | Fork change |
|---------|-------------|
| Not enough GPUs for train+eval concurrent | Megatron `--offload` during generate; eval **borrows** idle train GPUs |
| Stale eval CUDA procs block train wake | `/borrow/disable` **immediate kill** + VRAM drain to baseline |
| NCCL crash after CuMem wake | Actor **NCCL / grad-norm prewarm** before first collective |
| Eval / train race on shared GPUs | Rollout pause file + borrow enable/disable around train steps |
| Cold start vs resume | Shared `lib_train_mode.sh` (`cold` / `resume` / `resume_weights`) |

**Not in scope of this fork:** changing KernelBench reward design, GRPO math, or claiming better Pass@1 than upstream.

---

## Example 6-GPU layout (physical IDs)

Used by the NV multi-turn launcher in this repo:

| Role | GPUs | Notes |
|------|------|--------|
| Untouched | `0,1` | Leave for others / system |
| Megatron actor (train) | `2,3,4,5` | Offloaded while generating |
| SGLang rollout | `6` | Dedicated infer |
| Eval reserved | `7` | Always available for eval |
| Eval borrowable | `2,3,4,5` | Only when Megatron is asleep |

Launchers (override with env):

```bash
# Multi-turn RL (default TRAIN_MODE=cold from SFT)
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh

# Single-turn
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_single_turn.sh

# Resume RL checkpoint
TRAIN_MODE=resume bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh
```

Useful env vars:

| Var | Meaning |
|-----|---------|
| `PROJECT_ROOT` | Path to this repo |
| `TF_LOG_DIR` | Log root (default outside repo; **gitignored**) |
| `TRAIN_MODE` | `cold` \| `resume` \| `resume_weights` |
| `EVAL_BORROWABLE_DEVICES` / `EVAL_RESERVED_DEVICES` | Physical GPU ids for 蹭卡 |

Core train loop: `SLIME/train.py` (pause → borrow disable → train → borrow enable).  
Eval server: `KBenchEval/scripts/eval_server_subprocess.py`.

---

## What changed (high level)

### `SLIME/`

- `train.py` — sync train with borrow + rollout pause
- `slime/ray/ppo_actor.py` — CuMem offload + NCCL prewarm
- `slime/backends/sglang_utils/*` — router / HTTP engine fixes for this stack
- `slime/rollout/agent_rollout.py` — multi-epoch skip / fetch hardening
- Generators under `slime_plugins/rollout_buffer/generator/` — NV multi/single kernel rollout
- Scripts: `agent-example-*.sh`, `run_agent_*.sh`, `lib_train_mode.sh`
- Unit helpers: `SLIME/tests/run_*_unit.sh`

### `KBenchEval/`

- `scripts/eval_server_subprocess.py` — borrow enable/disable, immediate drain, VRAM baseline
- `scripts/eval_worker_entry.py` — worker entry for pinned GPUs
- Drain / race unit tests under `scripts/test_*.py`

---

## What is **not** in git

| Path | Reason |
|------|--------|
| `models/` | HF / Megatron weights & RL saves (100GB+) |
| `logs/`, `*.log` | Train / buffer / eval logs |
| `**/rollout_data/` | Per-step trajectory dumps |
| `wandb/`, `.venv/` | Local run / env |

Download models from HuggingFace as in upstream (e.g. `JinnP/Qwen3-8B-Kernelbook-SFT-*`), put under `models/`, then point scripts via `MCORE_MODEL_PATH` / `HF_MODEL_PATH`.

---

## Relation to upstream

```text
THUDM/slime
    └── RLsys-Foundation/TritonForge   (SFT + KernelBench RL, multi-turn)
            └── i3wanna2/TritonForge-Slime-6gpu   (this fork: limited-GPU infra)
```

Please cite / star upstream if you use this work. Upstream license: Apache-2.0.

---

## Monitoring tip

For RL, watch **`rollout/raw_reward`** (and a held-out KernelBench eval), not `train/loss` (GRPO advantages are near-zero by design).
