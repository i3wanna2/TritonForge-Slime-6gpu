# TritonForge-Slime-6gpu

Fork of [RLsys-Foundation/TritonForge](https://github.com/RLsys-Foundation/TritonForge)  
(upstream SLIME based on [THUDM/slime](https://github.com/THUDM/slime)).

This repository documents a **limited-GPU redesign** of the TritonForge train / rollout / eval loop. Algorithm hyper-parameters stay close to the upstream NV multi-turn recipe; the changes are about **how GPUs are shared over time**.

---

## Problem

Upstream TritonForge is written for a layout where **train, SGLang rollout, and KernelBench eval each own dedicated GPUs** for the whole run (roughly an 8-GPU-class setup).

We only had **6 usable GPUs** on a shared node (and preferred to leave some cards untouched for other jobs). A naive shrink of parallel degrees or batch sizes would hurt training quality. We wanted to keep the **8B + multi-turn** recipe as intact as possible and still finish training.

---

## Observation: the bottleneck was evaluation, not training

Profiling the original disaggregated loop:

1. **Generation (SGLang)** needs continuous GPU memory for the policy.
2. **Training (Megatron)** needs a large burst of GPUs for TP/CP, then can sleep if offload is enabled.
3. **Evaluation (KernelBench compile + run)** is the long pole. Each sample may compile and execute Triton kernels; wall time is high, but **instantaneous GPU utilization is often near zero** — most time is CPU-side compile, waiting on subprocesses, or a single kernel on one device while other eval GPUs sit idle.

So the hardware picture was: during eval-heavy phases, **several GPUs that could help score kernels were idle**, while the train GPUs (if reserved exclusively) were also idle waiting for the next batch. Card count was the constraint; **utilization was not**.

That suggested a design where train GPUs are **borrowed for eval only when Megatron has offloaded**, instead of permanently dedicating enough cards for peak concurrency.

---

## Design

### Goal

Run an upstream-like multi-turn RL job on **6 physical GPUs** by time-sharing train and eval, without rewriting the reward or GRPO math.

### Layout (example physical IDs)

| Role | GPUs | Behavior |
|------|------|----------|
| Left alone | `0,1` | Shared node / other users |
| Megatron actor | `2,3,4,5` | Full train when awake; **offload** during generate |
| SGLang rollout | `6` | Dedicated inference |
| Eval reserved | `7` | Always available for KernelBench |
| Eval borrowable | `2,3,4,5` | Same as actor; only while Megatron is asleep |

### Control loop

Each training step is ordered so borrow never overlaps Megatron compute:

```text
generate (rollout)     → Megatron offloaded; eval may borrow 2–5 + use 7
pause rollout
borrow disable         → stop new evals; kill in-flight borrow workers; wait VRAM ≈ baseline
wake Megatron + train  → exclusive use of 2–5
borrow enable          → open eval again for next generate
```

### Engineering pieces that made this reliable

| Issue seen in practice | Fix in this fork |
|------------------------|------------------|
| Eval children still hold CUDA memory after “done” | `/borrow/disable` **immediate kill** + drain until VRAM returns to the enable-time baseline |
| NCCL collectives fail after CuMem wake | **NCCL / grad-norm prewarm** once after first wake |
| Train starts while eval still occupies a train GPU | Rollout **pause file** + hard borrow gate in the eval server |
| Resume vs cold start from SFT | Shared `lib_train_mode.sh` (`cold` / `resume` / `resume_weights`) |

### What we deliberately did not change

- KernelBench reward definition and GRPO training objective  
- Claiming a new SOTA Pass@1 — this fork is an **infra** adaptation for scarce GPUs  

---

## How to run (this layout)

```bash
# Multi-turn RL (default: cold start from SFT)
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh

# Single-turn
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_single_turn.sh

# Resume RL save
TRAIN_MODE=resume bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh
```

Important env vars: `PROJECT_ROOT`, `TF_LOG_DIR`, `TRAIN_MODE`, `EVAL_BORROWABLE_DEVICES`, `EVAL_RESERVED_DEVICES`.

Core code paths:

- `SLIME/train.py` — pause / borrow / train orchestration  
- `SLIME/slime/ray/ppo_actor.py` — offload + NCCL prewarm  
- `KBenchEval/scripts/eval_server_subprocess.py` — borrow enable/disable and drain  

For install, Docker, and model download, follow the [upstream TritonForge README](https://github.com/RLsys-Foundation/TritonForge).

---

## Not in this git repo

`models/`, train logs, `rollout_data/`, wandb dumps, and `.venv` are gitignored (weights are 100GB+). Place HF / Megatron checkpoints under `models/` locally as upstream describes.

---

## Lineage

```text
THUDM/slime
  └── RLsys-Foundation/TritonForge
        └── i3wanna2/TritonForge-Slime-6gpu   (this fork)
```

Upstream license: Apache-2.0. Please cite / star upstream if you build on this work.

---

## Monitoring

Prefer **`rollout/raw_reward`** (and a fixed KernelBench eval) over `train/loss`. Under GRPO, batch advantages are centered near zero by design, so loss is a poor stopping signal.
