# TritonForge-Slime-6gpu

Fork of [RLsys-Foundation/TritonForge](https://github.com/RLsys-Foundation/TritonForge)  
(upstream SLIME based on [THUDM/slime](https://github.com/THUDM/slime)).

Algorithm hyper-parameters stay close to the upstream NV multi-turn recipe. This fork changes **how GPUs are owned over time**, not the RL objective.

---

## Overview (our thinking)

### Starting point

Upstream TritonForge assumes enough dedicated GPUs so that **train, rollout, and KernelBench eval can all stay resident**. On our machine we effectively needed an **~8-GPU** style footprint, but only had **6 GPUs we could use** (and we wanted to leave some cards free for other jobs). Shrinking TP/batch/length would change the recipe we cared about. So the question became: *can we keep the recipe and still run, by sharing cards smarter?*

### What we noticed

Watching `nvidia-smi` during a run, the expensive phase was not Megatron steps — it was **KernelBench eval**. Eval wall time is long, but **GPU utilization is often near zero**: compile happens on CPU, processes wait, one kernel runs on one device while sibling eval GPUs sit idle. Meanwhile, if train GPUs are reserved exclusively “for Megatron later,” they also sit idle during generate+eval.

So the real bottleneck was **eval throughput under a card budget**, not “training cannot start.” Card count was scarce; **duty cycle was wasteful**.

### Core idea: two kinds of eval GPU

We stopped treating “eval GPUs” as one pool. We split them by **ownership policy**:

| Kind | Role in the story | Rule |
|------|-------------------|------|
| **Reserved** | A small set of cards that **always belong to eval** | Eval can use them anytime. They never host Megatron. Guarantees progress even when train is awake. |
| **Borrowable** | Cards that **primarily belong to training** | Eval may use them **only while Megatron is offloaded** (generate / score phase). Before train wakes, borrow must stop and VRAM must return. |

Intuition:

- **Reserved** = baseline capacity so scoring never fully stalls.  
- **Borrowable** = temporary surplus capacity harvested from idle train GPUs when utilization would otherwise be ~0.  

Rollout (SGLang) stays on its own dedicated card(s): the policy must remain loaded for generation and should not fight eval or train for the same memory.

Together: *keep a floor of exclusive eval GPUs, and opportunistically multiplex train GPUs into eval when training does not need them.*

```text
                    ┌─────────────────────────┐
  generate + score  │  reserved  ✓ always     │
                    │  borrowable ✓ if asleep │
                    └─────────────────────────┘
                              │
                    clear borrow / restore VRAM
                              │
                    ┌─────────────────────────┐
  Megatron train    │  borrowable ✗ locked    │
                    │  reserved  ✓ still eval │
                    └─────────────────────────┘
```

### Consequence: borrow can OOM — reserved is the safety lane (with jump-the-queue)

Reserved is intentionally **small** (often one GPU). Borrowable cards are train cards after offload: free memory is good enough for many KernelBench jobs, but **not guaranteed** for large shapes or peak VRAM kernels. So borrow-first scoring will sometimes hit **CUDA OOM**.

We treat that as expected, not fatal:

1. **Default path** — try **borrowable** (or route obviously huge jobs straight to reserved).  
2. **On CUDA OOM on a borrowable GPU** — do not fail the sample; **requeue onto the reserved path** (`reserved_queue`).  
3. **Jump the queue (插队)** — OOM / large / `force_reserved` jobs get **higher priority** on the reserved worker so they are not stuck behind a long line of cheap borrowable-friendly jobs.  
4. **OOM on reserved itself** — stop retrying; fail that sample (no second safety pool).

So reserved is not only “always-on capacity,” it is also the **overflow lane** for work that borrowable could not hold. Without jump-the-queue, a single reserved GPU would become a FIFO bottleneck and OOM retries would wait forever behind normal traffic.

That is the whole design thesis. The sections below are just how we instantiated it on six cards and which failure modes we had to harden.

---

## Problem (constraints)

- Wanted upstream-like **8B multi-turn RL**, not a toy hyper-parameter shrink.  
- Only **6 usable GPUs** on a shared node.  
- Eval is slow and **under-utilizes** GPUs → opportunity to share.  
- Train and borrowable eval **must not overlap** in memory or NCCL.

---

## Concrete layout (example)

Physical IDs we used; adjust with env vars.

| Role | GPUs | Policy |
|------|------|--------|
| Left alone | `0,1` | Other jobs / system |
| Megatron actor | `2,3,4,5` | Train when awake; **offload** when generating → become **borrowable** |
| SGLang rollout | `6` | Dedicated inference (not borrowed) |
| Eval **reserved** | `7` | Always for KernelBench |
| Eval **borrowable** | `2,3,4,5` | Same devices as actor; only when offloaded |

---

## Control loop

Each step enforces the reserved / borrowable rule:

```text
generate (rollout)   → Megatron offloaded; eval uses reserved + may borrow
pause rollout
borrow disable       → no new evals on borrowable; kill in-flight borrow; VRAM ≈ baseline
wake + train         → exclusive Megatron on borrowable set; reserved still eval-only
borrow enable        → borrowable open again for next generate/score window
```

### Hardening (after the idea was clear)

| Failure mode | Mitigation |
|--------------|------------|
| Borrowable CUDA OOM / oversized shapes | Requeue to `reserved_queue` with **priority jump**; reserved OOM fails the sample |
| Eval child still holds CUDA after “finished” | Immediate kill on disable + drain to enable-time VRAM baseline |
| NCCL dies after CuMem wake | One-shot NCCL / grad-norm prewarm |
| Train starts while borrow still live | Rollout pause file + `borrow_enabled` gate in eval server |
| Cold start vs resume | `lib_train_mode.sh` (`cold` / `resume` / `resume_weights`) |

We did **not** change KernelBench rewards or GRPO math. This fork is infra for scarce GPUs.

---

## How to run

```bash
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh
bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_single_turn.sh
TRAIN_MODE=resume bash SLIME/scripts/run_agent_kbench_qwen3_8B_sft_nv_multi_turn.sh
```

Env: `PROJECT_ROOT`, `TF_LOG_DIR`, `TRAIN_MODE`, `EVAL_RESERVED_DEVICES`, `EVAL_BORROWABLE_DEVICES`.

Code: `SLIME/train.py`, `SLIME/slime/ray/ppo_actor.py`, `KBenchEval/scripts/eval_server_subprocess.py`.

Install and model download: [upstream TritonForge](https://github.com/RLsys-Foundation/TritonForge).

---

## Not in git

`models/`, logs, `rollout_data/`, wandb, `.venv` are ignored (weights are 100GB+).

---

## Lineage

```text
THUDM/slime
  └── RLsys-Foundation/TritonForge
        └── i3wanna2/TritonForge-Slime-6gpu
```

Apache-2.0. Please cite / star upstream if you use this.

---

## Monitoring

Watch **`rollout/raw_reward`** (and fixed KernelBench eval), not `train/loss` — GRPO advantages are centered near zero by design.
