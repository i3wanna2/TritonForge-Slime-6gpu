#!/usr/bin/env python3
"""Unit test: NCCL lazy workspace vs CuMemAllocator sleep/wake (蹭卡 offload path).

NOT testing: "NCCL ran while eval was still 蹭卡".
Crash logs already ordered as:
  borrow DISABLE drained=True vram_clear=True -> wake_up -> train
  fail at get_grad_norm all_reduce -> CUDA calloc 10MiB

IS testing: with --offload, CuMem model-pool sleep/wake can force NCCL to
re-calloc workspace on first post-wake collective; under activation-peak
pressure that calloc fails unless the collective was prewarmed before sleep.

Usage::

  CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \\
    SLIME/tests/test_nccl_prewarm_under_pressure.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
from cumem_allocator import CuMemAllocator


def _mib(n: int) -> float:
    return n / (1024 * 1024)


def _free_mib() -> float:
    free, _ = torch.cuda.mem_get_info()
    return _mib(free)


def _fill_until_free(target_free_mib: float) -> list[torch.Tensor]:
    blobs: list[torch.Tensor] = []
    chunk_mib = 256
    while _free_mib() > target_free_mib + chunk_mib:
        try:
            blobs.append(
                torch.empty(chunk_mib * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
            )
        except torch.cuda.OutOfMemoryError:
            chunk_mib = max(4, chunk_mib // 2)
            torch.cuda.empty_cache()
            if chunk_mib < 4:
                break
    for fine in (32, 8, 4, 2, 1):
        while _free_mib() > target_free_mib + fine:
            try:
                blobs.append(
                    torch.empty(fine * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
    torch.cuda.synchronize()
    return blobs


def _all_reduce(tag: str, big_mib: int) -> tuple[bool, str]:
    try:
        t = torch.ones(1, device="cuda", dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if big_mib > 0:
            big = torch.ones(big_mib * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
            dist.all_reduce(big, op=dist.ReduceOp.SUM)
            del big
        torch.cuda.synchronize()
        return True, f"{tag}: ok value={t.item()} free={_free_mib():.1f}MiB"
    except Exception as e:
        return False, f"{tag}: FAIL {type(e).__name__}: {e}\n{traceback.format_exc()}"


def run(prewarm: bool) -> dict:
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    pressure_free_mib = float(os.environ.get("PRESSURE_FREE_MIB", "16"))
    big_mib = int(os.environ.get("NCCL_TEST_BIG_MIB", "8"))
    model_mib = int(os.environ.get("FAKE_MODEL_MIB", "2048"))

    # CPU barrier only — NCCL barrier would itself prewarm collectives.
    gloo = dist.new_group(backend="gloo")

    alloc = CuMemAllocator.get_instance()
    with alloc.use_memory_pool(tag="model"):
        model = torch.nn.Linear(4096, 4096, device="cuda")
        weight_blob = torch.empty(model_mib * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        weight_blob.fill_(1.0)

    if prewarm:
        ok, msg = _all_reduce("prewarm_before_sleep", big_mib)
        if rank == 0:
            print(msg, flush=True)
        if not ok:
            return {"ok": False, "detail": msg, "mode": "prewarm"}

    dist.barrier(group=gloo)
    alloc.sleep(offload_tags=("model",))
    torch.cuda.empty_cache()
    if rank == 0:
        print(f"after_sleep free={_free_mib():.0f}MiB", flush=True)

    dist.barrier(group=gloo)
    alloc.wake_up(("model",))
    torch.cuda.empty_cache()
    if rank == 0:
        print(f"after_wake free={_free_mib():.0f}MiB", flush=True)

    # Reserve the scalar payload BEFORE pressure fill so the under-pressure
    # all_reduce is not blocked by torch.ones OOM (we care about NCCL malloc).
    probe = torch.ones(1, device="cuda", dtype=torch.float32)

    blobs = _fill_until_free(pressure_free_mib)
    if rank == 0:
        print(
            f"[mode={'prewarm' if prewarm else 'no_prewarm'}] "
            f"after_pressure_fill free={_free_mib():.1f}MiB blobs={len(blobs)}",
            flush=True,
        )

    # First NCCL use after wake under pressure (gloo barrier only above).
    # Scalar-only under pressure — matches get_grad_norm; big payload was for prewarm.
    try:
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        ok, msg = True, f"under_pressure_after_wake: ok value={probe.item()} free={_free_mib():.1f}MiB"
    except Exception as e:
        ok, msg = False, f"under_pressure_after_wake: FAIL {type(e).__name__}: {e}\n{traceback.format_exc()}"
    if rank == 0:
        print(msg, flush=True)

    # Avoid CuMem teardown crashes: sleep pool again, leave tensors alone.
    try:
        alloc.sleep(offload_tags=("model",))
    except Exception:
        pass
    del blobs
    torch.cuda.empty_cache()
    # Keep model/weight_blob referenced until process exit.
    _keepalive = (model, weight_blob)  # noqa: F841
    return {
        "ok": ok,
        "detail": msg,
        "mode": "prewarm" if prewarm else "no_prewarm",
    }


def main() -> int:
    mode = os.environ.get("NCCL_PREWARM_TEST_MODE", "prewarm")
    # Init NCCL default group + allow gloo subgroup.
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    result = run(prewarm=(mode == "prewarm"))
    # Do not destroy_process_group — CuMem + NCCL teardown is crashy; just exit.
    if rank != 0:
        return 0

    if mode == "prewarm":
        if result["ok"]:
            print("ASSERT_OK: prewarm-before-sleep survived wake+pressure", flush=True)
            return 0
        print("ASSERT_FAIL: prewarm path still failed", flush=True)
        return 2

    if not result["ok"]:
        print(
            "REPRO_OK: no-prewarm failed after CuMem sleep/wake under pressure "
            "(matches train calloc crash class)",
            flush=True,
        )
        return 0
    print(
        "REPRO_SOFT: no-prewarm still passed — tighten PRESSURE_FREE_MIB or "
        "raise NCCL_TEST_BIG_MIB / FAKE_MODEL_MIB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    # Force exit without NCCL teardown to avoid SIGSEGV in cumem free path.
    code = main()
    os._exit(code)
