#!/usr/bin/env python3
"""Unit tests: train hold must block late/mid-wait acquire on ALL eval GPUs.

Run inside KBenchEval venv:
  CUDA_VISIBLE_DEVICES=2,3,4,5,7 EVAL_PHYSICAL_IDS=1 \\
  EVAL_RESERVED_DEVICES=7 EVAL_BORROWABLE_DEVICES=2,3,4,5 \\
  .venv/bin/python scripts/test_borrow_acquire_race.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3,4,5,7")
os.environ.setdefault("EVAL_PHYSICAL_IDS", "1")
os.environ.setdefault("EVAL_RESERVED_DEVICES", "7")
os.environ.setdefault("EVAL_BORROWABLE_DEVICES", "2,3,4,5")


def _load_server():
    path = Path(__file__).resolve().parent / "eval_server_subprocess.py"
    spec = importlib.util.spec_from_file_location("eval_server_subprocess", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


async def _reset(mod):
    async with mod._device_cv:
        mod.accepting_evals = True
        mod.borrow_enabled = True
        mod._device_busy.clear()
        mod._reserved_waiters.clear()
        mod._refresh_available()
        mod._device_cv.notify_all()


async def _close_evals(mod):
    """Mirror /borrow/disable gate flip (no drain wait)."""
    async with mod._device_cv:
        mod.accepting_evals = False
        mod.borrow_enabled = False
        mod._fail_reserved_waiters_locked()
        mod._device_cv.notify_all()


async def test_late_acquire_after_disable(mod) -> None:
    await _reset(mod)
    local_id = mod.BORROWABLE_DEVICES[0]
    barrier = asyncio.Event()
    result = {"status": None}

    async def late_request():
        assert mod.borrow_enabled is True
        barrier.set()
        await asyncio.sleep(0.05)
        try:
            await mod._acquire_borrowable_device(local_id, wait_s=1.0)
            result["status"] = "acquired"
            await mod._release_device(local_id)
        except mod.HTTPException as e:
            result["status"] = f"blocked:{e.status_code}"

    async def train_disable():
        await barrier.wait()
        await _close_evals(mod)
        async with mod._device_cv:
            inflight = sorted(mod._device_busy)
        assert inflight == [], f"drain should see empty busy, got {inflight}"

    await asyncio.gather(late_request(), train_disable())
    assert result["status"] == "blocked:503", result["status"]
    print("PASS: late acquire after disable → 503")


async def test_mid_wait_then_disable(mod) -> None:
    await _reset(mod)
    local_id = mod.BORROWABLE_DEVICES[0]

    await mod._acquire_borrowable_device(local_id, wait_s=1.0)
    started = asyncio.Event()
    result = {"status": None}

    async def waiter():
        started.set()
        try:
            await mod._acquire_borrowable_device(local_id, wait_s=5.0)
            result["status"] = "acquired"
            await mod._release_device(local_id)
        except mod.HTTPException as e:
            result["status"] = f"blocked:{e.status_code}"

    task = asyncio.create_task(waiter())
    await started.wait()
    await asyncio.sleep(0.05)
    await _close_evals(mod)
    await mod._release_device(local_id)
    await task
    assert result["status"] == "blocked:503", result["status"]
    print("PASS: mid-wait disable → 503 (not steal after release)")


async def test_try_acquire_after_disable(mod) -> None:
    await _reset(mod)
    got = await mod._try_acquire_borrowable(None)
    assert got is not None
    await mod._release_device(got)
    await _close_evals(mod)
    got2 = await mod._try_acquire_borrowable(None)
    assert got2 is None
    print("PASS: _try_acquire_borrowable returns None after disable")


async def test_reserved_acquire_blocked_on_suspend(mod) -> None:
    await _reset(mod)
    reserved = mod.RESERVED_DEVICES[0]
    await _close_evals(mod)
    try:
        await mod._acquire_reserved(priority=0, preferred_local=reserved)
        status = "acquired"
    except mod.HTTPException as e:
        status = f"blocked:{e.status_code}"
    assert status == "blocked:503", status
    print("PASS: reserved acquire blocked while suspended")


async def test_drain_sees_reserved_busy(mod) -> None:
    await _reset(mod)
    reserved = mod.RESERVED_DEVICES[0]
    await mod._acquire_reserved(priority=0, preferred_local=reserved)
    async with mod._device_cv:
        mod.accepting_evals = False
        mod.borrow_enabled = False
        old = sorted(set(mod.BORROWABLE_DEVICES) & mod._device_busy)
        all_busy = sorted(mod._device_busy)
    assert old == []
    assert all_busy == [reserved], all_busy
    await mod._release_device(reserved)
    print("PASS: full busy drain covers reserved GPU")


async def test_drain_returns_when_busy_clears(mod) -> None:
    """_drain_busy_until_empty returns only after busy empty (no force-clear)."""
    await _reset(mod)
    local_id = mod.BORROWABLE_DEVICES[0]
    await mod._occupy_device(local_id, wait_s=1.0, require_borrow=True)

    async def release_soon():
        await asyncio.sleep(0.2)
        await mod._release_device(local_id)

    task = asyncio.create_task(release_soon())
    result = await mod._drain_busy_until_empty(5.0, 1.0)
    await task
    assert result["drained"] is True
    async with mod._device_cv:
        assert not mod._device_busy
    print("PASS: drain waits until busy empty")


def main() -> int:
    mod = _load_server()
    assert hasattr(mod, "_occupy_device")
    assert hasattr(mod, "_drain_busy_until_empty")
    assert hasattr(mod, "EvalsSuspended")

    async def run_all():
        await test_late_acquire_after_disable(mod)
        await test_mid_wait_then_disable(mod)
        await test_try_acquire_after_disable(mod)
        await test_reserved_acquire_blocked_on_suspend(mod)
        await test_drain_sees_reserved_busy(mod)
        await test_drain_returns_when_busy_clears(mod)

    try:
        asyncio.run(run_all())
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
