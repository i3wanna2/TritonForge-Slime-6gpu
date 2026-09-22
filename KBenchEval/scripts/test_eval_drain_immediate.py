#!/usr/bin/env python3
"""Unit tests: /borrow/disable must SIGKILL in-flight evals immediately (no long grace).

Run inside KBenchEval venv (no GPU required):
  .venv/bin/python scripts/test_eval_drain_immediate.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from pathlib import Path
from unittest.mock import patch

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


async def test_immediate_kill_before_settle(mod) -> None:
    """kill_immediately=True must call _kill_inflight_evals right away, not after settle_s."""
    await _reset(mod)
    calls: list[float] = []
    t0 = time.monotonic()

    def fake_kill():
        calls.append(time.monotonic() - t0)
        return [7]

    async with mod._device_cv:
        mod._device_busy.add(mod.BORROWABLE_DEVICES[0])
        mod._inflight_evals = 1

    async def release_after_kill():
        await asyncio.sleep(0.3)
        async with mod._device_cv:
            mod._device_busy.clear()
            mod._inflight_evals = 0
            mod._device_cv.notify_all()

    release_task = asyncio.create_task(release_after_kill())
    with patch.object(mod, "_kill_inflight_evals", side_effect=fake_kill):
        result = await asyncio.wait_for(
            mod._drain_busy_until_empty(60.0, 1.0, kill_immediately=True),
            timeout=5.0,
        )
    await release_task

    assert result["drained"] is True
    assert result["killed"] == [7]
    assert calls, "expected immediate kill"
    assert calls[0] < 0.5, f"kill too late: first_call={calls[0]:.2f}s"
    assert len(calls) >= 1
    print(f"PASS: immediate kill at t={calls[0]:.3f}s (settle would be 60s)")


async def test_no_immediate_kill_waits_settle(mod) -> None:
    """Without kill_immediately, first kill happens only after settle_s."""
    await _reset(mod)
    calls: list[float] = []
    t0 = time.monotonic()

    def fake_kill():
        calls.append(time.monotonic() - t0)
        return []

    async with mod._device_cv:
        mod._device_busy.add(mod.BORROWABLE_DEVICES[0])
        mod._inflight_evals = 1

    async def release_after_settle():
        await asyncio.sleep(0.35)
        async with mod._device_cv:
            mod._device_busy.clear()
            mod._inflight_evals = 0
            mod._device_cv.notify_all()

    release_task = asyncio.create_task(release_after_settle())
    with patch.object(mod, "_kill_inflight_evals", side_effect=fake_kill):
        result = await asyncio.wait_for(
            mod._drain_busy_until_empty(0.3, 1.0, kill_immediately=False),
            timeout=5.0,
        )
    await release_task

    assert result["drained"] is True
    assert not calls or calls[0] >= 0.25, f"kill too early without immediate: {calls}"
    print("PASS: non-immediate drain waits until settle")


async def _reset(mod):
    async with mod._device_cv:
        mod.accepting_evals = True
        mod.borrow_enabled = True
        mod._device_busy.clear()
        mod._inflight_evals = 0
        mod._reserved_waiters.clear()
        mod._refresh_available()
        mod._device_cv.notify_all()


async def test_drain_returns_when_busy_clears(mod) -> None:
    """Backward-compat: drain still waits for busy empty when slots release naturally."""
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
    async def run_all():
        await test_immediate_kill_before_settle(mod)
        await test_no_immediate_kill_waits_settle(mod)
        await test_drain_returns_when_busy_clears(mod)

    try:
        asyncio.run(run_all())
    except Exception as e:
        print(f"FAIL: {e}")
        raise
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
