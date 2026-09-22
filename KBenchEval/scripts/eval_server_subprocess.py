#!/usr/bin/env python3
"""
Subprocess-isolated evaluation server for KernelBench
Each evaluation runs in a separate process to prevent GPU context corruption
Supports both NVIDIA CUDA and AMD ROCm/HIP backends

Default timeout: 600 seconds (10 minutes) to account for:
- Process spawn overhead
- GPU context initialization per process
- Triton kernel compilation (no cross-process caching)
- Complex kernel evaluation
"""

import asyncio
import heapq
import subprocess
import json
import tempfile
import os
import signal
import sys
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Do NOT import torch at module level. spawn re-imports __main__; early CUDA init
# with multi-GPU CVD makes every child device=0 map to physical CVD[0] (GPU 2).

IS_AMD_GPU = bool("HIP_VISIBLE_DEVICES" in os.environ or os.path.exists("/opt/rocm"))
if IS_AMD_GPU:
    os.environ["ROCM_HOME"] = os.environ.get("ROCM_HOME", "/opt/rocm")
    os.environ["HIP_PLATFORM"] = "amd"
    os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")
    os.environ["HSA_ENABLE_COREDUMP"] = "0"
    os.environ["AMD_LOG_LEVEL"] = "0"
    os.environ["ROCM_DISABLE_CRASH_DUMP"] = "1"
    os.environ["HIP_ENABLE_COREDUMP"] = "0"

app = FastAPI(
    title="KernelBench Subprocess Isolation Server",
    description="Evaluation server with process isolation to prevent GPU context corruption (CUDA/ROCm)",
)


def _parse_id_list(env_name: str, default: list) -> list:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(default)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _cvd_physical_list() -> list:
    """Physical ids from CUDA_VISIBLE_DEVICES — no torch / CUDA context."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


_CVD_PHYSICAL = _cvd_physical_list()
_USE_PHYSICAL = os.environ.get("EVAL_PHYSICAL_IDS", "1").strip() not in ("0", "false", "False")
NUM_GPUS = len(_CVD_PHYSICAL)
gpu_type = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
print(f"[Server] CVD physical={_CVD_PHYSICAL} -> NUM_GPUS={NUM_GPUS} ({gpu_type})", flush=True)


def _to_local_ids(ids: list, label: str) -> list:
    """Map physical GPU ids → indices into CVD; pass-through if already local."""
    if not _USE_PHYSICAL:
        return [i for i in ids if 0 <= i < NUM_GPUS]
    local = []
    for gid in ids:
        if gid in _CVD_PHYSICAL:
            local.append(_CVD_PHYSICAL.index(gid))
        elif 0 <= gid < NUM_GPUS and gid not in _CVD_PHYSICAL:
            # already a local index
            local.append(gid)
        else:
            print(f"[Server] WARNING: {label} GPU {gid} not in CVD={_CVD_PHYSICAL}", flush=True)
    return local


RESERVED_DEVICES = _to_local_ids(
    _parse_id_list("EVAL_RESERVED_DEVICES", [0] if NUM_GPUS >= 1 else []),
    "reserved",
)
BORROWABLE_DEVICES = _to_local_ids(
    _parse_id_list(
        "EVAL_BORROWABLE_DEVICES",
        list(range(1, NUM_GPUS)) if NUM_GPUS > 1 else [],
    ),
    "borrowable",
)
# Drop invalid / duplicates
RESERVED_DEVICES = [i for i in RESERVED_DEVICES if 0 <= i < NUM_GPUS]
BORROWABLE_DEVICES = [i for i in BORROWABLE_DEVICES if 0 <= i < NUM_GPUS and i not in RESERVED_DEVICES]
LARGE_SHAPE_BYTES = int(os.environ.get("EVAL_LARGE_BYTES", str(256 * 1024 * 1024)))
EVAL_TIMEOUT_S = float(os.environ.get("EVAL_TIMEOUT_S", "600"))

# Exclusive occupancy (not a free-list token). Slot returns only after child is dead.
device_lock = asyncio.Lock()
_device_busy: set = set()
_device_cv = asyncio.Condition(device_lock)
# Two gates (set together by /borrow/enable|disable in sync train):
#   accepting_evals — any new occupy (borrow + reserved)
#   borrow_enabled  — extra gate for BORROWABLE (train) GPUs
borrow_enabled = False
accepting_evals = True
# Snapshot of borrowable VRAM at /borrow/enable (Megatron already offloaded).
# Disable must return to this baseline (+ margin); absolute soft gates are forbidden.
_borrow_vram_baseline: Dict[int, float] = {}
_priority_seq = 0
_reserved_waiters: list = []  # (-priority, seq, Future)
_inflight_procs_lock = threading.Lock()
_inflight_procs: Dict[int, subprocess.Popen] = {}
# /eval handlers currently running (including wait-for-GPU). Drain waits for this → 0.
_inflight_evals = 0
# Back-compat alias used by /health
available_devices = [i for i in range(NUM_GPUS) if i not in _device_busy]
gpu_semaphore = asyncio.Semaphore(max(1, NUM_GPUS)) if NUM_GPUS > 0 else None


class EvalsSuspended(Exception):
    """Raised into reserved waiters when train hold starts."""


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


def _kill_inflight_evals() -> list:
    """SIGKILL live eval children. Busy clears when their executor finally releases."""
    with _inflight_procs_lock:
        items = list(_inflight_procs.items())
    killed = []
    for local_id, proc in items:
        if proc.poll() is None:
            print(
                f"[Server] DRAIN kill eval local={local_id} phys={_phys(local_id)} pid={proc.pid}",
                flush=True,
            )
            _kill_proc_tree(proc)
            killed.append(_phys(local_id))
    return killed


def _kill_orphan_eval_worker_procs() -> list:
    """Sweep leftover eval_worker_entry PIDs not (or no longer) tracked in _inflight_procs.

    Child exit / killpg can leave stragglers; soft VRAM gates used to ignore them.
    """
    patterns = ("eval_worker_entry.py",)
    tracked = set()
    with _inflight_procs_lock:
        for proc in _inflight_procs.values():
            if proc.pid:
                tracked.add(proc.pid)
    killed = []
    for pid_s in os.listdir("/proc"):
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        if pid in tracked or pid == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if not any(p in cmd for p in patterns):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            print(f"[Server] DRAIN orphan SIGKILL eval_worker pid={pid}", flush=True)
        except ProcessLookupError:
            pass
        except PermissionError as e:
            print(f"[Server] DRAIN orphan kill failed pid={pid}: {e}", flush=True)
    return killed


def _phys(local_id: int) -> int:
    if 0 <= local_id < len(_CVD_PHYSICAL):
        return _CVD_PHYSICAL[local_id]
    return local_id


def _refresh_available():
    global available_devices
    available_devices = [i for i in range(NUM_GPUS) if i not in _device_busy]


print(
    f"[Server] CVD physical={_CVD_PHYSICAL} physical_ids={_USE_PHYSICAL} "
    f"reserved_local={RESERVED_DEVICES} (phys={[ _phys(i) for i in RESERVED_DEVICES ]}) "
    f"borrowable_local={BORROWABLE_DEVICES} (phys={[ _phys(i) for i in BORROWABLE_DEVICES ]}) "
    f"exclusive=1 timeout_s={EVAL_TIMEOUT_S} large_bytes={LARGE_SHAPE_BYTES}"
)


class EvalRequest(BaseModel):
    original_model_src: str
    custom_model_src: str
    seed_num: int = 42
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    verbose: bool = False
    measure_performance: bool = True
    preferred_device: Optional[int] = None
    backend: str = "cuda"
    # Scheduling hints (TritonForge borrow / large-shape priority)
    force_reserved: bool = False
    priority: int = 0
    estimated_bytes: Optional[int] = None


def _pin_physical_gpu(physical_gpu: int) -> str:
    """Set CVD to a single physical GPU. Must run BEFORE importing torch in this process."""
    pinned = str(int(physical_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = pinned
    if os.path.exists("/opt/rocm") or "HIP_VISIBLE_DEVICES" in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = pinned
    return pinned


def _pin_visible_device(device_id: int) -> str:
    """Pin by local CVD index (legacy helpers for health checks)."""
    if 0 <= device_id < len(_CVD_PHYSICAL):
        return _pin_physical_gpu(_CVD_PHYSICAL[device_id])
    return _pin_physical_gpu(device_id)


def _check_gpu_health(device_id: int) -> bool:
    """Check GPU health in isolated process (module-level for pickling)"""
    _pin_visible_device(device_id)
    import torch

    try:
        device = torch.device("cuda:0")
        torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def _get_gpu_info(device_id: int) -> Dict[str, Any]:
    """Get GPU info in isolated process (module-level for pickling)"""
    _pin_visible_device(device_id)
    import torch

    try:
        props = torch.cuda.get_device_properties(0)
        return {
            "device_id": device_id,
            "name": props.name,
            "available": True,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0),
            "compute_capability": (props.major, props.minor),
        }
    except Exception as e:
        return {
            "device_id": device_id,
            "error": str(e),
            "available": False,
        }


def run_isolated_evaluation(request_dict: Dict[str, Any], physical_gpu: int) -> Dict[str, Any]:
    """
    Run evaluation in an isolated spawn subprocess.
    CVD is pinned to one physical GPU BEFORE torch is imported → exclusive card.
    """
    import sys
    import traceback as _tb

    # CRITICAL: pin before any CUDA/torch import in this process.
    pinned = _pin_physical_gpu(physical_gpu)
    print(f"[EvalWorker] exclusive CVD={pinned} (physical={physical_gpu})", flush=True)

    if os.path.exists("/opt/rocm"):
        os.environ["ROCM_HOME"] = os.environ.get("ROCM_HOME", "/opt/rocm")
        os.environ["HIP_PLATFORM"] = "amd"
        os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")
        os.environ["HSA_ENABLE_COREDUMP"] = "0"
        os.environ["AMD_LOG_LEVEL"] = "0"
        os.environ["ROCM_DISABLE_CRASH_DUMP"] = "1"
        os.environ["HIP_ENABLE_COREDUMP"] = "0"
        os.environ["HIP_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_gpu_{physical_gpu}"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    import torch  # noqa: E402  — after CVD pin

    if os.path.exists("/opt/rocm"):
        from src.utils import set_gpu_arch

        set_gpu_arch(["MI300X", "gfx942"])

    try:
        from src.eval import eval_kernel_against_ref

        result = eval_kernel_against_ref(
            original_model_src=request_dict["original_model_src"],
            custom_model_src=request_dict["custom_model_src"],
            seed_num=request_dict["seed_num"],
            num_correct_trials=request_dict["num_correct_trials"],
            num_perf_trials=request_dict["num_perf_trials"],
            verbose=request_dict["verbose"],
            measure_performance=request_dict["measure_performance"],
            device=0,  # only visible device after exclusive CVD pin
            backend=request_dict["backend"],
        )

        if result is None:
            return {
                "success": False,
                "error": "Evaluation returned None (likely due to SyntaxError in code)",
                "category": "syntax_error",
                "details": "The kernel code contains syntax errors or failed to compile",
                "traceback": "",
                "cuda_oom": False,
            }

        return {
            "success": True,
            "result": result.dict() if hasattr(result, "dict") else result.__dict__,
            "cuda_oom": False,
        }

    except Exception as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
        error_info = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": _tb.format_exc(),
            "cuda_oom": is_oom,
        }
        error_str = str(e)
        if is_oom:
            error_info["category"] = "cuda_oom"
            error_info["details"] = "CUDA device memory exhausted"
        elif "out of resource: shared memory" in error_str:
            error_info["category"] = "shared_memory_exceeded"
            error_info["details"] = "Triton kernel requires more shared memory than available"
        elif "illegal memory access" in error_str:
            error_info["category"] = "illegal_memory_access"
            error_info["details"] = "Kernel accessed memory outside allocated bounds"
        elif "Unknown CUDA arch" in error_str:
            error_info["category"] = "unsupported_architecture"
            error_info["details"] = "GPU architecture not supported for this kernel"
        else:
            error_info["category"] = "unknown"
        return error_info


def _is_cuda_oom(result_dict: Dict[str, Any]) -> bool:
    """Only true CUDA device-OOM. Shared-memory / other errors are NOT OOM."""
    if not result_dict or result_dict.get("success"):
        return False
    if result_dict.get("cuda_oom") is True:
        return True
    return (result_dict.get("category") or "").lower() == "cuda_oom"


def _resolve_preferred_local(preferred_device: Optional[int]) -> Optional[int]:
    """Map request preferred_device (physical if EVAL_PHYSICAL_IDS) → local CVD index."""
    if preferred_device is None:
        return None
    pref = int(preferred_device)
    if _USE_PHYSICAL and pref in _CVD_PHYSICAL:
        return _CVD_PHYSICAL.index(pref)
    if 0 <= pref < NUM_GPUS:
        return pref
    return None


async def _occupy_device(local_id: int, wait_s: float, *, require_borrow: bool = False) -> None:
    """Occupy one GPU exclusively. Released only via _release_device after child exits.

    Gates (re-checked under the same lock — no check-then-acquire race):
      accepting_evals must be True
      if require_borrow: borrow_enabled must be True (train GPUs)
    """
    deadline = asyncio.get_event_loop().time() + wait_s
    async with _device_cv:
        while True:
            if not accepting_evals or (require_borrow and not borrow_enabled):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"evals suspended (train hold) phys={_phys(local_id)}"
                        if not accepting_evals
                        else f"borrow disabled for pinned train GPU phys={_phys(local_id)}"
                    ),
                )
            if local_id not in _device_busy:
                _device_busy.add(local_id)
                _refresh_available()
                print(
                    f"[Server] ACQUIRE exclusive GPU local={local_id} phys={_phys(local_id)} "
                    f"busy={sorted(_device_busy)}",
                    flush=True,
                )
                return
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise HTTPException(
                    status_code=503,
                    detail=f"Timed out waiting for exclusive GPU local={local_id} phys={_phys(local_id)}",
                )
            try:
                await asyncio.wait_for(_device_cv.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail=f"Timed out waiting for exclusive GPU local={local_id} phys={_phys(local_id)}",
                )


async def _acquire_device(local_id: int, wait_s: float) -> None:
    await _occupy_device(local_id, wait_s, require_borrow=False)


async def _acquire_borrowable_device(local_id: int, wait_s: float) -> None:
    if local_id not in BORROWABLE_DEVICES:
        await _occupy_device(local_id, wait_s, require_borrow=False)
        return
    await _occupy_device(local_id, wait_s, require_borrow=True)


async def _release_device(local_id: int) -> None:
    async with _device_cv:
        _device_busy.discard(local_id)
        _refresh_available()
        # Hand reserved GPU to next waiter only while evals are open.
        if local_id in RESERVED_DEVICES and _reserved_waiters and accepting_evals:
            while _reserved_waiters:
                _neg_p, _seq, fut = heapq.heappop(_reserved_waiters)
                if not fut.done():
                    _device_busy.add(local_id)
                    _refresh_available()
                    fut.set_result(local_id)
                    print(
                        f"[Server] Passed RESERVED GPU local={local_id} to priority waiter",
                        flush=True,
                    )
                    _device_cv.notify_all()
                    return
        print(
            f"[Server] RELEASE exclusive GPU local={local_id} phys={_phys(local_id)} busy={sorted(_device_busy)}",
            flush=True,
        )
        _device_cv.notify_all()


async def _eval_enter() -> None:
    """Mark an /eval request in-flight (before acquire)."""
    global _inflight_evals
    async with _device_cv:
        _inflight_evals += 1
        _device_cv.notify_all()


async def _eval_leave() -> None:
    """Mark /eval finished — kernel child returned and handler is exiting."""
    global _inflight_evals
    async with _device_cv:
        _inflight_evals = max(0, _inflight_evals - 1)
        _device_cv.notify_all()


async def _try_acquire_borrowable(preferred_local: Optional[int] = None) -> Optional[int]:
    async with _device_cv:
        if not accepting_evals or not borrow_enabled or not BORROWABLE_DEVICES:
            return None
        candidates = list(BORROWABLE_DEVICES)
        if preferred_local is not None and preferred_local in candidates:
            candidates = [preferred_local] + [c for c in candidates if c != preferred_local]
        for device_id in candidates:
            if device_id not in _device_busy:
                _device_busy.add(device_id)
                _refresh_available()
                print(
                    f"[Server] ACQUIRE BORROW local={device_id} phys={_phys(device_id)}",
                    flush=True,
                )
                return device_id
    return None


async def _acquire_reserved(priority: int = 0, preferred_local: Optional[int] = None) -> int:
    """Acquire a reserved GPU; higher priority jumps the queue (OOM fallback)."""
    global _priority_seq
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    wait_s = float(os.environ.get("EVAL_RESERVED_WAIT_S", "600"))

    async with _device_cv:
        if not accepting_evals:
            raise HTTPException(status_code=503, detail="evals suspended (train hold)")
        pool = [d for d in RESERVED_DEVICES if d not in _device_busy]
        if preferred_local is not None and preferred_local in pool:
            pool = [preferred_local]
        if pool and not _reserved_waiters:
            device_id = pool[0]
            _device_busy.add(device_id)
            _refresh_available()
            print(
                f"[Server] ACQUIRE RESERVED local={device_id} phys={_phys(device_id)} priority={priority}",
                flush=True,
            )
            return device_id
        _priority_seq += 1
        heapq.heappush(_reserved_waiters, (-int(priority), _priority_seq, fut))
        print(
            f"[Server] Queued for RESERVED priority={priority} waiters={len(_reserved_waiters)}",
            flush=True,
        )

    try:
        return await asyncio.wait_for(fut, timeout=wait_s)
    except EvalsSuspended:
        raise HTTPException(status_code=503, detail="evals suspended (train hold)")
    except asyncio.TimeoutError:
        async with _device_cv:
            _reserved_waiters[:] = [w for w in _reserved_waiters if w[2] is not fut]
            heapq.heapify(_reserved_waiters)
        raise HTTPException(
            status_code=503,
            detail=f"Timed out waiting for reserved GPU (priority={priority})",
        )


def _fail_reserved_waiters_locked() -> None:
    """Must hold _device_cv. Wake queued reserved waiters with EvalsSuspended."""
    while _reserved_waiters:
        _neg_p, _seq, fut = heapq.heappop(_reserved_waiters)
        if not fut.done():
            fut.set_exception(EvalsSuspended("evals suspended (train hold)"))


def _nvidia_mem_used_mib() -> Dict[int, int]:
    """physical GPU index -> memory.used MiB. Empty dict if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except Exception as e:
        print(f"[Server] nvidia-smi mem query failed: {e}", flush=True)
        return {}
    used = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                used[int(parts[0])] = int(float(parts[1]))
            except ValueError:
                continue
    return used


async def _wait_borrowable_vram_clear(timeout_s: float, margin_mib: float) -> Dict[str, Any]:
    """After busy slots empty, wait until borrowable VRAM returns to enable-time baseline.

    Child exit ≠ VRAM free. Compare against snapshot taken at /borrow/enable (Megatron offloaded).
    """
    if not BORROWABLE_DEVICES:
        return {"vram_clear": True, "skipped": True}
    phys = [_phys(i) for i in BORROWABLE_DEVICES]
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    last = None
    while True:
        used = _nvidia_mem_used_mib()
        snap = {p: used.get(p, -1) for p in phys}
        bad = {}
        for p in phys:
            m = snap.get(p, -1)
            base = float(_borrow_vram_baseline.get(p, 0.0))
            limit = base + margin_mib
            if m < 0 or m > limit:
                bad[p] = {"used": m, "baseline": base, "limit": limit}
        if snap != last:
            print(
                f"[Server] eval drain VRAM borrowable={snap} baseline={_borrow_vram_baseline} "
                f"margin_mib={margin_mib} bad={bad}",
                flush=True,
            )
            last = snap
        if not bad:
            return {
                "vram_clear": True,
                "borrowable_used_mib": snap,
                "baseline_mib": dict(_borrow_vram_baseline),
            }
        if loop.time() >= deadline:
            print(f"[Server] eval drain VRAM timeout still high={bad}", flush=True)
            return {
                "vram_clear": False,
                "borrowable_used_mib": snap,
                "baseline_mib": dict(_borrow_vram_baseline),
                "still_high": bad,
            }
        await asyncio.sleep(0.5)


async def _drain_busy_until_empty(
    settle_s: float,
    kill_interval_s: float,
    *,
    kill_immediately: bool = False,
) -> Dict[str, Any]:
    """Wait for all busy slots to clear; SIGKILL stuck eval children.

    When kill_immediately is True (train /borrow/disable), kill in-flight eval
    subprocesses first instead of waiting for kernels to finish naturally.
    settle_s is a short post-kill grace before periodic re-kill, not a long
    wait-before-kill window.

    Returns only when _device_busy is empty and _inflight_evals is 0.
    """
    loop = asyncio.get_event_loop()
    killed_phys: list = []
    killing = kill_immediately
    next_kill = loop.time()
    settle_deadline = loop.time() + max(0.0, settle_s)
    last_inflight = None

    if kill_immediately:
        killed_phys = _kill_inflight_evals()
        print(
            f"[Server] eval drain immediate kill phys={killed_phys}; "
            f"settle={settle_s}s re-kill every {kill_interval_s}s",
            flush=True,
        )

    while True:
        async with _device_cv:
            inflight_slots = sorted(_device_busy)
            n_handlers = _inflight_evals
            if not inflight_slots and n_handlers == 0:
                return {"drained": True, "killed": killed_phys}

        if inflight_slots != last_inflight or n_handlers:
            print(
                f"[Server] eval drain waiting on local={inflight_slots} "
                f"phys={[ _phys(i) for i in inflight_slots ]} handlers={n_handlers}",
                flush=True,
            )
            last_inflight = inflight_slots

        now = loop.time()
        if not killing and now >= settle_deadline:
            killing = True
            more = _kill_inflight_evals()
            if more:
                killed_phys = list(dict.fromkeys(killed_phys + more))
                print(
                    f"[Server] eval drain settle over; killed phys={more}; "
                    f"re-kill every {kill_interval_s}s until empty",
                    flush=True,
                )
            next_kill = now + kill_interval_s
        elif killing and now >= next_kill:
            more = _kill_inflight_evals()
            if more:
                killed_phys = list(dict.fromkeys(killed_phys + more))
                print(f"[Server] eval drain re-kill phys={more}", flush=True)
            next_kill = now + kill_interval_s

        await asyncio.sleep(0.5)


# Back-compat names
async def acquire_gpu_device(preferred_device: Optional[int] = None) -> int:
    local = _resolve_preferred_local(preferred_device)
    return await _acquire_reserved(priority=0, preferred_local=local)


async def release_gpu_device(device_id: int):
    await _release_device(device_id)


def _run_isolated_with_timeout(request_dict: Dict[str, Any], local_id: int) -> Dict[str, Any]:
    """Launch eval_worker_entry via subprocess with CVD set in the child env before Python starts.

    Uses sys.executable (the KBenchEval venv python that started this server).
    Avoids multiprocessing spawn, which re-imports __main__ and can init CUDA too early.
    Registers Popen so /borrow/disable can SIGKILL stuck children before train wakes.
    """
    physical = _phys(local_id)
    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_worker_entry.py")
    py = sys.executable

    req_fd, req_path = tempfile.mkstemp(prefix="eval_req_", suffix=".json")
    out_fd, out_path = tempfile.mkstemp(prefix="eval_out_", suffix=".json")
    os.close(req_fd)
    os.close(out_fd)
    proc: Optional[subprocess.Popen] = None
    try:
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(request_dict, f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical)
        if IS_AMD_GPU or "HIP_VISIBLE_DEVICES" in env:
            env["HIP_VISIBLE_DEVICES"] = str(physical)
        # Fresh Triton cache per physical GPU
        env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_gpu_{physical}"

        cmd = [
            py,
            entry,
            "--request",
            req_path,
            "--output",
            out_path,
            "--physical-gpu",
            str(physical),
        ]
        print(
            f"[Server] EXEC eval py={py} phys={physical} CVD={env['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        with _inflight_procs_lock:
            _inflight_procs[local_id] = proc
        try:
            stdout, _ = proc.communicate(timeout=EVAL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(
                f"[Server] EVAL TIMEOUT {EVAL_TIMEOUT_S}s — kill phys={physical}",
                flush=True,
            )
            _kill_proc_tree(proc)
            try:
                stdout, _ = proc.communicate(timeout=5)
            except Exception:
                stdout = ""
            if stdout:
                print(stdout[-2000:], flush=True)
            return {
                "success": False,
                "error": f"Evaluation timed out ({EVAL_TIMEOUT_S}s) on physical GPU {physical}",
                "category": "timeout",
                "details": "timeout",
                "traceback": "",
                "cuda_oom": False,
            }

        if stdout:
            for line in stdout.splitlines()[-40:]:
                print(line, flush=True)

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return {
                "success": False,
                "error": f"Eval subprocess produced no result (exit={proc.returncode})",
                "category": "worker_crash",
                "details": (stdout or "")[-1500:],
                "traceback": "",
                "cuda_oom": False,
            }
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if proc is not None:
            with _inflight_procs_lock:
                if _inflight_procs.get(local_id) is proc:
                    del _inflight_procs[local_id]
        for p in (req_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _format_eval_failure(device_id: int, result_dict: Dict[str, Any]) -> str:
    error_msg = f"Evaluation failed on GPU {device_id}: {result_dict.get('error')}"
    cat = result_dict.get("category")
    if cat == "cuda_oom":
        error_msg = f"CUDA OOM on GPU {device_id}. {result_dict.get('details')}"
    elif cat == "shared_memory_exceeded":
        error_msg = f"Triton kernel exceeded shared memory limit on GPU {device_id}. {result_dict.get('details')}"
    elif cat == "illegal_memory_access":
        error_msg = (
            f"CUDA illegal memory access on GPU {device_id}. "
            f"The evaluation was isolated and the GPU remains healthy. {result_dict.get('details')}"
        )
    elif cat == "unsupported_architecture":
        error_msg = f"Unsupported GPU architecture on device {device_id}. {result_dict.get('details')}"
    elif cat == "syntax_error":
        error_msg = f"Syntax error in kernel code: {result_dict.get('details')}"
    return error_msg


@app.post("/borrow/enable")
async def borrow_enable():
    """Generate phase: open evals and allow borrow on idle train GPUs."""
    global borrow_enabled, accepting_evals, _borrow_vram_baseline
    async with _device_cv:
        accepting_evals = True
        borrow_enabled = True
        _device_cv.notify_all()
    # Snapshot after Megatron offload — disable must return to this floor.
    used = _nvidia_mem_used_mib()
    _borrow_vram_baseline = {_phys(i): float(used.get(_phys(i), 0)) for i in BORROWABLE_DEVICES}
    print(
        f"[Server] evals OPEN + borrow ENABLED baseline_vram={_borrow_vram_baseline}",
        flush=True,
    )
    return {
        "borrow_enabled": True,
        "accepting_evals": True,
        "borrowable": [_phys(i) for i in BORROWABLE_DEVICES],
        "baseline_vram_mib": dict(_borrow_vram_baseline),
    }


@app.post("/borrow/disable")
async def borrow_disable():
    """Train phase: close ALL new evals, drain/kill busy, wait VRAM back to enable baseline."""
    global borrow_enabled, accepting_evals
    settle_s = float(os.environ.get("EVAL_BORROW_POST_KILL_SETTLE_S", "2"))
    kill_interval_s = float(os.environ.get("EVAL_BORROW_KILL_WAIT_S", "15"))
    vram_timeout_s = float(os.environ.get("EVAL_DRAIN_VRAM_TIMEOUT_S", "180"))
    # Must return near enable-time baseline; do NOT use a loose absolute MiB gate.
    margin_mib = float(os.environ.get("EVAL_DRAIN_VRAM_MARGIN_MIB", "512"))

    async with _device_cv:
        accepting_evals = False
        borrow_enabled = False
        _fail_reserved_waiters_locked()
        _device_cv.notify_all()

    print(
        f"[Server] evals CLOSED — immediate kill + drain "
        f"(settle={settle_s}s, kill_every={kill_interval_s}s)",
        flush=True,
    )
    result = await _drain_busy_until_empty(
        settle_s, kill_interval_s, kill_immediately=True
    )
    # Final sweep: registered children + any orphan eval_worker_entry.
    more = _kill_inflight_evals()
    orphans = _kill_orphan_eval_worker_procs()
    if orphans:
        await asyncio.sleep(1.0)
        orphans += _kill_orphan_eval_worker_procs()
    killed = list(dict.fromkeys((result.get("killed") or []) + more))
    vram = await _wait_borrowable_vram_clear(vram_timeout_s, margin_mib)
    print(
        f"[Server] eval drain complete killed={killed} vram={vram}",
        flush=True,
    )
    return {
        "borrow_enabled": False,
        "accepting_evals": False,
        "drained": True,
        "killed": killed,
        "vram_clear": vram.get("vram_clear"),
        "borrowable_used_mib": vram.get("borrowable_used_mib"),
        "baseline_mib": vram.get("baseline_mib"),
        "borrowable_free": [_phys(i) for i in BORROWABLE_DEVICES],
        "busy_local": [],
    }


@app.get("/borrow/status")
async def borrow_status():
    async with _device_cv:
        inflight = sorted(set(BORROWABLE_DEVICES) & _device_busy)
        return {
            "borrow_enabled": borrow_enabled,
            "accepting_evals": accepting_evals,
            "inflight_evals": _inflight_evals,
            "borrowable": [_phys(i) for i in BORROWABLE_DEVICES],
            "reserved": [_phys(i) for i in RESERVED_DEVICES],
            "busy_local": sorted(_device_busy),
            "borrowable_in_use": [_phys(i) for i in inflight],
            "reserved_waiters": len(_reserved_waiters),
            "large_bytes": LARGE_SHAPE_BYTES,
            "timeout_s": EVAL_TIMEOUT_S,
        }


@app.post("/eval")
async def evaluate_kernel(request: EvalRequest):
    """
    Exclusive one-GPU-per-eval (spawn + single-CVD child).
    Prefer worker-pinned preferred_device. CUDA OOM → HTTP 507 (client reserved_queue).
    """
    if NUM_GPUS <= 0:
        raise HTTPException(status_code=503, detail="No GPUs available on this system")

    await _eval_enter()
    try:
        return await _evaluate_kernel_impl(request)
    finally:
        await _eval_leave()


async def _evaluate_kernel_impl(request: EvalRequest):
    est = int(request.estimated_bytes or 0)
    large = est >= LARGE_SHAPE_BYTES
    # Large→reserved is client reserved_queue. Do not override a pinned preferred_device.
    prefer_reserved = bool(request.force_reserved)
    priority = int(request.priority or 0)
    if request.force_reserved:
        priority = max(priority, 1000)
    if large and request.preferred_device is None:
        prefer_reserved = True
        priority = max(priority, 100)

    preferred_local = _resolve_preferred_local(request.preferred_device)
    request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    for k in ("force_reserved", "priority", "estimated_bytes"):
        request_dict.pop(k, None)

    device_id = None
    result_dict = None

    # Worker-pinned path: occupy that exact GPU (borrow or reserved).
    if preferred_local is not None and not prefer_reserved:
        wait_s = float(os.environ.get("EVAL_RESERVED_WAIT_S", "600"))
        if preferred_local in BORROWABLE_DEVICES:
            # Locked re-check of borrow_enabled (no check-then-acquire race).
            await _acquire_borrowable_device(preferred_local, wait_s)
        else:
            await _acquire_device(preferred_local, wait_s)
        device_id = preferred_local
        try:
            print(
                f"[Server] Eval PINNED local={device_id} phys={_phys(device_id)} "
                f"backend={request.backend} est_bytes={est}",
                flush=True,
            )
            result_dict = await asyncio.get_event_loop().run_in_executor(
                None, _run_isolated_with_timeout, request_dict, device_id
            )
            if result_dict.get("success"):
                from src.eval import KernelExecResult

                return KernelExecResult(**result_dict["result"])
            if preferred_local in BORROWABLE_DEVICES and _is_cuda_oom(result_dict):
                # Client requeues to reserved_queue — no server-side cross-GPU hop.
                print(
                    f"[Server] PINNED CUDA OOM on phys={_phys(device_id)}; return 507 (no reserved hop)",
                    flush=True,
                )
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": _format_eval_failure(device_id, result_dict),
                        "category": "cuda_oom",
                        "cuda_oom": True,
                        "physical_gpu": _phys(device_id),
                    },
                )
            print(_format_eval_failure(device_id, result_dict), flush=True)
            if result_dict.get("category") == "timeout":
                raise HTTPException(status_code=504, detail=result_dict.get("error"))
            raise HTTPException(status_code=500, detail=_format_eval_failure(device_id, result_dict))
        finally:
            await _release_device(device_id)
            device_id = None

    # Unpinned borrow attempt (legacy / no preferred_device)
    if not prefer_reserved and preferred_local is None:
        device_id = await _try_acquire_borrowable(None)
        if device_id is not None:
            try:
                print(
                    f"[Server] Eval BORROW local={device_id} phys={_phys(device_id)} "
                    f"backend={request.backend} est_bytes={est}",
                    flush=True,
                )
                result_dict = await asyncio.get_event_loop().run_in_executor(
                    None, _run_isolated_with_timeout, request_dict, device_id
                )
                if result_dict.get("success"):
                    from src.eval import KernelExecResult

                    return KernelExecResult(**result_dict["result"])
                if _is_cuda_oom(result_dict):
                    print(
                        f"[Server] BORROW CUDA OOM on phys={_phys(device_id)}; return 507 (no reserved hop)",
                        flush=True,
                    )
                    raise HTTPException(
                        status_code=507,
                        detail={
                            "error": _format_eval_failure(device_id, result_dict),
                            "category": "cuda_oom",
                            "cuda_oom": True,
                            "physical_gpu": _phys(device_id),
                        },
                    )
                print(_format_eval_failure(device_id, result_dict), flush=True)
                raise HTTPException(status_code=500, detail=_format_eval_failure(device_id, result_dict))
            finally:
                await _release_device(device_id)
                device_id = None

    # Reserved (large / force_reserved only — OOM retry is client reserved_queue)
    if prefer_reserved or preferred_local is None:
        device_id = await _acquire_reserved(priority=priority, preferred_local=preferred_local)
        try:
            print(
                f"[Server] Eval RESERVED local={device_id} phys={_phys(device_id)} "
                f"backend={request.backend} est_bytes={est} priority={priority}",
                flush=True,
            )
            result_dict = await asyncio.get_event_loop().run_in_executor(
                None, _run_isolated_with_timeout, request_dict, device_id
            )
            if result_dict.get("success"):
                from src.eval import KernelExecResult

                return KernelExecResult(**result_dict["result"])
            print(_format_eval_failure(device_id, result_dict), flush=True)
            if result_dict.get("category") == "timeout":
                raise HTTPException(status_code=504, detail=result_dict.get("error"))
            if _is_cuda_oom(result_dict):
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": _format_eval_failure(device_id, result_dict),
                        "category": "cuda_oom",
                        "cuda_oom": True,
                        "physical_gpu": _phys(device_id),
                    },
                )
            raise HTTPException(status_code=500, detail=_format_eval_failure(device_id, result_dict))
        finally:
            await _release_device(device_id)

    raise HTTPException(status_code=503, detail="No GPU available for eval")


@app.post("/eval/wait_idle")
async def eval_wait_idle():
    """Block until every in-flight /eval has returned (kernel child finished + slot released).

    Does NOT flip accepting_evals — call this after pause, before /borrow/disable.
    """
    timeout_s = float(os.environ.get("EVAL_IDLE_WAIT_S", "600"))
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    last = None
    while True:
        async with _device_cv:
            inflight = _inflight_evals
            busy = sorted(_device_busy)
            if inflight == 0 and not busy:
                print("[Server] eval idle — all kernels returned", flush=True)
                return {"idle": True, "inflight": 0, "busy_local": []}
            snap = (inflight, tuple(busy))
            if snap != last:
                print(
                    f"[Server] eval wait_idle inflight={inflight} busy={busy} "
                    f"phys={[ _phys(i) for i in busy ]}",
                    flush=True,
                )
                last = snap
            remaining = deadline - loop.time()
            if remaining <= 0:
                return {
                    "idle": False,
                    "inflight": inflight,
                    "busy_local": busy,
                    "busy_phys": [_phys(i) for i in busy],
                }
            try:
                await asyncio.wait_for(_device_cv.wait(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                pass


@app.get("/")
async def root():
    """Root endpoint providing service information"""
    gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
    return {
        "service": "KernelBench Subprocess Isolation Server",
        "status": "running",
        "gpu_platform": gpu_platform,
        "cuda_available": NUM_GPUS > 0,
        "num_gpus": NUM_GPUS,
        "cvd_physical": list(_CVD_PHYSICAL),
        "backends": ["cuda", "triton"],
        "python": sys.executable,
    }


@app.get("/health")
async def health_check():
    """Health check — no CUDA init in the parent process."""
    busy_count = len(_device_busy)
    return {
        "status": "healthy",
        "gpu_platform": "ROCm/HIP" if IS_AMD_GPU else "CUDA",
        "cuda_available": NUM_GPUS > 0,
        "total_gpu_devices": NUM_GPUS,
        "available_gpu_devices": NUM_GPUS - busy_count,
        "busy_gpu_devices": busy_count,
        "available_device_ids": available_devices.copy(),
        "cvd_physical": list(_CVD_PHYSICAL),
        "busy_local": sorted(_device_busy),
        "isolation": "subprocess+env-CVD",
        "python": sys.executable,
    }


@app.post("/reset_gpu/{device_id}")
async def reset_gpu(device_id: int):
    """Reset a specific GPU (no-op with subprocess isolation)"""
    if device_id < 0 or device_id >= NUM_GPUS:
        raise HTTPException(status_code=400, detail=f"Invalid device ID: {device_id}")
    
    return {
        "status": "GPU reset not needed with subprocess isolation",
        "device_id": device_id,
        "message": "Each evaluation runs in a fresh process"
    }


@app.get("/gpu_status")
async def gpu_status():
    """GPU occupancy status without initializing CUDA in the parent."""
    async with _device_cv:
        devices = [
            {
                "device_id": i,
                "physical": _phys(i),
                "available": i not in _device_busy,
                "busy": i in _device_busy,
            }
            for i in range(NUM_GPUS)
        ]
        return {
            "total_devices": NUM_GPUS,
            "cvd_physical": list(_CVD_PHYSICAL),
            "devices": devices,
            "busy_local": sorted(_device_busy),
            "python": sys.executable,
        }


@app.post("/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint (no-op with subprocess isolation)"""
    return {
        "status": "cleanup not needed with subprocess isolation",
        "message": "Each evaluation runs in a fresh process which cleans up automatically"
    }


@app.get("/backend_info")
async def backend_info():
    """Get information about supported backends"""
    gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
    return {
        "supported_backends": ["cuda", "triton"],
        "default_backend": "triton" if IS_AMD_GPU else "cuda",
        "gpu_platform": gpu_platform,
        "backend_descriptions": {
            "cuda": f"Custom {gpu_platform} kernels compiled with PyTorch's C++ extension system",
            "triton": "Custom Triton kernels using OpenAI's Triton compiler"
        },
        "cuda_available": NUM_GPUS > 0,
        "triton_available": True,
        "process_isolation": True,
        "gpu_architecture": os.environ.get("PYTORCH_ROCM_ARCH") if IS_AMD_GPU else os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "cuda_dsa_enabled": not IS_AMD_GPU
    }


@app.post("/reset_devices")
async def reset_devices():
    """Reset device occupancy (for debugging). Fails pending reserved waiters."""
    async with _device_cv:
        global _reserved_waiters
        _device_busy.clear()
        _refresh_available()
        while _reserved_waiters:
            _neg_p, _seq, fut = heapq.heappop(_reserved_waiters)
            if not fut.done():
                fut.set_exception(RuntimeError("device pool reset"))
        _device_cv.notify_all()
        return {
            "status": "device occupancy reset",
            "busy": sorted(_device_busy),
            "available": list(available_devices),
        }


if __name__ == "__main__":
    # Check for GPUs
    if NUM_GPUS == 0:
        print("[Server] WARNING: No GPUs detected. Server will run but cannot evaluate kernels.")
    else:
        gpu_platform = "ROCm/HIP" if IS_AMD_GPU else "CUDA"
        gpu_arch = os.environ.get("PYTORCH_ROCM_ARCH") if IS_AMD_GPU else os.environ.get("TORCH_CUDA_ARCH_LIST", "auto")
        
        print(f"[Server] Starting subprocess isolation server")
        print(f"[Server] GPU Platform: {gpu_platform}")
        print(f"[Server] Number of GPUs: {NUM_GPUS}")
        print(f"[Server] GPU Architecture: {gpu_arch}")
        print(f"[Server] Each evaluation will run in an isolated process")
        
        if IS_AMD_GPU:
            print(f"[Server] AMD MI300X optimizations enabled")
            print(f"[Server] Core dumps disabled for stability")
        else:
            print(f"[Server] CUDA DSA will be enabled for all kernel compilations")
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=18188,
        log_level="info",
        access_log=True
    )