#!/usr/bin/env python3
"""Exclusive single-GPU eval worker.

Parent MUST launch this with CUDA_VISIBLE_DEVICES already set to one physical
GPU in the process environment (before Python starts). Do not rely on setting
CVD after import — spawn/re-import of the server module would race that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict


def run(request_dict: Dict[str, Any], physical_gpu: int) -> Dict[str, Any]:
    # CVD should already be set by parent env; enforce again before torch.
    pinned = str(int(physical_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = pinned
    if os.path.exists("/opt/rocm") or "HIP_VISIBLE_DEVICES" in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = pinned

    cvd_now = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[EvalWorker] pid={os.getpid()} exclusive CVD={cvd_now} (expect physical={physical_gpu})",
        flush=True,
    )
    if cvd_now != pinned:
        return {
            "success": False,
            "error": f"CVD mismatch: got {cvd_now!r} expected {pinned!r}",
            "category": "pin_failure",
            "details": "CUDA_VISIBLE_DEVICES not exclusive",
            "traceback": "",
            "cuda_oom": False,
        }

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

    import torch  # noqa: E402 — after exclusive CVD

    n = torch.cuda.device_count()
    print(f"[EvalWorker] torch.cuda.device_count()={n} (must be 1)", flush=True)
    if n != 1:
        return {
            "success": False,
            "error": f"Expected 1 visible GPU after pin, got {n} (CVD={cvd_now})",
            "category": "pin_failure",
            "details": f"device_count={n}",
            "traceback": "",
            "cuda_oom": False,
        }

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
            device=0,
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
            "traceback": traceback.format_exc(),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="JSON request file")
    parser.add_argument("--output", required=True, help="JSON result file")
    parser.add_argument("--physical-gpu", required=True, type=int)
    args = parser.parse_args()

    with open(args.request, "r", encoding="utf-8") as f:
        request_dict = json.load(f)

    result = run(request_dict, args.physical_gpu)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
