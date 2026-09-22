import copy
import json
import logging
import os
import random
import re
import time
import uuid
import warnings
from functools import partial
from multiprocessing import Process, Queue, Semaphore
from queue import Empty
from typing import Dict, List, Optional

import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from slime_plugins.rollout_buffer.generator.base_generator import BaseGenerator
from slime_plugins.rollout_buffer.generator.kernelbench_config import KERNELBENCH_REWARDS, KERNELBENCH_VALIDATION
from slime_plugins.rollout_buffer.generator.reward_utils.kernel_utils import (
    KernelEvalResult,
    KernelExecResult,
    extract_last_code,
    strip_thinking_tags,
)
from slime_plugins.rollout_buffer.generator.triton_ops import TRITON_CORE_OPS

TASK_TYPE = "kernelbench"
DEFAULT_REMOTE_EVAL_SERVER_URL = "http://localhost:18188"
# Scheme A: many gen workers + exactly one eval worker per GPU (no GPU token pool).
EVAL_CONCURRENCY = 5
EVAL_WORKER_GPUS = [
    int(x.strip())
    for x in os.environ.get("EVAL_WORKER_GPUS", "2,3,4,5,7").split(",")
    if x.strip()
]
EVAL_RESERVED_GPU = int(
    os.environ.get(
        "EVAL_RESERVED_DEVICES",
        os.environ.get("EVAL_RESERVED_GPU", "7"),
    )
    .split(",")[0]
    .strip()
    or "7"
)
# Default gen concurrency when --rollout-num-process is unset / small.
DEFAULT_GEN_NUM_PROCESS = int(os.environ.get("GEN_NUM_PROCESS", "20"))
SAMPLING_PARAMS = {
    "top_p": 1,
}

# Path to baseline timing data (prefer mounted TritonForge over stale /root symlink)
_TF_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/data/liuxiaoyan/docker-tritonforge/TritonForge",
)
BASELINE_TIMING_PATH = os.path.join(
    _TF_ROOT,
    "KBenchEval/results/timing/H100_together/baseline_time_torch_compile_inductor_default.json",
)
# When set, workers sleep instead of hammering SGLang/eval during Megatron train.
ROLLOUT_PAUSE_FILE = os.environ.get("TF_ROLLOUT_PAUSE_FILE", "/tmp/tf_rollout_pause")
LARGE_SHAPE_BYTES = int(os.environ.get("EVAL_LARGE_BYTES", str(256 * 1024 * 1024)))

logger = logging.getLogger(__name__)


def wait_if_rollout_paused(poll_s: float = 2.0):
    """Block while train holds GPUs so we do not busy-spin on dead SGLang."""
    warned = False
    while os.path.exists(ROLLOUT_PAUSE_FILE):
        if not warned:
            logger.info(f"Rollout paused ({ROLLOUT_PAUSE_FILE}); waiting for train to finish")
            warned = True
        time.sleep(poll_s)


def estimate_label_bytes(label: str) -> int:
    """Rough activation footprint from KernelBench label constants + get_inputs.

    Used only for scheduling (large → reserved-first). Not a hard OOM predictor.
    """
    if not label:
        return 0
    consts: Dict[str, int] = {}
    for m in re.finditer(
        r"^(?P<name>[A-Za-z_][\w]*)\s*=\s*(?P<val>\d+)\s*(?:#.*)?$",
        label,
        flags=re.M,
    ):
        consts[m.group("name")] = int(m.group("val"))

    def _resolve(tok: str) -> Optional[int]:
        tok = tok.strip()
        if tok.isdigit():
            return int(tok)
        return consts.get(tok)

    total = 0
    # torch.randn(a, b, c, ...)
    for m in re.finditer(r"torch\.randn\(\s*([^)]+)\)", label):
        dims = []
        ok = True
        for part in m.group(1).split(","):
            part = part.strip().split("=")[-1].strip()
            v = _resolve(part)
            if v is None:
                ok = False
                break
            dims.append(v)
        if ok and dims:
            numel = 1
            for d in dims:
                numel *= max(d, 1)
            total += numel * 4  # fp32 bytes

    # Heuristic: ref + custom + workspace
    if total > 0:
        total = int(total * 6)
    return total


def load_baseline_timings() -> Dict[str, Dict[str, Dict]]:
    """Load baseline timing data from JSON file.

    Returns:
        Dict with structure: {level: {problem_name: timing_stats}}
    """
    if not os.path.exists(BASELINE_TIMING_PATH):
        logger.warning(f"Baseline timing file not found at {BASELINE_TIMING_PATH}")
        return {}

    try:
        with open(BASELINE_TIMING_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading baseline timings: {e}")
        return {}


def get_baseline_runtime(level: int, problem_name: str, baseline_timings: Dict) -> Optional[float]:
    """Get baseline runtime for a specific problem.

    Args:
        level: Problem level (1-4)
        problem_name: Name of the problem (e.g., '1_Square_matrix_multiplication_.py')
        baseline_timings: Loaded baseline timing data

    Returns:
        Baseline runtime in milliseconds, or None if not found
    """
    level_key = f"level{level}"
    if level_key in baseline_timings and problem_name in baseline_timings[level_key]:
        return baseline_timings[level_key][problem_name].get("mean", None)
    return None


def is_valid_reward(reward: float) -> bool:
    """Check if a reward value is valid.

    Args:
        reward: The reward value to check

    Returns:
        bool: True if reward is valid (between 0 and max_reward), False otherwise
    """
    return KERNELBENCH_REWARDS["max_reward"] >= reward >= 0


def validate_submission(code: str) -> bool:
    """Pre-check if submission is valid before evaluation.

    Refined validation that allows torch._inductor utilities but ensures real Triton kernels.

    Args:
        code: The generated code to validate

    Returns:
        bool: True if code passes validation, False otherwise
    """
    if not code:
        return False

    # Check if validation is enabled
    if not KERNELBENCH_VALIDATION.get("require_triton_jit", True):
        return True

    # Must have @triton.jit decorator
    if "@triton.jit" not in code:
        logger.warning("Submission rejected: Missing @triton.jit decorator")
        return False

    # Extract all function definitions with @triton.jit decorator
    # Simplified pattern that's more robust
    triton_kernel_pattern = r"@triton\.jit\s*\n\s*def\s+(\w+)\s*\([^)]*\):"
    triton_kernels = list(re.finditer(triton_kernel_pattern, code, re.MULTILINE))
    
    if not triton_kernels:
        # Try a more lenient pattern (handles different formatting)
        triton_kernel_pattern = r"@triton\.jit.*?def\s+(\w+)"
        triton_kernels = list(re.finditer(triton_kernel_pattern, code, re.DOTALL))
    
    if not triton_kernels:
        logger.warning("Submission rejected: No Triton kernel functions found")
        return False

    # Check if we need to validate Triton operations
    if KERNELBENCH_VALIDATION.get("require_triton_ops", True):
        # For each Triton kernel, check it uses proper Triton operations
        for match in triton_kernels:
            kernel_name = match.group(1)
            # Find the full kernel body by looking from the match position to the next function/class
            match_start = match.start()
            match_end = match.end()
            
            # Find the end of the kernel function (next def, class, or end of file)
            next_def = code.find('\ndef ', match_end)
            next_class = code.find('\nclass ', match_end)
            next_decorator = code.find('\n@', match_end)
            
            # Find the minimum valid position
            ends = [pos for pos in [next_def, next_class, next_decorator] if pos > 0]
            kernel_end = min(ends) if ends else len(code)
            
            kernel_body = code[match_start:kernel_end]

            # Use comprehensive list of Triton operations
            has_triton_ops = any(op in kernel_body for op in TRITON_CORE_OPS)
            if not has_triton_ops:
                logger.warning(f"Submission rejected: Kernel '{kernel_name}' doesn't use Triton operations")
                return False

            # Only check for forbidden PyTorch operations if configured
            if not KERNELBENCH_VALIDATION.get("allow_torch_in_kernel", False):
                # Check for forbidden PyTorch operations inside kernel
                # Note: We allow torch operations outside kernels for setup/wrapper code
                forbidden_in_kernel = ["torch.", "nn.", ".cuda()", ".cpu()", "F.", "torch.ops", "aten.", ".backward()"]

                for pattern in forbidden_in_kernel:
                    if pattern in kernel_body:
                        logger.warning(
                            f"Submission rejected: Kernel '{kernel_name}' uses forbidden operation '{pattern}'"
                        )
                        return False

    # Additional check: Must have either a forward method or call function
    if "def forward(" not in code and "def call(" not in code:
        logger.warning("Submission rejected: No forward() or call() function found")
        return False

    logger.info("Submission passed pre-validation: Contains valid Triton kernel(s)")
    return True


def submit_kernel_eval_request(
    semaphore: Semaphore,
    eval_server_url: str,
    item: dict,
    backend: str = "triton",
    max_retry: int = 3,
    baseline_timings: Optional[Dict] = None,
    preferred_device: Optional[int] = None,
) -> KernelEvalResult:
    original_model_src = item["label"]
    messages = item["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"last message must be assistant, but got {messages[-1]['role']}")

    # Extract code - will automatically strip think tags
    custom_model_src = extract_last_code(messages[-1]["content"], strip_think_tags=True)
    if custom_model_src is None:
        return KernelEvalResult(
            eval_status="failed",
            eval_response="no custom model source found",
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            reward=0.0,
            exec_result=KernelExecResult(),
        )

    # Pre-validate submission before sending to evaluation
    if not validate_submission(custom_model_src):
        return KernelEvalResult(
            eval_status="rejected",
            eval_response="Submission failed validation: must contain @triton.jit kernel with Triton operations",
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            reward=0.0,
            exec_result=KernelExecResult(),
        )

    payload = {
        "original_model_src": original_model_src,
        "custom_model_src": custom_model_src,
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "measure_performance": True,
        "backend": backend,
        "verbose": False,
        "seed_num": 42,
    }
    if preferred_device is not None:
        payload["preferred_device"] = int(preferred_device)
    est = estimate_label_bytes(original_model_src)
    payload["estimated_bytes"] = est
    # Large / OOM→reserved routing is client-side (reserved_queue), not server hop.

    res = None
    with semaphore:
        for attempt in range(max_retry):
            wait_if_rollout_paused()
            try:
                response = requests.post(
                    f"{eval_server_url}/eval",
                    json=payload,
                    proxies={"http": None, "https": None},
                    timeout=600,
                )
                if response.status_code == 503:
                    logger.warning(f"Eval 503 (no GPU); retry {attempt + 1}/{max_retry}")
                    time.sleep(min(5 * (attempt + 1), 20))
                    continue
                if response.status_code == 507:
                    # CUDA OOM on pinned GPU — caller requeues to reserved_queue.
                    detail = {}
                    try:
                        body = response.json()
                        detail = body.get("detail", body) if isinstance(body, dict) else {}
                    except Exception:
                        detail = {"error": response.text[:500]}
                    if not isinstance(detail, dict):
                        detail = {"error": str(detail)}
                    return KernelEvalResult(
                        eval_status="cuda_oom",
                        eval_response=str(detail.get("error") or detail),
                        completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        reward=0.0,
                        exec_result=KernelExecResult(),
                    )
                if response.status_code == 200:
                    exec_result = KernelExecResult.model_validate(response.json())
                    reward, response_msg = 0.0, "Current implementation con't pass compile check"
                    if exec_result.compiled:
                        reward = KERNELBENCH_REWARDS["compilation"]
                        if exec_result.correctness:
                            reward = KERNELBENCH_REWARDS["correctness"]
                            response_msg = "Current implementation passes correctness check"

                            if exec_result.runtime > 0 and baseline_timings:
                                extra_info = item.get("extra_info", {})
                                level = extra_info.get("level", None)
                                problem_name = extra_info.get("problem_name", "")
                                problem_id = extra_info.get("problem_id", None)

                                if problem_id is not None and problem_name:
                                    baseline_key = f"{problem_id}_{problem_name}.py"
                                elif problem_name:
                                    baseline_key = problem_name + ".py"
                                else:
                                    baseline_key = None

                                if baseline_key:
                                    baseline_runtime = get_baseline_runtime(level, baseline_key, baseline_timings)
                                    if baseline_runtime and baseline_runtime > 0:
                                        speedup = baseline_runtime / exec_result.runtime
                                        performance_reward = min(max(speedup - 1.0, 0.0), 2.0)
                                        reward += performance_reward
                                        response_msg += (
                                            f" (Speedup: {speedup:.2f}x, Runtime: {exec_result.runtime:.3f}ms "
                                            f"vs Baseline: {baseline_runtime:.3f}ms)"
                                        )
                                        logger.info(
                                            f"Performance reward: {performance_reward:.3f} for speedup {speedup:.2f}x"
                                        )
                                    else:
                                        logger.warning(
                                            f"No baseline runtime found for level {level}, problem {baseline_key}"
                                        )
                                else:
                                    logger.warning(f"Could not construct baseline key from extra_info: {extra_info}")
                        else:
                            response_msg = "Current implementation passes compile check but fails correctness check"
                    res = KernelEvalResult(
                        eval_status="completed",
                        eval_response=response_msg,
                        completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        reward=reward,
                        exec_result=exec_result,
                    )
                    break
                logger.error(f"Eval HTTP {response.status_code}: {response.text[:500]}")
            except Exception as e:
                logger.error(f"Error submitting kernel eval request: {e}")
                if attempt + 1 >= max_retry:
                    return KernelEvalResult(
                        eval_status="failed",
                        eval_response=str(e),
                        completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        reward=0.0,
                        exec_result=KernelExecResult(),
                    )
                time.sleep(2)
    if res is None:
        return KernelEvalResult(
            eval_status="failed",
            eval_response="eval failed after retries",
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            reward=0.0,
            exec_result=KernelExecResult(),
        )
    return res


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def query_llm_with_retry(
    client: OpenAI,
    messages: List[dict],
    sampling_params: dict,
    tools: Optional[List[dict]] = None,
) -> str:
    response = client.chat.completions.create(
        model="custom",
        messages=messages,
        stream=False,
        seed=random.randint(1, 10000000),
        tools=tools,
        **sampling_params,
    )
    print(f"{response.choices[0]=}")
    return response.choices[0].message.content


def rollout_one_trajectory(
    item: dict,
    client: OpenAI,
    sampling_params: dict,
    remote_eval_server_url: str,
    eval_semaphore: Semaphore,
    backend: str = "triton",
    max_retry: int = 3,
) -> List[dict]:
    messages = item["prompt"]
    assistant_message = None

    for _ in range(max_retry):
        try:
            assistant_message_content = query_llm_with_retry(client, messages, sampling_params, tools=None)
            assistant_message = {
                "role": "assistant",
                "content": assistant_message_content,
            }
            break  # Success, exit retry loop
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            continue

    if assistant_message is None:
        # All retries failed, create a default error message
        assistant_message = {
            "role": "assistant",
            "content": "Error: Failed to generate response after multiple retries.",
        }

    messages.append(assistant_message)
    return messages


def _build_output_item(
    item: dict,
    messages: List[dict],
    eval_result: KernelEvalResult,
    sampling_params: dict,
    baseline_timings,
    preferred_device: Optional[int],
) -> dict:
    reward = eval_result.reward if hasattr(eval_result, "reward") else 0.0
    item["rollout_index"] = item.get("rollout_index", 1)
    item["reward"] = reward

    execution_details = {}
    if hasattr(eval_result, "exec_result") and eval_result.exec_result:
        exec_result = eval_result.exec_result
        execution_details["compiled"] = exec_result.compiled
        execution_details["correctness"] = exec_result.correctness
        execution_details["runtime"] = exec_result.runtime
        execution_details["runtime_stats"] = exec_result.runtime_stats

        if exec_result.runtime > 0 and baseline_timings:
            extra_info = item.get("extra_info", {})
            level = extra_info.get("level", None)
            problem_name = extra_info.get("problem_name", "")
            problem_id = extra_info.get("problem_id", None)

            if problem_id is not None and problem_name:
                baseline_key = f"{problem_id}_{problem_name}.py"
            elif problem_name:
                baseline_key = problem_name + ".py"
            else:
                baseline_key = None

            if baseline_key:
                baseline_runtime = get_baseline_runtime(level, baseline_key, baseline_timings)
                if baseline_runtime and baseline_runtime > 0:
                    speedup = baseline_runtime / exec_result.runtime
                    execution_details["speedup"] = speedup
                    execution_details["baseline_runtime"] = baseline_runtime
                    execution_details["performance_reward"] = min(max(speedup - 1.0, 0.0), 2.0)

    if hasattr(eval_result, "eval_status"):
        execution_details["eval_status"] = eval_result.eval_status
    if hasattr(eval_result, "eval_response"):
        execution_details["eval_response"] = eval_result.eval_response
    if preferred_device is not None:
        execution_details["eval_gpu"] = preferred_device

    original_extra_info = item.get("extra_info", {})
    item.update(sampling_params)
    item["timestamp"] = str(time.time())
    item["round_number"] = len([_ for _ in item["messages"] if _["role"] == "assistant"])

    return {
        "uid": item.pop("uid"),
        "messages": messages,
        "reward": reward,
        "instance_id": item.pop("instance_id"),
        "extra_info": {**original_extra_info, **item},
        "execution_details": execution_details,
    }


def gen_worker_process(
    task_queue,
    eval_queue,
    reserved_queue,
    done_queue,
    rollout_func,
    client,
    sampling_params,
    remote_eval_server_url,
    eval_semaphore,
):
    """Generate-only worker: LLM → route to eval_queue or reserved_queue (large shapes)."""
    logger.info("Gen worker started")
    while True:
        wait_if_rollout_paused()
        item = task_queue.get()
        if item == "STOP":
            break
        wait_if_rollout_paused()
        messages = rollout_func(item, client, sampling_params, remote_eval_server_url, eval_semaphore)
        item["messages"] = messages
        wait_if_rollout_paused()  # do not enqueue eval work while Megatron trains
        est = estimate_label_bytes(item.get("label", "") or "")
        if est >= LARGE_SHAPE_BYTES:
            logger.info(f"Large-shape gen est_bytes={est}; route reserved_queue")
            reserved_queue.put(item)
        else:
            eval_queue.put(item)
    done_queue.put("GEN_COMPLETE")


def eval_worker_process(
    eval_queue,
    reserved_queue,
    done_queue,
    reward_func,
    sampling_params,
    remote_eval_server_url,
    eval_semaphore,
    baseline_timings,
    preferred_device: int,
    reserved_gpu: int,
    consume_reserved: bool,
):
    """Eval-only worker pinned to one physical GPU. Reserved worker prefers reserved_queue."""
    logger.info(
        f"Eval worker started preferred_device={preferred_device} "
        f"consume_reserved={consume_reserved} reserved_gpu={reserved_gpu}"
    )
    while True:
        wait_if_rollout_paused()
        item = None
        from_reserved = False
        if consume_reserved:
            try:
                item = reserved_queue.get_nowait()
                from_reserved = True
            except Empty:
                try:
                    item = eval_queue.get(timeout=0.2)
                except Empty:
                    continue
        else:
            item = eval_queue.get()

        if item == "STOP":
            # Reserved worker only exits on reserved_queue STOP; put borrowable pills back.
            if consume_reserved and not from_reserved:
                eval_queue.put(item)
                time.sleep(0.05)
                continue
            break

        wait_if_rollout_paused()
        eval_result = reward_func(
            eval_semaphore,
            remote_eval_server_url,
            item,
            baseline_timings=baseline_timings,
            preferred_device=preferred_device,
        )

        # Borrowable OOM → reserved_queue (插队). Reserved GPU OOM is final.
        if getattr(eval_result, "eval_status", None) == "cuda_oom":
            if preferred_device != reserved_gpu and not consume_reserved:
                logger.warning(
                    f"CUDA OOM on phys={preferred_device}; requeue reserved_queue"
                )
                reserved_queue.put(item)
                continue
            logger.error(f"CUDA OOM on reserved phys={preferred_device}; fail sample")

        messages = item.get("messages", [])
        output_item = _build_output_item(
            item, messages, eval_result, sampling_params, baseline_timings, preferred_device
        )
        done_queue.put(output_item)

    done_queue.put("EVAL_COMPLETE")


def read_data_into_queue(
    input_file: str,
    skip_instance_ids: List[str],
    num_repeats: int,
    num_repeat_per_sample: int,
    task_queue: Queue,
    num_process: int,
):
    items = []
    actual_skipped_ids = []

    def _load(skip):
        loaded, skipped = [], []
        with open(input_file, "r") as r:
            for line in r:
                item = json.loads(line)
                if skip and item["instance_id"] in skip:
                    skipped.append(item["instance_id"])
                    continue
                loaded.append(item)
        return loaded, skipped

    items, actual_skipped_ids = _load(skip_instance_ids)
    # Resume metadata can cover the whole dataset after one pass; multi-epoch RL
    # must wrap and resample. Empty queue => workers exit with 0it forever.
    if skip_instance_ids and not items:
        logger.warning(
            "skip_instance_ids emptied the dataset (%d skipped); clearing skip for multi-epoch resample",
            len(actual_skipped_ids),
        )
        items, actual_skipped_ids = _load(None)

    random.shuffle(items)  # shuffle items
    logger.info(f"Read {len(items)} items, skipped {len(actual_skipped_ids)} items")

    if not items:
        raise ValueError(f"No items to roll out from {input_file} after skip handling")

    for _ in range(num_repeats):
        for item in items:
            for rollout_index in range(num_repeat_per_sample):
                item_repeat = copy.deepcopy(item)
                if "instance_id" not in item_repeat:
                    raise ValueError(f"instance_id not in item: {item}, the input data must have instance_id")

                if "uid" not in item_repeat:
                    item_repeat["uid"] = str(uuid.uuid4())

                item_repeat["rollout_index"] = rollout_index
                while task_queue.full():
                    time.sleep(1)
                task_queue.put(item_repeat)

    # Put STOP signal for each gen process
    for _ in range(num_process):
        task_queue.put("STOP")
    logger.info(f"Put {num_process} STOP signals into task_queue")


class KernelGenerator(BaseGenerator):
    """Trajectory generator for KernelBench (Scheme A: gen pool + pinned eval pool)."""

    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        num_repeat_per_sample=1,
        queue_size=1000000,
        num_process=10,
        task_type=TASK_TYPE,
        max_tokens=4096,
        num_repeats=10,
        skip_instance_ids: Optional[List[str]] = None,
        remote_eval_server_url: str = "http://localhost:18188",
        eval_concurrency: int = 10,
    ):
        super().__init__(
            remote_engine_url,
            remote_buffer_url,
            num_repeat_per_sample,
            queue_size,
            num_process,
            task_type,
            max_tokens,
            num_repeats,
            skip_instance_ids,
        )

        if remote_eval_server_url is None:
            remote_eval_server_url = DEFAULT_REMOTE_EVAL_SERVER_URL

        self.remote_eval_server_url = remote_eval_server_url
        self.eval_concurrency = eval_concurrency
        self.eval_semaphore = Semaphore(eval_concurrency)
        self.task_queue = Queue(maxsize=self.queue_size)
        self.eval_queue = Queue(maxsize=self.queue_size)
        self.reserved_queue = Queue(maxsize=self.queue_size)
        self.done_queue = Queue(maxsize=self.queue_size)

        self.sampling_params = SAMPLING_PARAMS.copy()
        self.sampling_params["max_tokens"] = max_tokens

        self.baseline_timings = load_baseline_timings()
        if self.baseline_timings:
            logger.info(f"Loaded baseline timings for {sum(len(v) for v in self.baseline_timings.values())} problems")
        else:
            logger.warning("No baseline timings loaded, performance rewards will not be calculated")

    def entry(self, input_file, rollout_func, reward_func, num_epoch=1):
        for _ in range(num_epoch):
            status = self.rollout_one_epoch(input_file, rollout_func, reward_func)

    def run(self, input_file, rollout_func, reward_func):
        warnings.warn("This method is deprecated. Please use rollout_one_epoch instead.")
        self.rollout_one_epoch(input_file, rollout_func, reward_func)

    def rollout_one_epoch(self, input_file, rollout_func, reward_func):
        worker_gpus = list(EVAL_WORKER_GPUS) if EVAL_WORKER_GPUS else []
        if not worker_gpus:
            raise ValueError("EVAL_WORKER_GPUS is empty; need pinned eval GPUs")

        reserved_gpu = EVAL_RESERVED_GPU
        if reserved_gpu not in worker_gpus:
            logger.warning(
                f"EVAL_RESERVED_GPU={reserved_gpu} not in EVAL_WORKER_GPUS={worker_gpus}; "
                f"OOM/large will have no reserved consumer"
            )

        num_gen = self.num_process
        num_eval = len(worker_gpus)
        borrowable_gpus = [g for g in worker_gpus if g != reserved_gpu]
        # Reserved worker is the one pinned to reserved_gpu (if present).
        num_borrowable = len(borrowable_gpus)
        has_reserved_worker = reserved_gpu in worker_gpus

        logger.info(
            f"Scheme A: gen_workers={num_gen} eval_workers={num_eval} "
            f"gpus={worker_gpus} reserved={reserved_gpu} "
            f"borrowable={borrowable_gpus} eval_concurrency={self.eval_concurrency}"
        )

        gen_processes = []
        for _ in range(num_gen):
            process = Process(
                target=partial(
                    gen_worker_process,
                    self.task_queue,
                    self.eval_queue,
                    self.reserved_queue,
                    self.done_queue,
                    rollout_func,
                    self.client,
                    self.sampling_params,
                    self.remote_eval_server_url,
                    self.eval_semaphore,
                ),
            )
            process.start()
            gen_processes.append(process)

        eval_processes = []
        for gpu in worker_gpus:
            consume_reserved = gpu == reserved_gpu
            process = Process(
                target=partial(
                    eval_worker_process,
                    self.eval_queue,
                    self.reserved_queue,
                    self.done_queue,
                    reward_func,
                    self.sampling_params,
                    self.remote_eval_server_url,
                    self.eval_semaphore,
                    self.baseline_timings,
                    gpu,
                    reserved_gpu,
                    consume_reserved,
                ),
            )
            process.start()
            eval_processes.append(process)

        reader_process = Process(
            target=read_data_into_queue,
            args=(
                input_file,
                set(self.skip_instance_ids) if self.skip_instance_ids else set(),
                self.num_repeats,
                self.num_repeat_per_sample,
                self.task_queue,
                num_gen,
            ),
        )
        reader_process.start()

        progress_bar = tqdm()
        gen_done = 0
        borrowable_done = 0
        reserved_done = 0
        borrowable_stops_sent = False
        reserved_stop_sent = False

        # Collect until all eval workers exit. Shutdown order:
        # 1) all gens done → STOP × borrowable on eval_queue
        # 2) all borrowable evals done → STOP on reserved_queue
        # 3) reserved eval done
        while True:
            if has_reserved_worker:
                finished = (
                    gen_done >= num_gen
                    and borrowable_done >= num_borrowable
                    and reserved_done >= 1
                )
            else:
                finished = gen_done >= num_gen and borrowable_done >= num_eval
            if finished:
                break

            item = self.done_queue.get()
            if item == "GEN_COMPLETE":
                gen_done += 1
                if gen_done >= num_gen and not borrowable_stops_sent:
                    n_stop = num_borrowable if has_reserved_worker else num_eval
                    for _ in range(n_stop):
                        self.eval_queue.put("STOP")
                    borrowable_stops_sent = True
                    logger.info(f"All {num_gen} gens done; sent {n_stop} STOP to eval_queue")
                    if not has_reserved_worker:
                        reserved_stop_sent = True
            elif item == "EVAL_COMPLETE":
                # Distinguish borrowable vs reserved via order of completion signals only:
                # borrowable exit first (after their STOPs); then we STOP reserved.
                if has_reserved_worker and borrowable_done < num_borrowable:
                    borrowable_done += 1
                    if borrowable_done >= num_borrowable and not reserved_stop_sent:
                        self.reserved_queue.put("STOP")
                        reserved_stop_sent = True
                        logger.info("All borrowable evals done; STOP reserved_queue")
                elif has_reserved_worker:
                    reserved_done += 1
                else:
                    borrowable_done += 1
            else:
                assert "reward" in item, f"reward not in item: {item}"
                assert "instance_id" in item, f"instance_id not in item: {item}"
                self.send_data_to_buffer(item)
                progress_bar.update(1)

        progress_bar.close()

        for process in gen_processes:
            process.join()
        for process in eval_processes:
            process.join()
        reader_process.join()

        return "finished"


def run_rollout(data: dict):
    logger.info(f"Starting kernel rollout with data: {data}")

    rollout_func = rollout_one_trajectory
    reward_func = submit_kernel_eval_request

    logger.info(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)
    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        logger.info(f"Set {k} to {v}", type(v))

    # num_process = generate workers. Eval workers = len(EVAL_WORKER_GPUS).
    num_gen = int(data.get("num_process", DEFAULT_GEN_NUM_PROCESS))
    num_eval = len(EVAL_WORKER_GPUS) or 5

    generator = KernelGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=num_gen,
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
        remote_eval_server_url=data.get("remote_eval_server_url", DEFAULT_REMOTE_EVAL_SERVER_URL),
        eval_concurrency=int(data.get("eval_concurrency", min(EVAL_CONCURRENCY, num_eval))),
    )

    generator.entry(data["input_file"], rollout_func, reward_func, int(data.get("num_epoch", 1)))
