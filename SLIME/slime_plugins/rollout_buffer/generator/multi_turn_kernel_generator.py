import copy
import json
import logging
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Process, Queue, Semaphore
from queue import Empty
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from slime_plugins.rollout_buffer.generator.base_generator import BaseGenerator
from slime_plugins.rollout_buffer.generator.kernel_generator import (
    DEFAULT_REMOTE_EVAL_SERVER_URL,
    EVAL_CONCURRENCY,
    EVAL_RESERVED_GPU,
    EVAL_WORKER_GPUS,
    LARGE_SHAPE_BYTES,
    SAMPLING_PARAMS,
    estimate_label_bytes,
    get_baseline_runtime,
    load_baseline_timings,
    query_llm_with_retry,
    submit_kernel_eval_request,
    wait_if_rollout_paused,
)
from slime_plugins.rollout_buffer.generator.kernelbench_config import KERNELBENCH_REWARDS, KERNELBENCH_COT_SETTINGS
from slime_plugins.rollout_buffer.generator.reward_utils.kernel_utils import extract_last_code, strip_thinking_tags

logger = logging.getLogger(__name__)

# Task type for registration with buffer server
TASK_TYPE = "kernelbench_multiturn"

# Multi-turn configuration
DEFAULT_MAX_TURNS = 3  # Maximum number of turns
DEFAULT_GAMMA = 0.4  # Discount factor for aggregated return

# Logging configuration
ENABLE_DETAILED_LOGGING = True  # Enable detailed multi-turn logging
LOG_DIR = "/root/slime/multi_turn_logs"  # Directory for detailed logs


def save_multi_turn_data_to_local(data: dict, turn_idx: int = None, is_final: bool = False):
    """Save multi-turn training data to local files for analysis.

    Args:
        data: Data to save (can be turn data or final trajectory data)
        turn_idx: Turn index if saving turn-specific data
        is_final: Whether this is the final trajectory data
    """
    if not ENABLE_DETAILED_LOGGING:
        return

    try:
        # Create log directory if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        instance_id = data.get("instance_id", "unknown")

        if is_final:
            # Save final trajectory data
            filename = f"{LOG_DIR}/trajectory_{instance_id}_{timestamp}.json"
            log_data = {
                "timestamp": timestamp,
                "instance_id": instance_id,
                "final_reward": data.get("reward", 0),
                "num_turns": data.get("num_turns", 0),
                "turn_rewards": data.get("turn_rewards", []),
                "aggregated_return": data.get("aggregated_return", 0),
                "history": data.get("history", []),
                "messages": data.get("messages", []),
                "execution_details": data.get("execution_details", {}),
                "extra_info": data.get("extra_info", {}),
            }

            # Log summary to console
            logger.info(f"=== Final Trajectory Summary for {instance_id} ===")
            logger.info(f"Total turns: {log_data['num_turns']}")
            logger.info(f"Turn rewards: {log_data['turn_rewards']}")
            logger.info(f"Aggregated return: {log_data['aggregated_return']:.4f}")
            logger.info(f"Final correctness: {log_data['execution_details'].get('final_correctness', False)}")
            logger.info(f"Final speedup: {log_data['execution_details'].get('final_speedup', 0):.2f}x")

        else:
            # Save turn-specific data
            filename = f"{LOG_DIR}/turn_{instance_id}_t{turn_idx}_{timestamp}.json"
            log_data = {"timestamp": timestamp, "instance_id": instance_id, "turn_idx": turn_idx, "turn_data": data}

            # Log turn summary to console
            logger.info(f"=== Turn {turn_idx} for {instance_id} ===")
            if "eval_result" in data:
                eval_result = data["eval_result"]
                logger.info(f"Compiled: {eval_result.get('compiled', False)}")
                logger.info(f"Correctness: {eval_result.get('correctness', False)}")
                logger.info(f"Runtime: {eval_result.get('runtime', 0):.3f}ms")
                logger.info(f"Speedup: {eval_result.get('speedup', 0):.2f}x")
                logger.info(f"Reward: {data.get('reward', 0):.4f}")
                if eval_result.get("error_message"):
                    logger.info(f"Error: {eval_result['error_message'][:100]}...")

        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)

        logger.debug(f"Saved multi-turn data to {filename}")

    except Exception as e:
        logger.error(f"Error saving multi-turn data: {e}")


def log_turn_summary(turn_idx: int, instance_id: str, turn_data: dict):
    """Print a concise summary of a turn's results."""
    eval_result = turn_data.get("eval_result", {})
    reward = turn_data.get("reward", 0)

    status_symbols = {True: "✓", False: "✗"}

    compiled = eval_result.get("compiled", False)
    correct = eval_result.get("correctness", False)
    runtime = eval_result.get("runtime", 0)
    speedup = eval_result.get("speedup", 0)

    logger.info(
        f"[Turn {turn_idx + 1}] {instance_id}: "
        f"Compile:{status_symbols[compiled]} "
        f"Correct:{status_symbols[correct]} "
        f"Runtime:{runtime:.2f}ms "
        f"Speedup:{speedup:.2f}x "
        f"Reward:{reward:.3f}"
    )


def construct_multi_turn_prompt(
    original_prompt: List[dict],
    turn_idx: int,
    history: List[dict],
) -> List[dict]:
    """Construct multi-turn prompt with history of previous attempts.

    Args:
        original_prompt: The original instruction prompt
        turn_idx: Current turn index (0-based)
        history: List of previous turns with kernel code and evaluation results

    Returns:
        Constructed prompt with history context
    """
    if turn_idx == 0:
        # First turn: use original prompt as-is
        return copy.deepcopy(original_prompt)

    # For subsequent turns, prepend history to the original prompt
    messages = copy.deepcopy(original_prompt)

    # Build context from previous attempts
    context_parts = []
    for i, turn_data in enumerate(history):
        context_parts.append(f"## Previous Attempt {i + 1}")

        # Add the generated kernel code
        if "kernel_code" in turn_data:
            context_parts.append("Generated Kernel:")
            context_parts.append("```python")
            context_parts.append(turn_data["kernel_code"])
            context_parts.append("```")

        # Add evaluation results
        if "eval_result" in turn_data:
            eval_result = turn_data["eval_result"]
            context_parts.append("\nEvaluation Results:")
            context_parts.append(f"- Compilation: {'✓ Passed' if eval_result.get('compiled', False) else '✗ Failed'}")
            context_parts.append(
                f"- Correctness: {'✓ Passed' if eval_result.get('correctness', False) else '✗ Failed'}"
            )

            if eval_result.get("runtime", 0) > 0:
                runtime = eval_result["runtime"]
                speedup = eval_result.get("speedup", 0)
                context_parts.append(f"- Runtime: {runtime:.3f}ms")
                if speedup > 0:
                    context_parts.append(f"- Speedup: {speedup:.2f}x")

            # Add any error messages
            if eval_result.get("error_message"):
                context_parts.append(f"- Error: {eval_result['error_message']}")

        context_parts.append("")  # Empty line between attempts

    # Add instruction for improvement
    context_parts.append(f"## Attempt {turn_idx + 1}")
    context_parts.append("Based on the previous attempts above, generate an improved kernel that:")
    context_parts.append("1. Fixes any compilation or correctness errors")
    context_parts.append("2. Improves performance if possible")
    context_parts.append("3. Maintains the same functionality as required")
    context_parts.append("\nGenerate the improved kernel code:")

    # Prepend context to the user message
    context_str = "\n".join(context_parts)

    # Find the user message and prepend context
    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = context_str + "\n\n" + msg["content"]
            break

    return messages


def calculate_aggregated_return(
    turn_rewards: List[float],
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Calculate aggregated return with discount factor.

    Args:
        turn_rewards: List of rewards for each turn
        gamma: Discount factor (default 0.4)

    Returns:
        Aggregated discounted return
    """
    if not turn_rewards:
        return 0.0

    aggregated_return = 0.0
    for t, reward in enumerate(turn_rewards):
        aggregated_return += (gamma**t) * reward

    return aggregated_return


def _submit_eval_with_pin(
    eval_semaphore: Semaphore,
    remote_eval_server_url: str,
    eval_item: dict,
    preferred_device: Optional[int],
    backend: str = "triton",
    baseline_timings: Optional[Dict] = None,
    eval_route: Optional[Dict] = None,
):
    """Pin eval to a GPU; large shapes / OOM fall back to reserved (same as single-turn).

    When eval_route is set (gen/eval split), hand the job to a pinned eval worker via
    eval_queue/reserved_queue and wait on reply_queue — same STOP/EVAL_COMPLETE drain as single.
    """
    est = estimate_label_bytes(eval_item.get("label", "") or "")
    if eval_route is not None:
        import uuid as _uuid

        reply_id = str(_uuid.uuid4())
        payload = {
            "eval_item": eval_item,
            "backend": backend,
            "baseline_timings": baseline_timings,
            "_reply_id": reply_id,
        }
        if est >= LARGE_SHAPE_BYTES:
            logger.info(f"Multi-turn large-shape est_bytes={est}; route reserved_queue")
            eval_route["reserved_queue"].put(payload)
        else:
            eval_route["eval_queue"].put(payload)
        reply_q = eval_route["reply_queue"]
        while True:
            wait_if_rollout_paused()
            msg = reply_q.get()
            if msg.get("_reply_id") == reply_id:
                return msg["eval_result"]
            # Shared reply bus: not ours — put back for the owning gen.
            reply_q.put(msg)
            time.sleep(0.01)

    if est >= LARGE_SHAPE_BYTES:
        devices = [EVAL_RESERVED_GPU]
        logger.info(f"Multi-turn large-shape est_bytes={est}; prefer reserved={EVAL_RESERVED_GPU}")
    elif preferred_device is not None:
        devices = [preferred_device]
        if preferred_device != EVAL_RESERVED_GPU:
            devices.append(EVAL_RESERVED_GPU)
    else:
        devices = [None]

    last = None
    for i, device in enumerate(devices):
        wait_if_rollout_paused()
        last = submit_kernel_eval_request(
            eval_semaphore,
            remote_eval_server_url,
            eval_item,
            backend=backend,
            baseline_timings=baseline_timings,
            preferred_device=device,
        )
        if getattr(last, "eval_status", None) != "cuda_oom":
            return last
        if i + 1 < len(devices):
            logger.warning(
                f"Multi-turn CUDA OOM on device={device}; retry reserved/next={devices[i + 1]}"
            )
    return last


def rollout_multi_turn_trajectory(
    item: dict,
    client: OpenAI,
    sampling_params: dict,
    remote_eval_server_url: str,
    eval_semaphore: Semaphore,
    backend: str = "triton",
    max_turns: int = DEFAULT_MAX_TURNS,
    gamma: float = DEFAULT_GAMMA,
    baseline_timings: Optional[Dict] = None,
    use_native_template: bool = True,
    preferred_device: Optional[int] = None,
    eval_route: Optional[Dict] = None,
) -> Tuple[List[dict], float, List[float], List[dict]]:
    """Execute multi-turn rollout for kernel generation.

    Args:
        item: Data item with prompt and metadata
        client: OpenAI client for LLM queries
        sampling_params: Sampling parameters for generation
        remote_eval_server_url: URL of evaluation server
        eval_semaphore: Semaphore for evaluation concurrency
        backend: Backend type (triton/cuda)
        max_turns: Maximum number of turns
        gamma: Discount factor for aggregated return
        baseline_timings: Baseline timing data for performance comparison
        preferred_device: Physical GPU for eval (蹭卡 / reserved pin)
        eval_route: Optional gen/eval split queues {eval_queue, reserved_queue, reply_queue}

    Returns:
        Tuple of (final_messages, aggregated_return, turn_rewards, history)
    """
    original_prompt = item["prompt"]
    history = []
    turn_rewards = []
    messages = None
    
    def eval_content(content: str) -> dict:
        # Handle thinking tags based on configuration
        if KERNELBENCH_COT_SETTINGS.get("strip_think_tags", True):
            # Strip thinking tags for code extraction and evaluation
            # But preserve original content for training
            cleaned_content, thinking_content = strip_thinking_tags(content)
        else:
            # Don't strip if disabled in config
            cleaned_content = content
            thinking_content = ""
        
        # Extract kernel code from cleaned content (without think tags if stripped)
        kernel_code = extract_last_code(cleaned_content, strip_think_tags=False)  # Already handled above

        # Create evaluation item with cleaned content for evaluation
        eval_item = copy.deepcopy(item)
        # Create temporary messages with cleaned content for evaluation only
        eval_messages = copy.deepcopy(messages[:-1])  # All but last message
        eval_messages.append({
            "role": "assistant",
            "content": cleaned_content  # Use cleaned content for evaluation
        })
        eval_item["messages"] = eval_messages
        
        eval_result = _submit_eval_with_pin(
            eval_semaphore,
            remote_eval_server_url,
            eval_item,
            preferred_device=preferred_device,
            backend=backend,
            baseline_timings=baseline_timings,
            eval_route=eval_route,
        )
        
        # Return kernel code, eval_result, and thinking content for logging
        return kernel_code, eval_result, thinking_content
        
    
    # original_prompt is already a list of messages, so we use it directly
    # For Qwen3 models, the prompt already contains the proper system message
    messages = copy.deepcopy(original_prompt) if isinstance(original_prompt, list) else [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates kernel code.",
        },
        original_prompt
    ]
    
    for turn_idx in range(max_turns):
        if use_native_template:
            # Log the current message count and last message role for debugging
            logger.info(f"[Turn {turn_idx}] Starting with {len(messages)} messages, last role: {messages[-1]['role'] if messages else 'None'}")
            
            # Generate response
            assistant_content = None  # Initialize to avoid UnboundLocalError
            try:
                assistant_content = query_llm_with_retry(client, messages, sampling_params, tools=None)
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                }
                
                # Log response length for debugging
                logger.info(f"[Turn {turn_idx}] Generated response with {len(assistant_content)} characters")
                if len(assistant_content) < 50:
                    logger.warning(f"[Turn {turn_idx}] Short response detected: {assistant_content[:100]}")
                    
            except Exception as e:
                logger.error(f"Error in turn {turn_idx}: {e}")
                assistant_content = f"Error: Failed to generate response in turn {turn_idx}"
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                }

            messages.append(assistant_message)
        else:
            # Construct prompt with history
            prompt = construct_multi_turn_prompt(original_prompt, turn_idx, history)
            
            # Generate response
            assistant_content = None  # Initialize to avoid UnboundLocalError
            try:
                assistant_content = query_llm_with_retry(client, prompt, sampling_params, tools=None)
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                }
                
                # Log response length for debugging
                logger.info(f"[Turn {turn_idx}] Generated response with {len(assistant_content)} characters")
                
            except Exception as e:
                logger.error(f"Error in turn {turn_idx}: {e}")
                assistant_content = f"Error: Failed to generate response in turn {turn_idx}"
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                }

            # Build messages for evaluation
            messages = prompt + [assistant_message] # in this case we don't accumulate messages
            
        kernel_code, eval_result, thinking_content = eval_content(assistant_content)

        # Extract reward and evaluation details
        turn_reward = eval_result.reward if hasattr(eval_result, "reward") else 0.0
        turn_rewards.append(turn_reward)

        # Build history entry
        history_entry = {
            "turn_idx": turn_idx,
            "kernel_code": kernel_code if kernel_code else "",
            "eval_result": {
                "compiled": eval_result.exec_result.compiled if hasattr(eval_result, "exec_result") else False,
                "correctness": eval_result.exec_result.correctness if hasattr(eval_result, "exec_result") else False,
                "runtime": eval_result.exec_result.runtime if hasattr(eval_result, "exec_result") else 0,
                "speedup": 0,  # Will be calculated if baseline available
                "error_message": eval_result.eval_response if hasattr(eval_result, "eval_response") else "",
            },
            "reward": turn_reward,
        }
        
        # Add CoT-related fields based on configuration
        if KERNELBENCH_COT_SETTINGS.get("log_thinking_content", True) and thinking_content:
            history_entry["thinking_content"] = thinking_content
        
        if KERNELBENCH_COT_SETTINGS.get("preserve_original_in_training", True):
            history_entry["original_content"] = assistant_content  # Full response with think tags

        # Calculate speedup if applicable
        if hasattr(eval_result, "exec_result") and eval_result.exec_result.runtime > 0 and baseline_timings:
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
                    speedup = baseline_runtime / eval_result.exec_result.runtime
                    history_entry["eval_result"]["speedup"] = speedup

        history.append(history_entry)

        # Log turn details
        log_turn_summary(turn_idx, item.get("instance_id", "unknown"), history_entry)

        # Save turn data to local file
        # Use the appropriate prompt based on template type
        prompt_for_logging = messages if use_native_template else prompt
        turn_log_data = {
            "instance_id": item.get("instance_id", "unknown"),
            "turn_idx": turn_idx,
            "prompt": prompt_for_logging,
            "response": assistant_content,  # Full response with think tags
            "kernel_code": kernel_code,
            "eval_result": history_entry["eval_result"],
            "reward": turn_reward,
            "extra_info": item.get("extra_info", {}),
        }
        
        # Add thinking content to log if available
        if KERNELBENCH_COT_SETTINGS.get("log_thinking_content", True) and thinking_content:
            turn_log_data["thinking_content"] = thinking_content
        save_multi_turn_data_to_local(turn_log_data, turn_idx=turn_idx, is_final=False)

        # Early termination conditions
        if turn_reward >= KERNELBENCH_REWARDS["correctness"] + 1.0:
            # Got correctness + good performance, no need to continue
            logger.info(f"Early termination at turn {turn_idx}: achieved good performance")
            break

        # Add improvement instruction for next turn (if not the last turn)
        if use_native_template and turn_idx < max_turns - 1:
            # Build improvement instruction based on evaluation results
            improvement_parts = []
            improvement_parts.append("Based on the previous attempt above, generate an improved kernel that:")
            
            if not history_entry["eval_result"]["compiled"]:
                improvement_parts.append("1. Fixes the compilation errors")
            elif not history_entry["eval_result"]["correctness"]:
                improvement_parts.append("1. Fixes the correctness issues")
            else:
                improvement_parts.append("1. Maintains correctness")
            
            improvement_parts.append("2. Improves performance if possible")
            improvement_parts.append("3. Maintains the same functionality as required")
            
            # Add evaluation feedback if available
            if history_entry["eval_result"].get("error_message"):
                improvement_parts.append(f"\nError from previous attempt: {history_entry['eval_result']['error_message'][:200]}")
            
            improvement_parts.append("\nPlease generate the improved kernel code:")
            
            improvement_message = {
                "role": "user",
                "content": "\n".join(improvement_parts)
            }
            messages.append(improvement_message)
            logger.debug(f"Added improvement instruction for turn {turn_idx + 1}")

        # Removed early termination for multiple failures - let it try all turns
        # if turn_idx > 0 and all(r == 0 for r in turn_rewards):
        #     # Multiple failures, unlikely to improve
        #     logger.info(f"Early termination at turn {turn_idx}: multiple failures")
        #     break

    aggregated_return = calculate_aggregated_return(turn_rewards, gamma)

    return messages, aggregated_return, turn_rewards, history


def gen_worker_process_multi_turn(
    task_queue,
    eval_queue,
    reserved_queue,
    reply_queue,
    done_queue,
    rollout_func,
    client,
    sampling_params,
    remote_eval_server_url,
    eval_semaphore,
    baseline_timings,
    max_turns,
    gamma,
):
    """Generate-only multi-turn worker: LLM + queue evals to pinned eval workers."""
    logger.info("Multi-turn gen worker started")
    eval_route = {
        "eval_queue": eval_queue,
        "reserved_queue": reserved_queue,
        "reply_queue": reply_queue,
    }
    while True:
        wait_if_rollout_paused()
        item = task_queue.get()
        if item == "STOP":
            break

        messages, aggregated_return, turn_rewards, history = rollout_func(
            item,
            client,
            sampling_params,
            remote_eval_server_url,
            eval_semaphore,
            max_turns=max_turns,
            gamma=gamma,
            baseline_timings=baseline_timings,
            preferred_device=None,  # routing handled by eval workers + large→reserved
            eval_route=eval_route,
        )

        item["messages"] = messages
        item["reward"] = aggregated_return
        item["turn_rewards"] = turn_rewards
        item["history"] = history
        item["num_turns"] = len(turn_rewards)
        item["rollout_index"] = 1

        if history:
            last_turn = history[-1]
            execution_details = {
                "final_compiled": last_turn["eval_result"]["compiled"],
                "final_correctness": last_turn["eval_result"]["correctness"],
                "final_runtime": last_turn["eval_result"]["runtime"],
                "final_speedup": last_turn["eval_result"].get("speedup", 0),
                "num_turns": len(history),
                "turn_rewards": turn_rewards,
                "aggregated_return": aggregated_return,
            }
        else:
            execution_details = {
                "num_turns": 0,
                "turn_rewards": [],
                "aggregated_return": 0,
            }

        original_extra_info = item.get("extra_info", {})
        item.update(sampling_params)
        item["timestamp"] = str(time.time())

        output_item = {
            "uid": item.pop("uid"),
            "messages": messages,
            "reward": aggregated_return,
            "instance_id": item.get("instance_id", "unknown"),
            "extra_info": {**original_extra_info, **item},
            "execution_details": execution_details,
            "multi_turn_data": {
                "history": history,
                "turn_rewards": turn_rewards,
                "aggregated_return": aggregated_return,
                "gamma": gamma,
                "max_turns": max_turns,
            },
        }

        final_log_data = {
            "instance_id": output_item["instance_id"],
            "reward": aggregated_return,
            "num_turns": len(turn_rewards),
            "turn_rewards": turn_rewards,
            "aggregated_return": aggregated_return,
            "history": history,
            "messages": messages,
            "execution_details": execution_details,
            "extra_info": original_extra_info,
        }
        save_multi_turn_data_to_local(final_log_data, is_final=True)
        output_item["instance_id"] = final_log_data["instance_id"]
        done_queue.put(output_item)

    done_queue.put("GEN_COMPLETE")


def eval_worker_process_multi_turn(
    eval_queue,
    reserved_queue,
    reply_queue,
    done_queue,
    remote_eval_server_url,
    eval_semaphore,
    preferred_device: int,
    reserved_gpu: int,
    consume_reserved: bool,
):
    """Eval-only worker pinned to one physical GPU (mirrors single-turn drain)."""
    logger.info(
        f"Multi-turn eval worker started preferred_device={preferred_device} "
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
            if consume_reserved and not from_reserved:
                eval_queue.put(item)
                time.sleep(0.05)
                continue
            break

        wait_if_rollout_paused()
        reply_id = item["_reply_id"]
        eval_item = item["eval_item"]
        backend = item.get("backend", "triton")
        baseline_timings = item.get("baseline_timings")

        eval_result = submit_kernel_eval_request(
            eval_semaphore,
            remote_eval_server_url,
            eval_item,
            backend=backend,
            baseline_timings=baseline_timings,
            preferred_device=preferred_device,
        )

        if getattr(eval_result, "eval_status", None) == "cuda_oom":
            if preferred_device != reserved_gpu and not consume_reserved:
                logger.warning(
                    f"Multi-turn CUDA OOM on phys={preferred_device}; requeue reserved_queue"
                )
                reserved_queue.put(item)
                continue
            logger.error(f"Multi-turn CUDA OOM on reserved phys={preferred_device}; fail sample")

        reply_queue.put({"_reply_id": reply_id, "eval_result": eval_result})

    done_queue.put("EVAL_COMPLETE")


class MultiTurnKernelGenerator(BaseGenerator):
    """Multi-turn trajectory generator for KernelBench (gen pool + pinned eval pool)."""

    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        num_repeat_per_sample=1,
        queue_size=1000000,
        num_process=10,
        task_type=None,  # Will use TASK_TYPE constant if not provided
        max_tokens=4096,
        num_repeats=10,
        skip_instance_ids: Optional[List[str]] = None,
        remote_eval_server_url: str = "http://localhost:18188",
        eval_concurrency: int = 10,
        max_turns: int = DEFAULT_MAX_TURNS,
        gamma: float = DEFAULT_GAMMA,
    ):
        super().__init__(
            remote_engine_url,
            remote_buffer_url,
            num_repeat_per_sample,
            queue_size,
            num_process,
            task_type or TASK_TYPE,  # Use constant if not provided
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
        # Shared reply bus inherited by gen+eval via Process args (not nested in eval_queue).
        self.reply_queue = Queue(maxsize=self.queue_size)
        self.done_queue = Queue(maxsize=self.queue_size)

        # Multi-turn parameters
        self.max_turns = max_turns
        self.gamma = gamma

        # Initialize sampling_params
        self.sampling_params = SAMPLING_PARAMS.copy()
        self.sampling_params["max_tokens"] = max_tokens

        # Load baseline timings
        self.baseline_timings = load_baseline_timings()
        if self.baseline_timings:
            logger.info(f"Loaded baseline timings for {sum(len(v) for v in self.baseline_timings.values())} problems")
        else:
            logger.warning("No baseline timings loaded, performance rewards will not be calculated")

        logger.info(f"Multi-turn kernel generator initialized with max_turns={max_turns}, gamma={gamma}")

    def rollout_one_epoch(self, input_file, rollout_func):
        """Execute one epoch of multi-turn rollout (gen/eval split like single-turn)."""
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
        num_borrowable = len(borrowable_gpus)
        has_reserved_worker = reserved_gpu in worker_gpus

        logger.info(
            f"Multi-turn Scheme A: gen_workers={num_gen} eval_workers={num_eval} "
            f"gpus={worker_gpus} reserved={reserved_gpu} borrowable={borrowable_gpus}"
        )

        gen_processes = []
        for _ in range(num_gen):
            process = Process(
                target=partial(
                    gen_worker_process_multi_turn,
                    self.task_queue,
                    self.eval_queue,
                    self.reserved_queue,
                    self.reply_queue,
                    self.done_queue,
                    rollout_func,
                    self.client,
                    self.sampling_params,
                    self.remote_eval_server_url,
                    self.eval_semaphore,
                    self.baseline_timings,
                    self.max_turns,
                    self.gamma,
                ),
            )
            process.start()
            gen_processes.append(process)

        eval_processes = []
        for gpu in worker_gpus:
            consume_reserved = gpu == reserved_gpu
            process = Process(
                target=partial(
                    eval_worker_process_multi_turn,
                    self.eval_queue,
                    self.reserved_queue,
                    self.reply_queue,
                    self.done_queue,
                    self.remote_eval_server_url,
                    self.eval_semaphore,
                    gpu,
                    reserved_gpu,
                    consume_reserved,
                ),
            )
            process.start()
            eval_processes.append(process)

        from slime_plugins.rollout_buffer.generator.kernel_generator import read_data_into_queue

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

        progress_bar = tqdm(desc="Multi-turn rollout")
        gen_done = 0
        borrowable_done = 0
        reserved_done = 0
        borrowable_stops_sent = False
        reserved_stop_sent = False

        # Same shutdown order as single-turn:
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
                if "multi_turn_data" in item:
                    mt_data = item["multi_turn_data"]
                    logger.info(
                        f"Instance {item['instance_id']}: "
                        f"turns={len(mt_data['turn_rewards'])}, "
                        f"rewards={mt_data['turn_rewards']}, "
                        f"aggregated={mt_data['aggregated_return']:.3f}"
                    )

        progress_bar.close()

        for process in gen_processes:
            process.join()
        for process in eval_processes:
            process.join()
        reader_process.join()

        return "finished"

    def entry(self, input_file, rollout_func, num_epoch=1):
        """Entry point for multi-turn rollout."""
        for epoch in range(num_epoch):
            logger.info(f"Starting epoch {epoch + 1}/{num_epoch}")
            status = self.rollout_one_epoch(input_file, rollout_func)
            logger.info(f"Epoch {epoch + 1} completed with status: {status}")


def run_rollout(data: dict):
    """Run multi-turn rollout with provided configuration (entry point for buffer server)."""
    logger.info(f"Starting multi-turn kernel rollout with data: {data}")

    rollout_func = rollout_multi_turn_trajectory

    logger.info(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)

    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        logger.info(f"Set {k} to {v}")

    # Extract multi-turn parameters
    max_turns = int(data.get("max_turns", DEFAULT_MAX_TURNS))
    gamma = float(data.get("gamma", DEFAULT_GAMMA))

    generator = MultiTurnKernelGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
        remote_eval_server_url=data.get("remote_eval_server_url", DEFAULT_REMOTE_EVAL_SERVER_URL),
        eval_concurrency=int(data.get("eval_concurrency", EVAL_CONCURRENCY)),
        max_turns=max_turns,
        gamma=gamma,
    )

    generator.entry(data["input_file"], rollout_func, int(data.get("num_epoch", 1)))
