import os
import time

import ray
import requests

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_group
from slime.utils.arguments import parse_args


def _http_no_proxy():
    return {"http": None, "https": None}


def _set_rollout_pause(paused: bool):
    """Stop buffer workers from hammering SGLang while Megatron trains."""
    path = os.environ.get("TF_ROLLOUT_PAUSE_FILE", "/tmp/tf_rollout_pause")
    if paused:
        with open(path, "w") as f:
            f.write("1\n")
        print(f"[train] rollout PAUSED ({path})", flush=True)
    else:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        print(f"[train] rollout RESUMED ({path})", flush=True)


def _ensure_no_residual_eval_procs(timeout_s: float = 120.0):
    """After borrow disable: kill leftover eval_worker_entry / orphan CUDA eval procs, wait until gone."""
    import signal
    import time as _time

    patterns = (
        "eval_worker_entry.py",
        "KBenchEval/scripts/eval_worker",
    )
    deadline = _time.time() + timeout_s
    attempt = 0
    while _time.time() < deadline:
        attempt += 1
        pids = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
            except OSError:
                continue
            if any(p in cmd for p in patterns):
                pids.append(int(pid))
        if not pids:
            if attempt > 1:
                print(f"[train] residual eval procs cleared after attempt={attempt}", flush=True)
            else:
                print("[train] no residual eval_worker procs", flush=True)
            return
        print(f"[train] residual eval procs={pids}; SIGKILL (attempt={attempt})", flush=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _time.sleep(2)
    raise RuntimeError(f"residual eval procs still present after {timeout_s}s")


def _set_borrow(enabled: bool):
    """Sync gate: enable for generate; disable kills in-flight evals and blocks until GPUs are free."""
    url = os.environ.get("EVAL_SERVER_URL", "http://127.0.0.1:18188").rstrip("/")
    endpoint = f"{url}/borrow/{'enable' if enabled else 'disable'}"
    settle = float(os.environ.get("EVAL_BORROW_POST_KILL_SETTLE_S", "2"))
    kill_wait = float(os.environ.get("EVAL_BORROW_KILL_WAIT_S", "15"))
    grace = settle if not enabled else 0.0
    timeout = 30 if enabled else settle + kill_wait * 8 + 200

    if enabled:
        try:
            r = requests.post(endpoint, timeout=timeout, proxies=_http_no_proxy())
            print(f"[train] borrow ENABLE -> {r.status_code} {r.text[:300]}", flush=True)
        except Exception as e:
            print(f"[train] borrow ENABLE failed ({endpoint}): {e}", flush=True)
        return

    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.post(endpoint, timeout=timeout, proxies=_http_no_proxy())
            body = {}
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text[:200]}
            print(f"[train] borrow DISABLE -> {r.status_code} {body}", flush=True)
            if r.status_code == 200 and isinstance(body, dict) and body.get("drained") is True:
                if body.get("killed"):
                    print(f"[train] eval drain killed phys={body.get('killed')}", flush=True)
                if body.get("vram_clear") is False:
                    print(
                        f"[train] borrowable VRAM not back to baseline: {body.get('borrowable_used_mib')} "
                        f"baseline={body.get('baseline_mib')}; retry ({attempt})",
                        flush=True,
                    )
                    # Do NOT soft-proceed — must be truly clear.
                else:
                    _ensure_no_residual_eval_procs()
                    return
            print(
                f"[train] drain not confirmed (attempt={attempt}); retry in 5s",
                flush=True,
            )
        except Exception as e:
            print(
                f"[train] borrow DISABLE failed attempt={attempt} ({endpoint}): {e}; retry in 5s",
                flush=True,
            )
        time.sleep(5)


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)

    actor_model = create_actor_group(args, pgs["actor"])

    # create the rollout generator, with sglang engines inside.
    rollout_generator = create_rollout_group(args, pgs["rollout"])

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_generator.data_buffer.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_generator.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_generator))

    colocate = bool(getattr(args, "colocate", False))

    if args.offload:
        if colocate:
            # colocate: wake SGLang; weight sync uses CPU actor copy via IPC.
            ray.get(rollout_generator.async_onload())
        else:
            # disaggregated: distributed NCCL sync needs Megatron params on GPU.
            ray.get(actor_model.async_onload())

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    # After first sync: free train GPUs for generate+borrow (disaggregated).
    if args.offload and not colocate:
        ray.get(actor_model.async_offload())

    # train loop (sync): generate while actor offloaded → train while sglang stays up
    # (disaggregated: rollout GPU is dedicated; only Megatron sleeps for eval borrow).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0:
            _set_rollout_pause(False)
            _set_borrow(True)
            ray.get(rollout_generator.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id))

        # Generate phase: Megatron should be asleep when --offload; allow borrow on train GPUs.
        _set_rollout_pause(False)
        _set_borrow(True)
        ray.get(rollout_generator.async_generate(rollout_id))

        # Train phase: pause new work → kill stale in-flight evals → verify drain/residuals.
        _set_rollout_pause(True)
        _set_borrow(False)

        # Only colocate needs SGLang GPU sleep; disaggregated keeps SGLang on its own card.
        if args.offload and colocate:
            ray.get(rollout_generator.async_offload())

        ray.get(actor_model.async_train(rollout_id))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_generator.data_buffer.save.remote(rollout_id))

        if args.offload and colocate:
            # colocate: offload actor → wake SGLang → IPC weight update from CPU copy.
            ray.get(actor_model.async_offload())
            ray.get(rollout_generator.async_onload())
            ray.get(actor_model.async_update_weights())
        elif args.offload:
            # disaggregated: model still on GPU after train → NCCL sync → then sleep for borrow.
            ray.get(actor_model.async_update_weights())
            ray.get(actor_model.async_offload())
        else:
            ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            _set_rollout_pause(False)
            _set_borrow(True)
            ray.get(rollout_generator.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
