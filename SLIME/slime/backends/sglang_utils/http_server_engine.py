import multiprocessing
import os
import time
from typing import List, Optional

import requests
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError


def _session_no_proxy() -> requests.Session:
    """Health/control calls must not use http_proxy (common in this image).

    Proxying 172.17.0.2 returns a fake HTTP 503 and hangs launch forever.
    """
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    # Ray/CUDA workers: fork after CUDA init deadlocks sglang post-load init.
    # spawn matches a clean process (solo probe becomes healthy in ~30s).
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return

    base_url = server_args.url()

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {server_args.api_key}",
    }

    # Wait until HTTP 200. Do NOT proceed early on 503 — that left train hung in
    # connect_rollout_engines while sglang was still Starting/UnHealthy.
    # Optional: SGLANG_LAUNCH_PROCEED_AFTER_503>0 re-enables the old escape hatch.
    deadline = time.time() + float(os.environ.get("SGLANG_LAUNCH_HEALTH_TIMEOUT", "600"))
    proceed_after = float(os.environ.get("SGLANG_LAUNCH_PROCEED_AFTER_503", "0"))
    # /health is enough when skip_server_warmup=True; /health_generate needs a token.
    health_path = os.environ.get("SGLANG_LAUNCH_HEALTH_PATH", "/health")
    first_http_t = None
    with _session_no_proxy() as session:
        while True:
            try:
                response = session.get(
                    f"{base_url}{health_path}", headers=headers, timeout=30
                )
                if response.status_code == 200:
                    print(
                        f"[sglang-launch] {health_path} ready status=200 url={base_url}",
                        flush=True,
                    )
                    break
                if first_http_t is None:
                    first_http_t = time.time()
                print(
                    f"[sglang-launch] {health_path} status={response.status_code} "
                    f"url={base_url} alive={p.is_alive()}",
                    flush=True,
                )
                if (
                    proceed_after > 0
                    and response.status_code == 503
                    and first_http_t is not None
                    and (time.time() - first_http_t) >= proceed_after
                    and p.is_alive()
                ):
                    print(
                        f"[sglang-launch] proceeding after {proceed_after:.0f}s of HTTP 503 "
                        f"(server process still alive)",
                        flush=True,
                    )
                    break
            except (requests.RequestException, NewConnectionError) as e:
                print(f"[sglang-launch] {health_path} wait: {e}", flush=True)

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")
            if time.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for sglang {health_path} at {base_url} "
                    f"(last status may be 503/Starting). "
                    f"Set SGLANG_LAUNCH_HEALTH_TIMEOUT / SGLANG_LAUNCH_PROCEED_AFTER_503."
                )

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=30)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")
            if time.time() > deadline:
                print(
                    f"[sglang-launch] flush_cache timed out at {base_url}; continuing",
                    flush=True,
                )
                break

            time.sleep(2)

    return p


class HttpServerEngineAdapter:
    """
    You can use this class to launch a server from a VerlEngine instance.
    We recommend using this class only you need to use http server.
    Otherwise, you can use Engine directly.
    """

    def __init__(self, router_ip=None, router_port=None, **kwargs):
        self.router_ip = router_ip
        self.router_port = router_port
        self.server_args = ServerArgs(**kwargs)
        self.node_rank = self.server_args.node_rank
        print(f"Launch HttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}")
        self.process = launch_server_process(self.server_args)
        if self.node_rank == 0 and self.router_ip and self.router_port:
            # Newer sglang-router removed GET/POST /add_worker?url=... in favor of
            # POST /workers with JSON body. Keep a fallback for older routers.
            worker_url = f"http://{self.server_args.host}:{self.server_args.port}"
            with _session_no_proxy() as session:
                try:
                    r = session.post(
                        f"http://{self.router_ip}:{self.router_port}/workers",
                        json={"url": worker_url},
                        timeout=30,
                    )
                    if r.status_code in (200, 201, 202):
                        print(
                            f"[sglang-launch] registered worker via /workers "
                            f"status={r.status_code} body={r.text[:200]}",
                            flush=True,
                        )
                    else:
                        # legacy query-param API
                        r2 = session.post(
                            f"http://{self.router_ip}:{self.router_port}/add_worker"
                            f"?url={worker_url}",
                            timeout=30,
                        )
                        print(
                            f"[sglang-launch] /workers->{r.status_code}; "
                            f"legacy add_worker->{r2.status_code} {r2.text[:200]}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[sglang-launch] worker registration failed: {e}", flush=True)

    def _make_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        with _session_no_proxy() as session:
            response = session.post(url, json=payload or {}, timeout=120)
            response.raise_for_status()
            return response.json()

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        with _session_no_proxy() as session:
            while True:
                try:
                    response = session.get(
                        f"http://{self.server_args.host}:{self.server_args.port}/flush_cache",
                        timeout=30,
                    )
                    if response.status_code == 200:
                        break
                except NewConnectionError as e:
                    raise e
                except Exception as e:
                    print(f"Error flushing cache: {e}")
                    continue

    def shutdown(self):
        with _session_no_proxy() as session:
            try:
                session.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_args.host}:{self.server_args.port}",
                    timeout=30,
                )
            except Exception:
                pass
        kill_process_tree(self.process.pid)

    def release_memory_occupation(self):
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self):
        return self._make_request("resume_memory_occupation")

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        return self._make_request(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).removeprefix("torch.") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
            },
        )

    def pause_generation(self):
        return self._make_request("pause_generation")

    def continue_generation(self):
        return self._make_request("continue_generation")
