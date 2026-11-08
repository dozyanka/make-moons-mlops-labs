"""Real localhost/Uvicorn hot-swap drill with an independent monitoring process.

No TestClient is used: requests travel over HTTP to two subprocesses.  Both
processes and their model/monitor state live in a dedicated runtime directory.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(port: int, method: str, path: str, data: dict | None = None,
              timeout: float = 8.0) -> tuple[int, dict]:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    request = Request(f"http://127.0.0.1:{port}{path}", data=payload, method=method,
                      headers={"Content-Type": "application/json"} if payload else {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def wait_for(port: int, proc: subprocess.Popen, *, timeout_s: float = 25.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Uvicorn on port {port} stopped unexpectedly: {proc.returncode}")
        try:
            if http_json(port, "GET", "/health", timeout=.5)[0] == 200:
                return
        except (OSError, URLError):
            time.sleep(.1)
    raise TimeoutError(f"Uvicorn not ready on port {port}")


def drill(runtime_dir: Path, *, requests: int = 480) -> dict:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    monitor_port, model_port = free_port(), free_port()
    while model_port == monitor_port:
        model_port = free_port()
    logfiles = []
    processes = []

    def start_process(service: str, port: int, extra: list[str] | None = None) -> subprocess.Popen:
        log = (runtime_dir / f"{service}_uvicorn.log").open("w", encoding="utf-8")
        logfiles.append(log)
        command = [sys.executable, str(ROOT / "scripts/serve_lab6.py"),
                   "--service", service, "--port", str(port),
                   "--runtime-dir", str(runtime_dir)] + (extra or [])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    try:
        monitor_process = start_process("monitor", monitor_port)
        wait_for(monitor_port, monitor_process)
        model_process = start_process("model", model_port, [
            "--monitor-url", f"http://127.0.0.1:{monitor_port}", "--reset-registry",
        ])
        wait_for(model_port, model_process)
        assert http_json(model_port, "GET", "/models")[1]["active"] == "v2"

        # Prove that the two independent applications communicate via HTTP.
        before = http_json(monitor_port, "GET", "/metrics")[1]["count"]
        assert http_json(model_port, "POST", "/predict", {"x1": .2, "x2": .1})[0] == 200
        after = http_json(monitor_port, "GET", "/metrics")[1]["count"]
        if after <= before:
            raise AssertionError("model did not send HTTP observation to independent monitor")

        failed = []
        versions = []
        latencies = []
        results_lock = threading.Lock()
        begin = threading.Event()
        workers = 6
        per_worker = max(20, (requests + workers - 1) // workers)

        def worker(worker_id: int) -> None:
            begin.wait()
            for i in range(per_worker):
                t0 = time.perf_counter()
                try:
                    status, response = http_json(model_port, "POST", "/predict", {
                        "x1": .12 + (i % 25) * .001, "x2": .18,
                    }, timeout=8)
                    with results_lock:
                        if status != 200:
                            failed.append(f"HTTP {status}: {response}")
                        else:
                            versions.append(response["model_version"])
                            latencies.append(1000 * (time.perf_counter() - t0))
                except Exception as exc:
                    with results_lock:
                        failed.append(f"{type(exc).__name__}: {exc}")
                time.sleep(.008)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(worker, i) for i in range(workers)]
            begin.set()
            time.sleep(.18)
            promote_status, _ = http_json(model_port, "POST", "/promote", {
                "version": "v1", "reason": "real HTTP network drill",
            })
            time.sleep(.35)
            rollback_status, _ = http_json(model_port, "POST", "/rollback")
            for job in jobs:
                job.result()

        active = http_json(model_port, "GET", "/models")[1]["active"]
        service_metrics = http_json(model_port, "GET", "/metrics")[1]
        monitor_metrics = http_json(monitor_port, "GET", "/metrics")[1]
        expected = workers * per_worker
        result = {
            "mode": "real HTTP on two separate Uvicorn processes",
            "expected_requests": expected,
            "successful_requests": len(versions),
            "lost_or_failed": len(failed),
            "failures_sample": failed[:8],
            "observed_versions": sorted(set(versions)),
            "promote_http_status": promote_status,
            "rollback_http_status": rollback_status,
            "active_after_rollback": active,
            "monitor_points_before": before,
            "monitor_points_after_one_prediction": after,
            "monitor_service_requests": monitor_metrics["service_request_count"],
            "monitor_service_errors": monitor_metrics["service_error_count"],
            "model_request_count": service_metrics["request_count"],
            "max_request_latency_ms": max(latencies) if latencies else None,
        }
        if (len(versions) != expected or failed or set(versions) != {"v1", "v2"}
                or promote_status != 200 or rollback_status != 200 or active != "v2"
                or monitor_metrics["service_request_count"] <= 0):
            raise AssertionError("real network hot-swap failed: " + json.dumps(result))
        return result
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for logfile in logfiles:
            logfile.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "demo_output/LAB6_network_runtime")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "demo_output/LAB6_network")
    parser.add_argument("--requests", type=int, default=480)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    result = drill(args.runtime_dir.resolve(), requests=args.requests)
    output = args.report_dir / "network_drill.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("LAB6 NETWORK PASSED:", output)


if __name__ == "__main__":
    main()
