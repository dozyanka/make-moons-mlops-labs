"""Comparable LAB6 vs optimized LAB8 p95: actual HTTP, same request shape/RPS.

Reference is the UNMODIFIED LAB6 FastAPI route serving its published ML v2;
candidate is LAB8 FastAPI route serving the immutable LAB3 DL checkpoint.
The two model families differ, as they did in the course assignment.
Published LAB6 TestClient p95 is ALSO shown; no acceptance threshold changes.
Outputs are diagnostics, not full 600s/10000-request evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moons_lab.lab8_live import KeepAliveHTTP, http, setup_registry, stop_service, utc, wait_healthy


def spawn(port: int, registry: Path, *, unmodified: bool) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / "scripts" / "lab8_runtime_api.py"),
           "--service", "model", "--registry", str(registry), "--port", str(port)]
    if unmodified:
        cmd.append("--unmodified-lab6")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    # Logging to file prevents silent worker failures and avoids mixing with the report.
    log = (registry.parent / ("lab6_reference.log" if unmodified else "lab8_candidate.log"))
    with log.open("ab") as fh:
        return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=fh,
                                creationflags=flags)


def measure(base: str, *, label: str, jobs: int, size: int, samples: int,
            warmup: int, target_rps: float, version: str) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(909)
    x = rng.normal(size=(size, 2))
    payload = {"items": [{"x1": float(a), "x2": float(b)} for a,b in x], "n_jobs": jobs}
    rows = []
    with KeepAliveHTTP(base, timeout=30) as client:
        for _ in range(warmup):
            status, result = client.request("/predict/batch", "POST", payload)
            if status != 200 or result.get("count") != size:
                raise RuntimeError(f"warm-up {label}: status={status}, data={result}")
        next_at = time.monotonic()
        for i in range(samples):
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            start = time.perf_counter()
            status, data = client.request("/predict/batch", "POST", payload)
            elapsed = (time.perf_counter() - start) * 1000
            next_at += 1.0 / target_rps
            correct = status == 200 and data.get("count") == size and all(
                r.get("model_version") == version for r in data.get("results", []))
            rows.append({"label": label, "timestamp_utc": utc(), "request": i,
                         "latency_ms": elapsed, "status": status, "correct": correct,
                         "n_jobs": jobs, "batch_size": size})
            if not correct:
                raise RuntimeError(f"HTTP failed in {label}: {status} {data}")
    latency = np.asarray([r["latency_ms"] for r in rows])
    return {"count": len(rows), "min_ms": float(np.min(latency)),
            "p50_ms": float(np.quantile(latency, .5)),
            "p95_ms": float(np.quantile(latency, .95)),
            "p99_ms": float(np.quantile(latency, .99))}, rows


def compare(threads: int, samples: int, warmup: int, target_rps: float,
            reference_port: int, candidate_port: int, output: Path) -> dict:
    if reference_port == candidate_port:
        raise ValueError("reference and candidate must use different ports")
    # Reject occupied ports explicitly; otherwise a pre-existing local service
    # could silently contaminate the reference measurements.
    for port in (reference_port, candidate_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
            try:
                test_socket.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"reference/candidate port {port} is occupied") from exc
    output.mkdir(parents=True, exist_ok=True)
    allrows = []
    result = {"mode": "DIAGNOSTIC_ONLY", "http_transport": "persistent_http_1_1_loopback", "timestamp_utc": utc(),
              "samples_per_case": samples, "warmup_requests": warmup,
              "target_rps": target_rps, "logical_cores_requested": threads,
              "published_lab6_p95_ms": None, "cases": []}
    reference = json.loads((ROOT / "reports/LAB6/service_metrics.json").read_text(encoding="utf-8"))
    original_p95 = float(reference["p95_request_latency_ms"])
    result["published_lab6_p95_ms"] = original_p95
    with tempfile.TemporaryDirectory(prefix="lab8_http_compare_", ignore_cleanup_errors=True) as dirname:
        regdir = Path(dirname) / "registry"
        registry = setup_registry(ROOT, regdir)
        reference_proc = None
        candidate_proc = None
        try:
            reference_proc = spawn(reference_port, registry, unmodified=True)
            candidate_proc = spawn(candidate_port, registry, unmodified=False)
            refbase = f"http://127.0.0.1:{reference_port}"
            candidatebase = f"http://127.0.0.1:{candidate_port}"
            wait_healthy(refbase, timeout_s=45)
            wait_healthy(candidatebase, timeout_s=45)
            status, _ = http(candidatebase, "/promote", "POST", {"version": "v3_dl", "reason": "LAB8 matched HTTP benchmark"})
            if status != 200:
                raise RuntimeError(f"cannot promote DL candidate: {status}")
            status, warm = http(candidatebase, "/lab8/warmup", "POST")
            if status != 200 or warm.get("version") != "v3_dl":
                raise RuntimeError(f"DL warmup failed: {status} {warm}")
            for name, count, batch_size in (("sequential", 1, 8),
                                             ("parallel", threads, max(8, threads))):
                baseline, rows = measure(refbase, label=f"lab6_{name}", jobs=count, size=batch_size,
                                         samples=samples, warmup=warmup, target_rps=target_rps,
                                         version="v2")
                allrows.extend(rows)
                candidate, rows = measure(candidatebase, label=f"lab8_{name}", jobs=count, size=batch_size,
                                          samples=samples, warmup=warmup, target_rps=target_rps,
                                          version="v3_dl")
                allrows.extend(rows)
                result["cases"].append({"name": name, "jobs": count, "batch_size": batch_size,
                                        "lab6_original_http": baseline,
                                        "lab8_optimized_dl_http": candidate,
                                        "matched_p95_ratio": candidate["p95_ms"] / max(baseline["p95_ms"], 1e-9),
                                        "published_lab6_p95_ratio": candidate["p95_ms"] / original_p95})
            status, stats = http(candidatebase, "/lab8/parallel-stats")
            result["parallel_stats"] = stats if status == 200 else {"error": status}
        finally:
            stop_service(candidate_proc)
            stop_service(reference_proc)
    with (output / "matched_http_latency.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=allrows[0].keys())
        writer.writeheader()
        writer.writerows(allrows)
    (output / "matched_http_latency.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rps", type=float, default=18.0)
    parser.add_argument("--reference-port", type=int, default=18784)
    parser.add_argument("--candidate-port", type=int, default=18785)
    parser.add_argument("--output", type=Path, default=Path("demo_output/LAB8_comparison"))
    args = parser.parse_args()
    if args.samples < 30 or args.warmup < 1 or args.rps <= 0:
        parser.error("use at least 30 measurements and 1 warmup, positive RPS")
    cores = max(1, os.cpu_count() or 1)
    report = compare(cores, args.samples, args.warmup, args.rps,
                     args.reference_port, args.candidate_port, args.output.resolve())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
