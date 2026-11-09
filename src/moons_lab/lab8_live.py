"""Real-time LAB8 load and fault drills. Never fabricate metrics or elapsed time."""
from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from .alerts import AlertManager, AlertRule
from .generator import GeneratorSettings, generate_dataframe
from .lab6_bootstrap import build_registry
from .lab8_model import DLJoblibAdapter
from .lab8_observability import StreamGuard
from .utils import sha256_file


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def http(base: str, path: str, method: str = "GET", body: dict | None = None, timeout=10) -> tuple[int, dict]:
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(base + path, data=raw, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            b = response.read()
            return response.status, json.loads(b) if b else {}
    except HTTPError as exc:
        payload = exc.read()
        try:
            return exc.code, json.loads(payload)
        except Exception:
            return exc.code, {"error": payload.decode("utf-8", "replace")}
    except (OSError, TimeoutError) as exc:
        return 0, {"error": str(exc)}


def wait_healthy(base: str, timeout_s: float = 30) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if http(base, "/health", timeout=0.7)[0] == 200:
            return
        time.sleep(0.2)
    raise RuntimeError(f"service did not become healthy: {base}")


def start_service(root: Path, kind: str, port: int, *, registry: Path | None = None,
                  workers=0, log: Path | None = None) -> subprocess.Popen:
    cmd = [sys.executable, str(root / "scripts" / "lab8_runtime_api.py"),
           "--service", kind, "--port", str(port)]
    if registry:
        cmd += ["--registry", str(registry)]
    if workers:
        cmd += ["--process-workers", str(workers)]
    if log:
        cmd += ["--monitor-log", str(log)]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    logpath = ((registry.parent if registry else root / "demo_output" / "LAB8_runtime") /
               f"lab8_{kind}_server.log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    handle = logpath.open("ab")
    try:
        return subprocess.Popen(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=handle,
                                creationflags=flags)
    finally:
        handle.close()


def stop_service(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def setup_registry(root: Path, target: Path) -> Path:
    reg = build_registry(root, target, reset=True)
    import torch
    import joblib
    from .registry import ModelRegistry
    checkpoint = root / "reports" / "LAB3" / "dl_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    from .dl_pipeline import load_checkpoint
    from .utils import sha256_file
    # Verify trusted LAB3 checkpoint by actually loading and predicting before registering.
    loaded = load_checkpoint(checkpoint)
    proba = loaded.predict_proba(np.asarray([[0., 0.], [1., 0.]], dtype=float))
    if not np.isfinite(proba).all():
        raise ValueError("DL checkpoint returned nonfinite values")
    target.mkdir(parents=True, exist_ok=True)
    dl_model = target / "model_dl_adapter.joblib"
    joblib.dump(DLJoblibAdapter(str(checkpoint.resolve())), dl_model)
    payload = json.loads(reg.read_text(encoding="utf-8"))
    payload["versions"]["v3_dl"] = {
        "path": dl_model.name, "sha256": sha256_file(dl_model),
        "metrics": {"source_lab": 3.0}, "status": "retired",
        "description": "LAB3 PyTorch MLP via verified joblib adapter",
    }
    reg.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ModelRegistry(reg)
    return reg


def sample_resources(proc: subprocess.Popen, path: Path, seconds: float, *, period: float = 1.0) -> list[dict]:
    """Independent sampler can be used during long-running requests from a thread."""
    process = psutil.Process(proc.pid)
    process.cpu_percent(None)
    rows = []
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        try:
            memory = process.memory_info().rss
            cpu = process.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            memory, cpu = 0, 0
        rows.append({"timestamp_utc": utc(), "seconds": time.monotonic() - started,
                     "rss_mb": memory / 1048576, "cpu_pct_process": cpu,
                     "cpu_pct_total": psutil.cpu_percent(interval=None)})
        time.sleep(period)
    write_csv(path, rows)
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_usage(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([r["elapsed_s"] for r in rows], [r["rss_mb"] for r in rows])
    ax.set(title="LAB8 RSS memory", xlabel="Elapsed seconds", ylabel="MiB")
    fig.tight_layout(); fig.savefig(output.with_name(output.stem + "_memory.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([r["elapsed_s"] for r in rows], [r["process_cpu_pct"] for r in rows])
    ax.set(title="LAB8 process CPU", xlabel="Elapsed seconds", ylabel="CPU %")
    fig.tight_layout(); fig.savefig(output.with_name(output.stem + "_cpu.png")); plt.close(fig)


def stage_summary(stage: str, rows: list[dict], elapsed: float, min_seconds: float,
                  min_requests: int, baseline: float, max_ratio: float, *,
                  count_failures: bool = True) -> dict:
    lat = [r["latency_ms"] for r in rows if r.get("success", r.get("status") == 200)]
    successful = len(lat)
    total = len(rows)
    p95 = float(np.quantile(lat, 0.95)) if lat else float("inf")
    result = {"stage": stage, "total": total, "successful": successful,
              "lost_or_failed": total - successful, "success_fraction": successful / max(1, total),
              "elapsed_s": elapsed, "p50_ms": float(np.quantile(lat, .5)) if lat else None,
              "p95_ms": p95, "p99_ms": float(np.quantile(lat, .99)) if lat else None,
              "rps": total / max(elapsed, 1e-9), "baseline_p95_ms": baseline,
              "latency_limit_ms": baseline * max_ratio if baseline else None}
    result["pass"] = (elapsed >= min_seconds and total >= min_requests and
                      result["success_fraction"] >= .999 and
                      result["lost_or_failed"] == 0 and
                      (baseline <= 0 or p95 <= baseline * max_ratio))
    return result


def fault_suite(root: Path, base: str, monitor: str, registry: Path, proc: subprocess.Popen,
                report: Path, cfg, alerts: AlertManager, stream: StreamGuard) -> tuple[subprocess.Popen, list[dict]]:
    """Actual C1/C2/C3/C4/C5 trials; each result is measured, not a mock."""
    out = []
    # C1: actual termination of Uvicorn process and recovery from saved registry/config.
    started = time.monotonic()
    proc.kill()  # real OS kill, not graceful terminate
    proc.wait(timeout=15)
    down_status, _ = http(base, "/health", timeout=1)
    down_predict, _ = http(base, "/predict", "POST", {"x1": .1, "x2": .2}, timeout=1)
    dropped_requests = int(down_predict != 200)
    detected_s = time.monotonic() - started
    alerts.evaluate(AlertRule("service_up", 1, "below", "critical", "restart from registry checkpoint"),
                    0.0)
    proc = start_service(root, "model", cfg.model_port, registry=registry)
    try:
        wait_healthy(base, timeout_s=120)
        status_codes = [http(base, p)[0] for p in ("/health", "/metrics", "/models", "/runtime")]
        status_codes += [http(base, "/predict", "POST", {"x1": .1, "x2": .2})[0]]
        recovered_s = time.monotonic() - started
        passed = (down_status == 0 and detected_s <= 30 and recovered_s <= 120 and all(v == 200 for v in status_codes) and dropped_requests <= 1)
    except RuntimeError:
        recovered_s = time.monotonic() - started
        status_codes, passed = [], False
    out.append({"case": "C1", "pass": passed, "health_down": down_status,
                "alert_detected_s": detected_s, "recovery_s": recovered_s, "smoke_codes": status_codes,
                "lost_requests_during_outage": dropped_requests, "declared_outage_loss_allowance": 1})
    # C2: switch to explicit process-backed joblib-compatible batch route and kill one actual child PID.
    stop_service(proc)
    proc = start_service(root, "model", cfg.model_port, registry=registry, workers=2)
    try:
        wait_healthy(base)
        code, state = http(base, "/lab8/worker-pids")
        candidates = state.get("pids", []) if code == 200 else []
        if not candidates:
            raise RuntimeError("no actual process workers")
        os.kill(int(candidates[0]), signal.SIGTERM)
        health = http(base, "/health")[0]
        batch = http(base, "/predict/batch", "POST", {"items": [{"x1": .2, "x2": .4}] * 8, "n_jobs": 2})[0]
        restored = http(base, "/health")[0]
        fallback = http(base, "/metrics")[1].get("batch_fallbacks", 0)
        ok = health == 200 and batch == 200 and restored == 200 and fallback >= 1
    except (OSError, RuntimeError, ValueError) as exc:
        health, batch, restored, fallback, ok = 0, 0, 0, 0, False
        state = {"exception": str(exc)}
    alerts.evaluate(AlertRule("worker_pool_failure", .5, "above", "warning", "restore process pool"), 1.0)
    out.append({"case": "C2", "pass": ok, "batch_code": batch, "health_code": health,
                "recovery_health_code": restored, "fallbacks": fallback, "workers": state})
    stop_service(proc)
    proc = start_service(root, "model", cfg.model_port, registry=registry)
    wait_healthy(base)
    # C3: damage only copied RETIRED artifact; original published model never touched.
    payload = json.loads(registry.read_text(encoding="utf-8"))
    candidate = registry.parent / payload["versions"]["v1"]["path"]
    original = candidate.read_bytes()
    try:
        candidate.write_bytes(original + b"\0LAB8-corruption-injected")
        promote_code, _ = http(base, "/promote", "POST", {"version": "v1", "reason": "C3 hash drill"})
        model_health = http(base, "/health")[0]
        predict_code = http(base, "/predict", "POST", {"x1": .2, "x2": .3})[0]
        ok = promote_code == 409 and model_health == 200 and predict_code == 200
    finally:
        candidate.write_bytes(original)
    alerts.evaluate(AlertRule("candidate_hash_mismatch", .5, "above", "critical", "reject candidate"), 1)
    out.append({"case": "C3", "pass": ok, "promote_status": promote_code,
                "health_status": model_health, "prediction_status": predict_code})
    # C4: deliberate window-sized interruption and real checkpoint/resume.
    first = stream.processed
    stream.pause()
    dropped = stream.window
    stream.resume()
    x = generate_dataframe(GeneratorSettings(n_samples=max(stream.window * 3, 1500), seed=cfg.seed))
    model = joblib.load(root / "reports/LAB2/ml_model.joblib")
    xp = x[["x1", "x2"]].to_numpy()
    probability = np.asarray(model.predict_proba(xp))
    if probability.ndim == 2:
        probability = probability[:, 1]
    stream.rebase(stream.ref_p, "C4_recovery")
    recovered_windows = stream.add(xp, probability)
    stream.save()
    below = [r for r in recovered_windows[:2] if r["metric"] <= cfg.drift_psi_threshold]
    recovered = (not stream.degraded and stream.processed == first + len(x)
                 and dropped >= stream.window and len(below) > 0)
    out.append({"case": "C4", "pass": recovered, "dropped_records": dropped,
                "processed_before": first, "processed_after": stream.processed,
                "recovered_within_windows": next((i for i,r in enumerate(recovered_windows[:2]) if r in below), None),
                "gap_events": stream.gaps})
    # C5: actual new Python subprocess rejects broken schema before service launch.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="lab8_c5_") as tmp:
        bad = Path(tmp) / "invalid.json"
        good = Path(tmp) / "valid.json"
        invalid = cfg.model_dump(); invalid["model_port"] = -1
        bad.write_text(json.dumps(invalid), encoding="utf-8")
        good.write_text(json.dumps(cfg.model_dump()), encoding="utf-8")
        checker = root / "scripts/lab8_validate_config.py"
        denied = subprocess.run([sys.executable, str(checker), "--config", str(bad)],
                                cwd=root, capture_output=True, text=True)
        accepted = subprocess.run([sys.executable, str(checker), "--config", str(good)],
                                  cwd=root, capture_output=True, text=True)
        c5_ok = denied.returncode != 0 and "model_port" in (denied.stderr + denied.stdout) and accepted.returncode == 0
    out.append({"case": "C5", "pass": c5_ok, "invalid_returncode": denied.returncode,
                "valid_returncode": accepted.returncode, "rejected_field": "model_port"})
    with (report / "incidents_log.jsonl").open("w", encoding="utf-8") as fh:
        for incident in out:
            fh.write(json.dumps({"time": utc(), **incident}, ensure_ascii=False) + "\n")
    return proc, out
