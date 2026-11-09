"""LAB8 negative and deterministic unit tests; full wall-clock proof is separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from moons_lab.alerts import AlertManager, AlertRule
from moons_lab.lab8_config import Lab8FullConfig, load_lab8_config
from moons_lab.lab8_live import stage_summary
from moons_lab.lab8_model import DLJoblibAdapter
from moons_lab.lab8_observability import StreamGuard, hash_manifest


def cfg():
    return Lab8FullConfig(
        seed=42, drift_psi_threshold=.2, validator_reject_rate_threshold=.15,
        service_error_rate_threshold=.01, alert_cooldown_seconds=60,
    )


@pytest.mark.parametrize("updates", [
    {"model_port": -1}, {"model_port": 18780, "monitor_port": 18780},
    {"stage_seconds": 20}, {"min_stage_requests": 9999},
    {"sample_interval_s": 2.1}, {"window_step": 2000, "window_size": 500},
    {"drift_shift_x": float("inf")},
])
def test_bad_configuration_rejected(updates):
    with pytest.raises((ValueError, ValidationError)):
        Lab8FullConfig.model_validate({**cfg().model_dump(), **updates})


def test_default_config_is_valid():
    f = Path(__file__).resolve().parents[1] / "configs/lab8_full.json"
    value = load_lab8_config(f)
    assert value.min_stage_requests == 10_000 and value.stage_seconds == 600
    assert value.warmup_seconds >= 300 and value.window_step <= value.window_size


def test_alerts_and_dedup(tmp_path):
    manager = AlertManager(tmp_path / "events.jsonl", cooldown_seconds=60)
    rule = AlertRule("drift", .2, "above", "warning", "inspect")
    assert manager.evaluate(rule, 0.0, now=1000) is None
    assert manager.evaluate(rule, 0.9, now=1000) is not None
    assert manager.evaluate(rule, 1.0, now=1001) is None
    assert manager.evaluate(rule, 1.0, now=1061) is not None
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and set(rows[0]) == {"time", "metric", "value", "threshold", "level", "action"}


def test_stream_checkpoint_recovery_and_drift(tmp_path):
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(1000, 2))
    prob = np.clip(.5 + ref[:, 0] / 10, .01, .99)
    guard = StreamGuard(ref, prob, window=200, step=100, threshold=.2,
                        alert_path=tmp_path / "alerts.jsonl", checkpoint=tmp_path / "stream.json")
    own = guard.add(ref[:500], prob[:500])
    assert len(own) == 4
    previous = guard.processed
    guard.pause()
    assert guard.degraded and (tmp_path / "stream.json").exists()
    guard.resume()
    assert guard.processed == previous and not guard.degraded
    far = ref[:400] + np.asarray([4., 0.])
    rows = guard.add(far, prob[:400])
    assert rows and any(e["metric"] > .2 for e in rows)
    guard.export_csv(tmp_path / "metrics.csv")
    assert (tmp_path / "metrics.csv").exists()


def test_stage_acceptance_denies_fake_short_run():
    records = [{"status": 200, "success": True, "latency_ms": 4} for _ in range(100)]
    result = stage_summary("short", records, elapsed=10, min_seconds=600,
                           min_requests=10_000, baseline=10, max_ratio=1.5)
    assert not result["pass"]
    fake_content = [{"status": 200, "success": False, "latency_ms": 4} for _ in range(10_000)]
    result = stage_summary("bad", fake_content, elapsed=600, min_seconds=600,
                           min_requests=10_000, baseline=10, max_ratio=1.5)
    assert result["success_fraction"] == 0 and not result["pass"]


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_manifest_digest_and_ignored_cache(tmp_path, line_ending):
    (tmp_path / "src").mkdir()
    p = tmp_path / "src" / "demo.py"
    p.write_bytes(b"a=1" + line_ending)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "secret.py").write_text("secret", encoding="utf-8")
    target = tmp_path / "manifest.json"
    result = hash_manifest(tmp_path, target)
    assert result["count"] == 1
    assert result["items"][0]["sha256"] == hashlib.sha256(b"a=1" + line_ending).hexdigest()


def test_adapter_real_checkpoint():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "reports/LAB3/dl_model.pt"
    if not checkpoint.exists():
        pytest.skip("published LAB3 checkpoint unavailable")
    obj = DLJoblibAdapter(str(checkpoint))
    p = obj.predict_proba(np.asarray([[.1, .2], [1, -1]]))
    assert len(p) == 2 and np.all((p >= 0) & (p <= 1))


def test_real_uvicorn_and_five_fault_drills(tmp_path):
    """Real OS processes, including one actually terminated batch worker."""
    import socket
    import subprocess
    import sys
    from moons_lab.generator import GeneratorSettings, generate_dataframe
    from moons_lab.lab8_live import (setup_registry, start_service, stop_service,
                                      wait_healthy, http, fault_suite)
    root = Path(__file__).resolve().parents[1]

    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    model_port = free_port()
    monitor_port = free_port()
    while monitor_port == model_port:
        monitor_port = free_port()
    payload = {**cfg().model_dump(), "model_port": model_port, "monitor_port": monitor_port}
    settings = Lab8FullConfig.model_validate(payload)
    registry = setup_registry(root, tmp_path / "registry")
    df = generate_dataframe(GeneratorSettings(n_samples=1100, seed=42))
    x = df[["x1", "x2"]].to_numpy()
    model = joblib.load(root / "reports/LAB2/ml_model.joblib")
    p = np.asarray(model.predict_proba(x))
    p = p[:, 1] if p.ndim == 2 else p
    guard = StreamGuard(x, p, window=200, step=100,
                        threshold=.2, alert_path=tmp_path / "alerts.jsonl",
                        checkpoint=tmp_path / "checkpoint.json")
    manager = AlertManager(tmp_path / "alerts.jsonl")
    base = f"http://127.0.0.1:{model_port}"
    monitor = f"http://127.0.0.1:{monitor_port}"
    mproc = start_service(root, "monitor", monitor_port)
    proc = None
    try:
        proc = start_service(root, "model", model_port, registry=registry)
        wait_healthy(base, timeout_s=30)
        wait_healthy(monitor, timeout_s=30)
        proc, faults = fault_suite(root, base, monitor, registry, proc,
                                  tmp_path, settings, manager, guard)
        assert [x["case"] for x in faults] == ["C1", "C2", "C3", "C4", "C5"]
        assert all(x["pass"] for x in faults), faults
        assert (tmp_path / "incidents_log.jsonl").is_file()
    finally:
        stop_service(proc)
        stop_service(mproc)


def test_validated_route_rejects_out_of_distribution_without_prediction(tmp_path):
    import sys
    from fastapi.testclient import TestClient
    from moons_lab.lab8_live import setup_registry
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from lab8_runtime_api import make_app
    registry = setup_registry(root, tmp_path / "registry")
    client = TestClient(make_app(registry))
    result = client.post("/lab8/predict/validated", json={
        "items": [{"x1": 12.0, "x2": 12.0}, {"x1": 11.0, "x2": 11.0}],
        "n_jobs": 1,
    })
    assert result.status_code == 200
    data = result.json()
    assert data["silent_accepts"] == 0
    assert data["rejected_count"] == 2 and data["accepted_count"] == 0
    assert all(v["rejected"] and "prediction" not in v and v["reason"] for v in data["results"])
    assert len(data["monitor_probabilities"]) == 2


def test_fast_dl_adapter_matches_published_checkpoint_and_parallel_threads():
    """Only LAB8 serving is optimized; published LAB3 math stays identical."""
    from concurrent.futures import ThreadPoolExecutor
    from moons_lab.dl_pipeline import load_checkpoint
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "reports/LAB3/dl_model.pt"
    rng = np.random.default_rng(2048)
    samples = rng.normal(size=(128, 2))
    expected = load_checkpoint(checkpoint).predict_proba(samples)
    adapter = DLJoblibAdapter(str(checkpoint))
    adapter.warmup()
    actual = adapter.predict_proba(samples)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    with ThreadPoolExecutor(max_workers=4) as executor:
        chunks = list(executor.map(adapter.predict_proba, np.array_split(samples, 4)))
    np.testing.assert_allclose(np.concatenate(chunks), expected, rtol=1e-6, atol=1e-7)


def test_lab8_persistent_parallel_pool_uses_real_threads_and_preserves_versions(tmp_path):
    import os
    import sys
    from fastapi.testclient import TestClient
    from moons_lab.lab8_live import setup_registry
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from lab8_runtime_api import make_app
    registry = setup_registry(root, tmp_path / "registry")
    jobs = min(4, max(1, os.cpu_count() or 1))
    with TestClient(make_app(registry)) as client:
        items = [{"x1": float(i) / 10, "x2": float(i % 7) / 10}
                 for i in range(max(16, jobs * 8))]
        # Original LAB6 ML route still works without the LAB8 adapter.
        first = client.post("/predict/batch", json={"items": items, "n_jobs": 1})
        assert first.status_code == 200
        assert {v["model_version"] for v in first.json()["results"]} == {"v2"}
        promoted = client.post("/promote", json={"version": "v3_dl", "reason": "LAB8 parallel proof"})
        assert promoted.status_code == 200
        warmed = client.post("/lab8/warmup")
        assert warmed.status_code == 200 and warmed.json()["version"] == "v3_dl"
        proof = client.get("/lab8/parallel-proof", params={"workers": jobs})
        assert proof.status_code == 200 and proof.json()["simultaneous_threads"] == jobs
        response = client.post("/predict/batch", json={"items": items, "n_jobs": jobs})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["count"] == len(items) and data["n_jobs"] == jobs
        assert len(data["results"]) == len(items)
        assert {v["model_version"] for v in data["results"]} == {"v3_dl"}
        x = np.asarray([[item["x1"], item["x2"]] for item in items])
        expected = DLJoblibAdapter(str(root / "reports/LAB3/dl_model.pt")).predict_proba(x)
        np.testing.assert_allclose([v["probability"] for v in data["results"]], expected,
                                   rtol=1e-6, atol=1e-7)
        metrics = client.get("/lab8/parallel-stats").json()
        assert metrics["parallel_batches"] >= 1
        assert metrics["last_requested_workers"] == jobs
        assert metrics["max_workers_observed"] >= min(2, jobs)
        rollback = client.post("/rollback")
        assert rollback.status_code == 200 and rollback.json()["active"] == "v2"
        recovered = client.post("/predict/batch", json={"items": items, "n_jobs": 1})
        assert recovered.status_code == 200
        assert {v["model_version"] for v in recovered.json()["results"]} == {"v2"}


def test_lab8_api_lifespan_without_legacy_event_handler(tmp_path, monkeypatch):
    """Starlette 1.x removed FastAPI.add_event_handler; cleanup uses lifespan."""
    import sys
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from moons_lab.lab8_live import setup_registry

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from lab8_runtime_api import make_app

    monkeypatch.setattr(FastAPI, "add_event_handler", None, raising=False)
    registry = setup_registry(root, tmp_path / "registry")
    app = make_app(registry)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/lab8/parallel-stats")
        assert response.status_code == 200
        assert response.json()["backend"] == "persistent_thread_pool"


def test_lab8_keepalive_reuses_connection_and_no_remote_hosts():
    """Real loopback HTTP/1.1, verify three POSTs use the same TCP connection."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from moons_lab.lab8_live import KeepAliveHTTP

    remote_ports = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            remote_ports.append(self.client_address[1])
            raw = json.dumps({"count": len(payload["items"]), "results": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with KeepAliveHTTP(base, timeout=4) as client:
            for _ in range(3):
                status, response = client.request(
                    "/predict/batch", "POST", {"items": [{"x1": .2, "x2": .3}], "n_jobs": 1}
                )
                assert status == 200 and response["count"] == 1
        assert len(remote_ports) == 3 and len(set(remote_ports)) == 1
        with pytest.raises(ValueError, match="local"):
            KeepAliveHTTP("http://example.com:8080")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=4)


def test_lab8_drift_latches_one_active_alert_per_phase(tmp_path):
    rng = np.random.default_rng(172)
    x = rng.normal(size=(1000, 2))
    proba = np.full(1000, 0.4)
    guard = StreamGuard(x, proba, window=100, step=50, threshold=.2,
                        alert_path=tmp_path / "alerts.jsonl",
                        checkpoint=tmp_path / "state.json", cooldown=0)
    guard.rebase(proba, "drift")
    far = rng.normal(size=(2000, 2)) + np.array([7, 0])
    events = guard.add(far, np.full(len(far), .4))
    assert sum(e["alert"] for e in events) == 1
    guard.save()
    other = StreamGuard(x, proba, window=100, step=50, threshold=.2,
                        alert_path=tmp_path / "alerts.jsonl",
                        checkpoint=tmp_path / "state.json", cooldown=0)
    other.restore()
    assert other.phase_alert_latched
    assert not any(e["alert"] for e in other.add(far, np.full(len(far), .4)))
    other.rebase(proba, "new_drift")
    assert sum(e["alert"] for e in other.add(far, np.full(len(far), .4))) == 1
    alerts = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert len(alerts) == 2


def test_full_manifest_counts_each_report_file_once(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "source.py").write_bytes(b"print(1)\n")
    (tmp_path / ".gitattributes").write_bytes(b"reports/LAB7/*.html linguist-generated\n")
    folder = tmp_path / "reports" / "LAB8"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_bytes(b"{}")
    # Editable pip installations and runtime generation leave ignored paths on disk:
    # none may leak into the immutable GitHub evidence manifest.
    pkginfo = tmp_path / "src" / "make_moons.egg-info"
    pkginfo.mkdir()
    (pkginfo / "PKG-INFO").write_bytes(b"volatile pip metadata")
    generated = tmp_path / "data" / "runtime_synthetic"
    generated.mkdir(parents=True)
    (generated / "chunk.csv").write_bytes(b"runtime data")
    pytest_cache = tmp_path / "reports" / "LAB4" / ".pytest_runtime"
    pytest_cache.mkdir(parents=True)
    (pytest_cache / "garbage.json").write_bytes(b"ignored")
    manifest = hash_manifest(tmp_path, folder / "artifacts_manifest.json")
    assert {v["path"] for v in manifest["items"]} == {"src/source.py", ".gitattributes"}
    assert {v["path"] for v in manifest["runtime_evidence"]} == {"reports/LAB8/summary.json"}
    assert manifest["count"] == 3


def test_final_evidence_verifier_refuses_smoke_and_tampered_manifest(tmp_path):
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from lab8_verify_full import EvidenceError, verify_full, verify_manifest
    r = tmp_path / "reports" / "LAB8"
    r.mkdir(parents=True)
    (r / "summary.json").write_text(json.dumps({"evidence_mode": "SMOKE_ONLY", "all_pass": True}), encoding="utf-8")
    with pytest.raises(EvidenceError, match="SMOKE_ONLY"):
        verify_full(tmp_path, r)
    files = tmp_path / "src"
    files.mkdir()
    artifacts = []
    for idx in range(21):
        path = files / f"file_{idx}.py"
        path.write_bytes(f"assert {idx} == {idx}\n".encode())
        artifacts.append({"path": path.relative_to(tmp_path).as_posix(),
                          "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    attrs = tmp_path / ".gitattributes"
    attrs.write_bytes(b"reports/LAB7/*.html linguist-generated\n")
    artifacts.append({"path": ".gitattributes", "bytes": attrs.stat().st_size,
                      "sha256": hashlib.sha256(attrs.read_bytes()).hexdigest()})
    runtime = [{"path": "reports/LAB8/summary.json", "bytes": (r / "summary.json").stat().st_size,
                "sha256": hashlib.sha256((r / "summary.json").read_bytes()).hexdigest()}]
    (r / "artifacts_manifest.json").write_text(json.dumps({"items": artifacts, "runtime_evidence": runtime,
                                                             "count": len(artifacts) + len(runtime)}))
    assert verify_manifest(tmp_path, r) == len(artifacts) + len(runtime)
    (files / "file_3.py").write_text("CHANGED", encoding="utf-8")
    with pytest.raises(EvidenceError, match="size mismatch|SHA mismatch"):
        verify_manifest(tmp_path, r)
