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
