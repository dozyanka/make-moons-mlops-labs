"""LAB6 fixes: persistent registry, model statuses, typed schemas and HTTP metrics."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from moons_lab.lab6_bootstrap import build_registry
from moons_lab.monitor_service import MonitorState, Observation, ServiceObservation, create_monitor_app
from moons_lab.registry import ModelRegistry
from moons_lab.service import create_app
from moons_lab.utils import sha256_file


@pytest.fixture
def fixture_project(tmp_path: Path):
    """Small real joblib models for self-contained tests (no published assets modified)."""
    root = tmp_path / "project"
    reports = root / "reports" / "LAB2"
    reports.mkdir(parents=True)
    x = np.array([[-1., 0.], [-.7, -.2], [.7, .2], [1., 0.], [-.5, .4], [.5, -.4]])
    y = np.array([0, 0, 1, 1, 0, 1])
    for c, path in [(1., reports / "linear_baseline.joblib"),
                    (2., reports / "ml_model.joblib")]:
        joblib.dump(LogisticRegression(C=c).fit(x, y), path)
    (reports / "comparison.json").write_text(json.dumps({
        "linear_baseline_test": {"accuracy": .8}, "full_test": {"accuracy": .95},
    }), encoding="utf-8")
    registry_dir = tmp_path / "registry"
    registry_path = build_registry(root, registry_dir)
    return root, registry_path


def test_bootstrap_preserves_promotion_and_history(fixture_project):
    root, path = fixture_project
    registry = ModelRegistry(path)
    registry.promote("v1", "test persistence")
    before = registry.snapshot()
    assert build_registry(root, path.parent) == path
    after = ModelRegistry(path).snapshot()
    assert after == before
    assert after["active"] == "v1"
    # Deliberate reset is possible only in a reproducibility/test context.
    build_registry(root, path.parent, reset=True)
    assert ModelRegistry(path).active_version == "v2"


def test_bootstrap_rejects_modified_candidate_instead_of_repairing(fixture_project):
    root, path = fixture_project
    model = path.parent / "model_v1.joblib"
    with model.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_registry(root, path.parent)
    assert sha256_file(model) != sha256_file(root / "reports/LAB2/linear_baseline.joblib")


def test_stage_lifecycle_and_transactional_reject(fixture_project):
    _, path = fixture_project
    reg = ModelRegistry(path)
    assert reg.stage("v1", "preparing")["versions"]["v1"]["status"] == "preparing"
    with pytest.raises(ValueError, match="canary-tested"):
        reg.promote("v1")
    assert ModelRegistry(path).active_version == "v2"
    assert reg.stage("v1", "canary")["versions"]["v1"]["status"] == "canary"
    reg.promote("v1")
    assert ModelRegistry(path).active_version == "v1"
    with pytest.raises(ValueError, match="active production"):
        reg.stage("v1", "retired")
    reg.rollback()
    snap = ModelRegistry(path).snapshot()
    assert snap["active"] == "v2"
    assert snap["versions"]["v1"]["status"] == "retired"


def test_http_422_counted_and_metrics_toggle(fixture_project):
    _, path = fixture_project
    with TestClient(create_app(path)) as client:
        assert client.post("/predict", json={"x1": "bad", "x2": 0.2}).status_code == 422
        first = client.get("/metrics").json()
        assert first["request_count"] >= 1
        assert first["error_count"] == 1
        assert first["error_rate"] > 0
        assert client.patch("/runtime", json={"metrics_enabled": False}).status_code == 200
        before = client.get("/metrics").json()
        for _ in range(3):
            assert client.post("/predict", json={"x1": .2, "x2": .2}).status_code == 200
            assert client.post("/predict", json={"x1": "wrong", "x2": .2}).status_code == 422
        after = client.get("/metrics").json()
        assert after["request_count"] == before["request_count"]
        assert after["prediction_count"] == before["prediction_count"]
        assert after["error_count"] == before["error_count"]
        client.patch("/runtime", json={"metrics_enabled": True})
        assert client.post("/predict", json={"x1": .2, "x2": .2}).status_code == 200
        assert client.get("/metrics").json()["prediction_count"] > after["prediction_count"]


def test_openapi_response_models_and_new_status_endpoint(fixture_project):
    _, path = fixture_project
    with TestClient(create_app(path)) as client:
        spec = client.get("/openapi.json").json()
        for endpoint in ["/health", "/predict", "/predict/batch", "/metrics", "/models", "/runtime"]:
            assert endpoint in spec["paths"]
            verb = "post" if "predict" in endpoint else "get"
            schema = spec["paths"][endpoint][verb]["responses"]["200"]["content"]["application/json"]["schema"]
            assert "$ref" in schema, (endpoint, schema)
        response = client.patch("/models/v1/status", json={"status": "preparing"})
        assert response.status_code == 200
        assert client.post("/promote", json={"version": "v1"}).status_code == 409
        assert client.patch("/models/v1/status", json={"status": "canary"}).status_code == 200
        assert client.post("/promote", json={"version": "v1"}).status_code == 200


def test_monitor_receives_http_events_and_responses_typed(fixture_project, tmp_path):
    _, path = fixture_project
    monitor = MonitorState(log_path=tmp_path / "observations.jsonl")
    monitor_app = create_monitor_app(monitor)
    with TestClient(monitor_app) as monitor_client:
        def prediction_sink(a, b, p):
            response = monitor_client.post("/ingest", json={"x1": a, "x2": b, "probability": p})
            assert response.status_code == 200

        def http_sink(endpoint, status, latency, version, uptime):
            response = monitor_client.post("/ingest/service", json={"path": endpoint,
                "status_code": status, "latency_ms": latency,
                "model_version": version, "model_uptime_s": uptime})
            assert response.status_code == 200

        with TestClient(create_app(path, monitor_sink=prediction_sink,
                                   monitor_request_sink=http_sink)) as client:
            assert client.post("/predict", json={"x1": .2, "x2": .1}).status_code == 200
            assert client.post("/predict", json={"x1": "invalid", "x2": .1}).status_code == 422
        metrics = monitor_client.get("/metrics").json()
        assert metrics["count"] == 1
        assert metrics["service_request_count"] == 2
        assert metrics["service_error_count"] == 1
        assert metrics["model_version"] == "v2"
        logs = (tmp_path / "observations.jsonl").read_text(encoding="utf-8")
        assert '"kind": "prediction"' in logs and '"kind": "http"' in logs
        assert monitor_client.get("/plot.png").headers["content-type"] == "image/png"


def test_concurrent_promotion_no_failures(fixture_project):
    _, path = fixture_project
    with TestClient(create_app(path)) as client:
        def predict(_):
            response = client.post("/predict", json={"x1": .2, "x2": .2})
            return response.status_code
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(predict, index) for index in range(160)]
            assert client.post("/promote", json={"version": "v1"}).status_code == 200
            assert client.post("/rollback").status_code == 200
            statuses = [future.result() for future in futures]
        assert all(code == 200 for code in statuses)
        assert client.get("/health").json()["active_version"] == "v2"
