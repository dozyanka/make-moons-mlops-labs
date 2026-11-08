"""LAB6 registry, contracts, monitoring, concurrent hot-swap negative controls."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from moons_lab.lab6_bootstrap import build_registry
from moons_lab.monitor_service import MonitorState, Observation, create_monitor_app
from moons_lab.registry import ModelRegistry
from moons_lab.service import create_app
from moons_lab.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(tmp_path: Path):
    reg = build_registry(ROOT, tmp_path / "registry")
    state = MonitorState(log_path=tmp_path / "monitor.jsonl")
    model = TestClient(create_app(reg, monitor_sink=lambda a, b, p: state.add(Observation(x1=a, x2=b, probability=p))))
    monitor = TestClient(create_monitor_app(state))
    return reg, model, monitor, state


def test_two_real_versions_and_original_hashes(runtime):
    registry_path, client, _, _ = runtime
    payload = client.get("/models").json()
    assert set(payload["versions"]) == {"v1", "v2"}
    assert payload["active"] == "v2"
    assert payload["versions"]["v1"]["sha256"] == sha256_file(ROOT / "reports/LAB2/linear_baseline.joblib")
    assert payload["versions"]["v2"]["sha256"] == sha256_file(ROOT / "reports/LAB2/ml_model.joblib")
    assert client.get("/health").json()["active_version"] == "v2"


@pytest.mark.parametrize("payload", [
    {"x1": .2}, {"x1": "bad", "x2": .2}, {"x1": "0.2", "x2": .2},
    {"x1": 999, "x2": 0}, {"x1": .2, "x2": False},
    {"x1": .2, "x2": .2, "extra": 1},
])
def test_422_for_invalid_input(runtime, payload):
    assert runtime[1].post("/predict", json=payload).status_code == 422


def test_promote_and_rollback_and_monitor(runtime):
    _, client, monitor, state = runtime
    assert client.post("/predict", json={"x1": 0.2, "x2": 0.4}).json()["model_version"] == "v2"
    assert client.post("/promote", json={"version": "v1", "reason": "test"}).status_code == 200
    assert client.post("/predict", json={"x1": 0.2, "x2": 0.4}).json()["model_version"] == "v1"
    assert client.post("/rollback").status_code == 200
    assert client.post("/predict", json={"x1": 0.2, "x2": 0.4}).json()["model_version"] == "v2"
    assert monitor.get("/health").status_code == 200
    assert monitor.get("/metrics").json()["count"] == 3
    assert monitor.get("/plot.png").headers["content-type"] == "image/png"
    assert len(state.log_path.read_text(encoding="utf-8").splitlines()) == 3
    assert client.get("/metrics").json()["p95_request_latency_ms"] >= 0


def test_bad_candidate_hash_blocks_switch_and_keeps_active(runtime):
    path, client, _, _ = runtime
    candidate = path.parent / "model_v1.joblib"
    with candidate.open("ab") as out:
        out.write(b"corrupt")
    response = client.post("/promote", json={"version": "v1", "reason": "tampered"})
    assert response.status_code == 409
    assert client.get("/models").json()["active"] == "v2"
    assert client.post("/predict", json={"x1": .2, "x2": .1}).status_code == 200


def test_bad_registry_refused_at_startup(runtime):
    path, _, _, _ = runtime
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["versions"]["v2"]["sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ModelRegistry(path)


def test_under_concurrent_switch_no_request_lost(runtime):
    _, client, _, _ = runtime
    def worker(i):
        return [client.post("/predict", json={"x1": .2, "x2": .1}).status_code for _ in range(60)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = [pool.submit(worker, i) for i in range(8)]
        client.post("/promote", json={"version": "v1"})
        client.post("/rollback")
        results = [response for task in tasks for response in task.result()]
    assert len(results) == 480
    assert set(results) == {200}
    assert client.get("/models").json()["active"] == "v2"
