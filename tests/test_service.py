import json
import shutil
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from moons_lab.registry import ModelRegistry
from moons_lab.service import ServiceState, create_app
from moons_lab.utils import sha256_file


def _registry_copy(tmp_path: Path) -> Path:
    root = Path.cwd()
    v1 = tmp_path / "v1.joblib"
    v2 = tmp_path / "v2.joblib"
    shutil.copy2(root / "models/registry/model_v1.joblib", v1)
    shutil.copy2(root / "models/registry/model_v2.joblib", v2)
    payload = {
        "active": "v2",
        "versions": {
            "v1": {"path": str(v1), "sha256": sha256_file(v1), "metrics": {}, "status": "retired"},
            "v2": {"path": str(v2), "sha256": sha256_file(v2), "metrics": {}, "status": "production"},
        },
        "history": [{"action": "promote", "from": "v1", "to": "v2", "time": "x", "reason": "test"}],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registry_promote_rollback_and_hash(tmp_path):
    path = _registry_copy(tmp_path)
    reg = ModelRegistry(path)
    assert reg.active_version == "v2"
    reg.promote("v1", "test")
    assert reg.active_version == "v1"
    reg.rollback()
    assert reg.active_version == "v2"
    with open(reg.snapshot()["versions"]["v2"]["path"], "ab") as fh:
        fh.write(b"tamper")
    reg._cache.clear()
    try:
        reg.get_model("v2")
        assert False, "tamper should fail"
    except ValueError:
        pass


def test_service_contract_and_batch_fallback(tmp_path):
    path = _registry_copy(tmp_path)
    state = ServiceState()
    client = TestClient(create_app(path, state=state))
    assert client.get("/health").status_code == 200
    good = client.post("/predict", json={"x1": 0.2, "x2": 0.3, "y_true": 1})
    assert good.status_code == 200
    assert client.post("/predict", json={"x1": 100, "x2": 0}).status_code == 422
    assert client.get("/models").status_code == 200
    assert client.patch("/runtime", json={"metrics_enabled": False, "log_level": "WARNING"}).json()["metrics_enabled"] is False
    assert client.post("/promote", json={"version": "missing"}).status_code == 404
    assert client.post("/promote", json={"version": "v1", "reason": "test"}).status_code == 200
    assert client.post("/rollback").status_code == 200

    class BrokenParallel:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            raise RuntimeError("broken")

    with patch("moons_lab.service.Parallel", BrokenParallel):
        response = client.post("/predict/batch", json={"items": [{"x1": 0.1, "x2": 0.2}] * 3, "n_jobs": 2})
    assert response.status_code == 200
    assert response.json()["count"] == 3
    assert client.get("/metrics").json()["batch_fallbacks"] == 1
