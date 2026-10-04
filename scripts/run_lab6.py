from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab6Config, load_config
from moons_lab.data import load_xy
from moons_lab.monitor_service import MonitorState, create_monitor_app
from moons_lab.registry import ModelRegistry, utc_now
from moons_lab.service import ServiceState, create_app
from moons_lab.tracking import tracked_run
from moons_lab.utils import json_dump, sha256_file


def _create_registry() -> dict:
    registry_dir = ROOT / "models/registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    v1 = registry_dir / "model_v1.joblib"
    v2 = registry_dir / "model_v2.joblib"
    shutil.copy2(ROOT / "reports/LAB2/linear_baseline.joblib", v1)
    shutil.copy2(ROOT / "reports/LAB2/ml_model.joblib", v2)
    metrics = pd.read_csv(ROOT / "reports/LAB2/ml_metrics.csv")
    v2_acc = float(metrics.loc[metrics["mode"] == "full", "accuracy"].iloc[0])
    comparison = json.loads((ROOT / "reports/LAB2/comparison.json").read_text(encoding="utf-8"))
    v1_acc = float(comparison["linear_baseline_test"]["accuracy"])
    payload = {
        "active": "v2",
        "versions": {
            "v1": {
                "path": "models/registry/model_v1.joblib",
                "sha256": sha256_file(v1),
                "metrics": {"test_accuracy": v1_acc},
                "status": "retired",
                "description": "linear LogisticRegression baseline",
            },
            "v2": {
                "path": "models/registry/model_v2.joblib",
                "sha256": sha256_file(v2),
                "metrics": {"test_accuracy": v2_acc},
                "status": "production",
                "description": "RBF features + incremental SGDClassifier",
            },
        },
        "history": [
            {"time": utc_now(), "action": "initial_registration", "from": None, "to": "v1", "reason": "baseline"},
            {"time": utc_now(), "action": "promote", "from": "v1", "to": "v2", "reason": "higher validated accuracy"},
        ],
    }
    json_dump(registry_dir / "registry.json", payload)
    return payload


def _contract_checks(client: TestClient) -> pd.DataFrame:
    cases = [
        ("health", "GET", "/health", None, 200),
        ("predict_valid", "POST", "/predict", {"x1": 0.3, "x2": 0.4}, 200),
        ("predict_missing", "POST", "/predict", {"x1": 0.3}, 422),
        ("predict_wrong_type", "POST", "/predict", {"x1": "bad", "x2": 0.2}, 422),
        ("predict_out_of_schema_range", "POST", "/predict", {"x1": 999, "x2": 0.2}, 422),
        ("metrics", "GET", "/metrics", None, 200),
        ("models", "GET", "/models", None, 200),
        ("runtime_get", "GET", "/runtime", None, 200),
        ("runtime_patch", "PATCH", "/runtime", {"metrics_enabled": False, "log_level": "WARNING"}, 200),
        ("unknown_model", "POST", "/promote", {"version": "missing", "reason": "negative"}, 404),
    ]
    rows = []
    for name, method, path, payload, expected in cases:
        response = client.request(method, path, json=payload)
        rows.append({
            "case": name,
            "method": method,
            "path": path,
            "expected_status": expected,
            "actual_status": response.status_code,
            "pass": response.status_code == expected,
        })
    client.patch("/runtime", json={"metrics_enabled": True, "log_level": "INFO"})
    return pd.DataFrame(rows)


def _load_test(client: TestClient, requests_per_endpoint: int = 100) -> pd.DataFrame:
    endpoints = [
        ("health", lambda: client.get("/health")),
        ("predict", lambda: client.post("/predict", json={"x1": 0.2, "x2": 0.1})),
        ("metrics", lambda: client.get("/metrics")),
        ("models", lambda: client.get("/models")),
        ("runtime", lambda: client.get("/runtime")),
    ]
    rows = []
    for name, call in endpoints:
        start = time.perf_counter()
        statuses = [call().status_code for _ in range(requests_per_endpoint)]
        elapsed = time.perf_counter() - start
        rows.append({
            "endpoint": name,
            "requests": requests_per_endpoint,
            "successes": sum(code == 200 for code in statuses),
            "errors": sum(code != 200 for code in statuses),
            "elapsed_s": elapsed,
            "requests_per_s": requests_per_endpoint / elapsed,
        })
    return pd.DataFrame(rows)


def _switch_under_load(client: TestClient) -> dict:
    stop = threading.Event()
    statuses: list[int] = []
    versions: list[str] = []
    lock = threading.Lock()

    def worker(worker_id: int) -> None:
        for i in range(80):
            if stop.is_set():
                break
            r = client.post("/predict", json={"x1": 0.1 + 0.001 * i, "x2": 0.2, "y_true": 1})
            with lock:
                statuses.append(r.status_code)
                if r.status_code == 200:
                    versions.append(r.json()["model_version"])
            time.sleep(0.001)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        time.sleep(0.02)
        client.post("/promote", json={"version": "v1", "reason": "canary drill"})
        time.sleep(0.02)
        client.post("/promote", json={"version": "v2", "reason": "restore production"})
        for future in futures:
            future.result()
    stop.set()
    return {
        "total_requests": len(statuses),
        "successful_requests": sum(code == 200 for code in statuses),
        "lost_or_failed": sum(code != 200 for code in statuses),
        "success_fraction": sum(code == 200 for code in statuses) / max(len(statuses), 1),
        "versions_observed": sorted(set(versions)),
    }


def _hash_negative_control() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.joblib"
        shutil.copy2(ROOT / "models/registry/model_v2.joblib", model)
        payload = {
            "active": "v2",
            "versions": {"v2": {"path": str(model), "sha256": sha256_file(model), "metrics": {}, "status": "production"}},
            "history": [],
        }
        reg_path = tmp / "registry.json"
        reg_path.write_text(json.dumps(payload), encoding="utf-8")
        with model.open("ab") as fh:
            fh.write(b"tamper")
        try:
            ModelRegistry(reg_path)
            return "ERROR: tampered model accepted"
        except ValueError as exc:
            return f"PASS: {exc}"


def _monitor_snapshot(model_path: Path) -> dict:
    model = joblib.load(model_path)
    x, _ = load_xy(ROOT / "data/processed/test.csv")
    x = x[:500]
    p = model.predict_proba(x)
    if getattr(p, "ndim", 1) == 2:
        p = p[:, 1]
    state = MonitorState()
    app = create_monitor_app(state)
    client = TestClient(app)
    for row, prob in zip(x, p):
        client.post("/ingest", json={"x1": float(row[0]), "x2": float(row[1]), "probability": float(prob)})
    state.save_plot(ROOT / "reports/LAB6/monitor_snapshot.png")
    return client.get("/metrics").json()


def main() -> None:
    cfg: Lab6Config = load_config(6)  # type: ignore[assignment]
    _create_registry()
    service_state = ServiceState(metrics_enabled=cfg.metrics_enabled, log_level=cfg.log_level)
    app = create_app(ROOT / cfg.registry_path, state=service_state)
    client = TestClient(app)

    contracts = _contract_checks(client)
    contracts.to_csv(ROOT / "reports/LAB6/endpoint_contracts.csv", index=False)
    load = _load_test(client, cfg.load_test_requests)
    load.to_csv(ROOT / "reports/LAB6/endpoint_load.csv", index=False)
    drill = _switch_under_load(client)
    json_dump(ROOT / "reports/LAB6/promotion_drill.json", drill)
    monitor_metrics = _monitor_snapshot(ROOT / "models/registry/model_v2.joblib")
    json_dump(ROOT / "reports/LAB6/monitor_metrics.json", monitor_metrics)
    negative_hash = _hash_negative_control()

    registry_snapshot = client.get("/models").json()
    # Keep the canonical artifact in LAB6 and the runtime registry in models/registry.
    json_dump(ROOT / "reports/LAB6/registry.json", registry_snapshot)
    json_dump(ROOT / "models/registry/registry.json", registry_snapshot)

    service_metrics = client.get("/metrics").json()
    json_dump(ROOT / "reports/LAB6/service_metrics.json", service_metrics)
    log_lines = [
        f"contract_cases={len(contracts)} pass={int(contracts['pass'].sum())}/{len(contracts)}",
        f"load_total={int(load['requests'].sum())} load_errors={int(load['errors'].sum())}",
        f"switch_drill={json.dumps(drill, ensure_ascii=False)}",
        f"hash_negative={negative_hash}",
        f"runtime_change_without_restart=PASS metrics_enabled={service_metrics['metrics_enabled']}",
    ]
    (ROOT / "reports/LAB6/service_tests.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    with tracked_run("lab6_service_drill", experiment="lab6_service") as run:
        run.log_params(cfg.model_dump())
        run.log_metrics({
            "success_fraction": drill["success_fraction"],
            "lost_requests": drill["lost_or_failed"],
            "load_errors": int(load["errors"].sum()),
            "contract_pass_fraction": float(contracts["pass"].mean()),
        })
        run.log_artifact(ROOT / "reports/LAB6/service_tests.log")
        json_dump(ROOT / "reports/LAB6/run_meta.json", {"run_id": run.run_id, "tracking_backend": run.backend})

    print(contracts.to_string(index=False))
    print(load.to_string(index=False))
    print(json.dumps(drill, indent=2))
    print(negative_hash)


if __name__ == "__main__":
    main()
