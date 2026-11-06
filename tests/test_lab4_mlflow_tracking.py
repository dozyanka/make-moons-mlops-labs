"""MLflow backend compatibility regression tests for Lab4 patch 02."""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _lab4_runner():
    # Import the script as a module: does not execute main() or retrain models.
    spec = importlib.util.spec_from_file_location("lab4_runner_for_test", ROOT / "scripts/run_lab4.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracking_enables_file_store_before_mlflow_operations(monkeypatch):
    runner = _lab4_runner()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    observed = []
    fake = types.ModuleType("mlflow")

    def set_tracking_uri(uri):
        observed.append(("set_tracking_uri", os.environ.get("MLFLOW_ALLOW_FILE_STORE"), uri))

    def set_experiment(name):
        observed.append(("set_experiment", os.environ.get("MLFLOW_ALLOW_FILE_STORE"), name))

    @contextmanager
    def start_run(run_name):
        observed.append(("start_run", os.environ.get("MLFLOW_ALLOW_FILE_STORE"), run_name))
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="stub-run-123"))

    fake.set_tracking_uri = set_tracking_uri
    fake.set_experiment = set_experiment
    fake.start_run = start_run
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    with runner.tracking_run("full", offline=False) as (run_id, backend, tracking):
        assert run_id == "stub-run-123"
        assert backend == "mlflow"
        assert tracking is fake

    assert [event[0] for event in observed] == ["set_tracking_uri", "set_experiment", "start_run"]
    assert all(event[1] == "true" for event in observed)
    assert observed[1][2] == "lab4_validator"


def test_offline_tracking_does_not_touch_mlflow(monkeypatch):
    runner = _lab4_runner()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    with runner.tracking_run("chunks", offline=True) as (run_id, backend, tracking):
        assert run_id == "OFFLINE_NOT_MLFLOW_chunks"
        assert backend == "offline_diagnostic"
        assert tracking is None
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"


def test_real_mlflow_run_ids_if_installed(monkeypatch):
    if importlib.util.find_spec("mlflow") is None:
        pytest.skip("MLflow package not installed in this test environment")
    import mlflow

    out = ROOT / "reports/LAB4/validator_metrics.csv"
    if not out.is_file():
        pytest.fail("Run scripts/run_lab4.py before integration validation")
    with out.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["mode"] for row in rows} == {"full", "chunks"}
    assert all(row["tracking_backend"] == "mlflow" for row in rows), (
        "Expected real MLflow runs; rerun without --offline-test"
    )
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    for row in rows:
        info = mlflow.get_run(row["run_id"])
        assert info.info.status == "FINISHED"
        assert info.data.params["mode"] == row["mode"]


def test_independent_repeat_runs_recorded_in_mlflow(monkeypatch):
    if importlib.util.find_spec("mlflow") is None:
        pytest.skip("MLflow is unavailable; full integration verification runs on Windows/CI")
    import json
    import mlflow

    comparison = ROOT / "reports/LAB4/repeat_verification.json"
    metrics = ROOT / "reports/LAB4/validator_metrics.csv"
    assert comparison.exists(), "Run scripts/run_lab4.py to create independent repeat runs"
    assert metrics.exists()
    summary = json.loads(comparison.read_text(encoding="utf-8"))
    assert summary["all_pass"] is True
    assert summary["source_hashes_unchanged"] is True
    with metrics.open(encoding="utf-8", newline="") as stream:
        originals = {row["mode"]: row for row in csv.DictReader(stream)}
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    for mode in ("full", "chunks"):
        record = summary[mode]
        assert record["npz_sha256_equal"] is True
        assert record["decisions_equal"] is True
        assert record["primary_run_id"] == originals[mode]["run_id"]
        repeat = mlflow.get_run(record["repeat_run_id"])
        assert repeat.info.status == "FINISHED"
        assert repeat.data.params["repeat_of_run_id"] == originals[mode]["run_id"]
        assert repeat.data.params["mode"] == mode
