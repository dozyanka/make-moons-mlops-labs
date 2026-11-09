"""LAB7 tests: reproducibility, window/step, live API, drift, fallback, Evidently."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from moons_lab.config import Lab7Config, load_config
from moons_lab.evidently_adapter import evaluate_evidently_window
from moons_lab.generator import GeneratorSettings, generate_dataframe
from moons_lab.lab6_bootstrap import build_registry
from moons_lab.lab7_pipeline import (
    SCENARIOS, assess_window, collect_window_metrics, drifting_settings, summarize_drill,
)
from moons_lab.service import create_app

ROOT = Path(__file__).resolve().parents[1]


def config(**changes) -> Lab7Config:
    base = load_config(7, ROOT / "configs/lab7.json").model_dump()
    base.update({"window_size": 300, "window_step": 150, "reference_size": 700,
                 "n_windows": 6, "drift_start_window": 3})
    base.update(changes)
    return Lab7Config.model_validate(base)


def fake_predict(frame: pd.DataFrame) -> np.ndarray:
    x = frame[["x1", "x2"]].to_numpy()
    logits = np.clip(2 * x[:, 0] + x[:, 1], -30, 30)
    return 1 / (1 + np.exp(-logits))


def test_lab7_pydantic_invalid_inputs():
    with pytest.raises(ValidationError, match="window_step"):
        config(window_step=500)
    with pytest.raises(ValidationError, match="drift_start_window"):
        config(drift_start_window=6)
    with pytest.raises(ValidationError, match="jobs"):
        config(jobs=[1, 0])
    with pytest.raises(ValidationError, match="drift_class_ratio"):
        config(drift_class_ratio=2.0)


def test_all_scenarios_and_error():
    cfg = config()
    for scenario in SCENARIOS:
        first = drifting_settings(cfg, scenario, 0)
        after = drifting_settings(cfg, scenario, cfg.drift_start_window)
        assert first.n_samples == cfg.window_size
        assert after.n_samples == cfg.window_step
        assert first.shift_x == 0 and first.class_1_ratio == 0.5
        if scenario == "mean_shift":
            assert after.shift_x == cfg.drift_shift_x
        if scenario == "class_ratio_shift":
            assert after.class_1_ratio == cfg.drift_class_ratio
    with pytest.raises(ValueError, match="unknown"):
        drifting_settings(cfg, "invented", 1)


def test_generator_chunk_is_deterministic():
    cfg = config()
    first = generate_dataframe(drifting_settings(cfg, "class_ratio_shift", 3))
    second = generate_dataframe(drifting_settings(cfg, "class_ratio_shift", 3))
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_sliding_windows_reproducible_and_bounded():
    cfg = config()
    reference = generate_dataframe(GeneratorSettings(n_samples=700, seed=21))
    evaluations = []

    def check_window(name, idx, frame):
        assert len(frame) == cfg.window_size
        assert set(frame.columns) == {"x1", "x2", "probability"}
        evaluations.append((name, idx))
        return {"evidently_evaluated": True, "evidently_metrics_count": 4}

    left, last = collect_window_metrics(cfg, "mean_shift", reference, fake_predict,
                                        on_window=check_window)
    right, _ = collect_window_metrics(cfg, "mean_shift", reference, fake_predict)
    assert len(left) == cfg.n_windows == len(evaluations)
    assert last.shape == (cfg.window_size, 3)
    assert (left.end_row_exclusive - left.start_row).eq(cfg.window_size).all()
    assert list(left.start_row) == [i * cfg.window_step for i in range(cfg.n_windows)]
    assert (left.loc[left.window < cfg.drift_start_window, "drift_fraction"] == 0).all()
    assert left.loc[left.window == cfg.drift_start_window, "drift_fraction"].iloc[0] == cfg.window_step / cfg.window_size
    pd.testing.assert_frame_equal(left[right.columns], right, check_exact=True)
    assert left["evidently_evaluated"].all()


def test_healthy_stream_has_no_false_alarm():
    cfg = config()
    ref = generate_dataframe(GeneratorSettings(n_samples=700, seed=101))
    rows, _ = collect_window_metrics(cfg, "clean", ref, fake_predict)
    result = summarize_drill(rows, "clean", cfg)
    assert result["accepted"] is True
    assert result["false_alerts"] <= cfg.max_clean_false_alerts


@pytest.mark.parametrize("scenario", ["mean_shift", "class_ratio_shift"])
def test_drift_is_detected_within_deadline(scenario):
    cfg = config()
    ref = generate_dataframe(GeneratorSettings(n_samples=700, seed=101))
    rows, _ = collect_window_metrics(cfg, scenario, ref, fake_predict)
    outcome = summarize_drill(rows, scenario, cfg)
    assert outcome["accepted"] is True
    assert outcome["first_alert_window"] >= cfg.drift_start_window
    assert outcome["detection_delay_windows"] <= cfg.detection_deadline_windows


def test_bad_prediction_count_refused():
    cfg = config()
    ref = generate_dataframe(GeneratorSettings(n_samples=700, seed=21))
    with pytest.raises(ValueError, match="shape"):
        collect_window_metrics(cfg, "clean", ref, lambda df: np.array([0.5]))
    with pytest.raises(ValueError, match="length mismatch"):
        assess_window(ref, ref.iloc[:cfg.window_size], fake_predict(ref),
                      np.array([0.1]), scenario="clean", index=0, drift_fraction=0, cfg=cfg)


def test_live_api_batch_order_parity_and_worker_failure(tmp_path):
    reg = build_registry(ROOT, tmp_path / "registry", reset=True)
    client = TestClient(create_app(reg))
    single = client.post("/predict/batch", json={"items": [{"x1": 0.1, "x2": 0.2}] * 8, "n_jobs": 1})
    parallel = client.post("/predict/batch", json={"items": [{"x1": 0.1, "x2": 0.2}] * 8,
                                                  "n_jobs": min(4, os.cpu_count() or 1)})
    assert single.status_code == parallel.status_code == 200
    assert single.json()["results"] == parallel.json()["results"]
    assert parallel.json()["n_jobs"] <= (os.cpu_count() or 1)
    assert client.post("/predict/batch", json={"items": [], "n_jobs": 1}).status_code == 422
    assert client.post("/predict/batch", json={"items": [{"x1": 0.1, "x2": 0.2}], "n_jobs": 0}).status_code == 422
    from unittest.mock import patch

    class Failing:
        def __init__(self, *a, **k):
            pass
        def __call__(self, *a, **k):
            raise RuntimeError("simulated joblib failure")

    with patch("moons_lab.service.Parallel", Failing):
        failed = client.post("/predict/batch", json={"items": [{"x1": 0.1, "x2": 0.2}] * 4, "n_jobs": 2})
        assert client.get("/health").status_code == 200
    assert failed.status_code == 200
    assert failed.json()["count"] == 4
    assert client.get("/metrics").json()["batch_fallbacks"] == 1


def test_evidently_v07_real_html(tmp_path):
    pytest.importorskip("evidently")
    ref = generate_dataframe(GeneratorSettings(n_samples=300, seed=7))
    cur = generate_dataframe(GeneratorSettings(n_samples=300, seed=8, shift_x=.9))
    ref["probability"] = fake_predict(ref)
    cur["probability"] = fake_predict(cur)
    result = evaluate_evidently_window(ref, cur, tmp_path / "evidently.html")
    assert result["evidently_evaluated"] is True
    assert (tmp_path / "evidently.html").stat().st_size > 100


def test_runner_benchmark_and_report_helpers(tmp_path):
    """Exercise real TestClient joblib timing, parity and the report writer."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lab7_runner_for_tests", ROOT / "scripts/run_lab7.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = config(benchmark_batch_size=120, benchmark_repeats=2)
    reg = build_registry(ROOT, tmp_path / "registry", reset=True)
    client = TestClient(create_app(reg))
    samples = generate_dataframe(GeneratorSettings(n_samples=120, seed=123))
    benchmark = runner._benchmark(client, samples[["x1", "x2"]].to_numpy(), cfg, quick=True)
    assert benchmark["repeat"].nunique() == 2
    assert benchmark["effective_jobs"].min() == 1
    assert benchmark["predictions_match_single_worker"].all()
    assert all(benchmark["warmup_requests"] == 2)
    assert (benchmark["batch_latency_ms"] > 0).all()
    failure = runner._worker_failure(client)
    assert failure["pass"] is True
    drill = pd.DataFrame([
        {"scenario": "clean", "first_alert_window": None,
         "detection_delay_windows": None, "false_alerts": 0, "accepted": True},
    ])
    run_id = "test-run-id"
    report = tmp_path / "report"
    report.mkdir()
    runner._make_report(report, cfg, benchmark, drill, {
        "worker_failure": failure,
        "source_model_sha256": "sha-test",
        "config_sha256": "config-test",
    }, run_id, "test")
    content = (report / "lab7_report.md").read_text(encoding="utf-8")
    assert run_id in content and "p50 / p95 / p99" in content
