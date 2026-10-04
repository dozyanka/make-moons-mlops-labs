from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import psutil
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab7Config, load_config
from moons_lab.data import load_xy
from moons_lab.drift import compute_drift
from moons_lab.evidently_adapter import evidently_available, save_evidently_report
from moons_lab.generator import GeneratorSettings, generate_dataframe
from moons_lab.service import ServiceState, create_app
from moons_lab.tracking import tracked_run
from moons_lab.utils import json_dump


def _benchmark(client: TestClient, x: np.ndarray, cfg: Lab7Config) -> pd.DataFrame:
    items = [{"x1": float(row[0]), "x2": float(row[1])} for row in x[: cfg.benchmark_batch_size]]
    rows = []
    cpu_count = max(1, os.cpu_count() or 1)
    # Требование с лекции: отдельно проверяем режим, использующий все доступные ядра.
    job_values = list(dict.fromkeys([*cfg.jobs, cpu_count]))
    process = psutil.Process(os.getpid())
    for jobs in job_values:
        effective = min(jobs, cpu_count)
        # warm-up
        warm = items[: min(200, len(items))]
        response = client.post("/predict/batch", json={"items": warm, "n_jobs": jobs})
        if response.status_code != 200:
            raise RuntimeError(response.text)
        for repeat in range(cfg.benchmark_repeats):
            cpu_before = process.cpu_times()
            rss_before = process.memory_info().rss
            start = time.perf_counter()
            response = client.post("/predict/batch", json={"items": items, "n_jobs": jobs})
            elapsed = time.perf_counter() - start
            cpu_after = process.cpu_times()
            rss_after = process.memory_info().rss
            if response.status_code != 200:
                raise RuntimeError(response.text)
            cpu_s = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
            rows.append({
                "requested_jobs": jobs,
                "effective_jobs": response.json()["n_jobs"],
                "uses_all_available_cores": response.json()["n_jobs"] == cpu_count,
                "cpu_count": cpu_count,
                "repeat": repeat + 1,
                "batch_size": len(items),
                "elapsed_s": elapsed,
                "latency_ms_per_item": 1000 * elapsed / len(items),
                "throughput_items_s": len(items) / elapsed,
                "process_cpu_s": cpu_s,
                "cpu_utilization_pct_of_machine": 100.0 * cpu_s / max(elapsed * cpu_count, 1e-12),
                "rss_before_mb": rss_before / (1024 * 1024),
                "rss_after_mb": rss_after / (1024 * 1024),
                "warmup_done": True,
                "expected_effective_jobs": effective,
            })
    return pd.DataFrame(rows)


def _predict(model, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x))
    return p[:, 1] if p.ndim == 2 else p


def _drift_drill(model, cfg: Lab7Config) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ref_df = generate_dataframe(GeneratorSettings(mode="similar", n_samples=5000, seed=cfg.seed, noise=0.22))
    ref_x = ref_df[["x1", "x2"]].to_numpy()
    ref_p = _predict(model, ref_x)
    rows = []
    drill_rows = []
    first_alert = None
    last_df = None
    for window in range(cfg.n_windows):
        if window < cfg.drift_start_window:
            settings = GeneratorSettings(
                mode="similar", n_samples=cfg.window_size, seed=cfg.seed + 10 + window, noise=0.22
            )
            injected = False
        else:
            settings = GeneratorSettings(
                mode="similar",
                n_samples=cfg.window_size,
                seed=cfg.seed + 10 + window,
                noise=0.22,
                shift_x=cfg.drift_shift_x,
                shift_y=cfg.drift_shift_y,
            )
            injected = True
        cur_df = generate_dataframe(settings)
        last_df = cur_df
        cur_x = cur_df[["x1", "x2"]].to_numpy()
        cur_p = _predict(model, cur_x)
        metrics = compute_drift(ref_x, cur_x, ref_p, cur_p)
        alert = metrics.max_feature_psi > cfg.psi_alert_threshold or metrics.csi_prediction > cfg.csi_alert_threshold
        if alert and first_alert is None:
            first_alert = window
        rows.append({
            "window": window,
            "injected_drift": injected,
            "psi_x1": metrics.psi_x1,
            "psi_x2": metrics.psi_x2,
            "ks_x1": metrics.ks_x1,
            "ks_x2": metrics.ks_x2,
            "csi_prediction": metrics.csi_prediction,
            "alert": alert,
        })
    pre = [row for row in rows if row["window"] < cfg.drift_start_window]
    post = [row for row in rows if row["window"] >= cfg.drift_start_window]
    drill_rows.append({
        "drill": "clean_stream",
        "expected": "no alerts",
        "actual_alerts": sum(int(r["alert"]) for r in pre),
        "pass": sum(int(r["alert"]) for r in pre) == 0,
    })
    drill_rows.append({
        "drill": "shift_x_y",
        "expected": f"detect no later than window {cfg.drift_start_window + 1}",
        "actual_alerts": sum(int(r["alert"]) for r in post),
        "pass": first_alert is not None and first_alert <= cfg.drift_start_window + 1,
    })

    status = "unavailable"
    if evidently_available() and last_df is not None:
        status = save_evidently_report(
            ref_df[["x1", "x2", "y"]],
            last_df[["x1", "x2", "y"]],
            ROOT / "reports/LAB7/evidently_drift_report.html",
        )
    return pd.DataFrame(rows), pd.DataFrame(drill_rows), status


def _worker_failure_fallback(client: TestClient) -> dict:
    class BrokenParallel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise RuntimeError("simulated worker-pool failure")

    with patch("moons_lab.service.Parallel", BrokenParallel):
        response = client.post(
            "/predict/batch",
            json={"items": [{"x1": 0.1, "x2": 0.2}, {"x1": 0.2, "x2": 0.3}], "n_jobs": 2},
        )
    metrics = client.get("/metrics").json()
    return {
        "status_code": response.status_code,
        "returned_predictions": response.json().get("count") if response.status_code == 200 else 0,
        "batch_fallbacks": metrics["batch_fallbacks"],
        "pass": response.status_code == 200 and response.json().get("count") == 2 and metrics["batch_fallbacks"] >= 1,
    }


def main() -> None:
    cfg: Lab7Config = load_config(7)  # type: ignore[assignment]
    state = ServiceState()
    app = create_app(ROOT / "models/registry/registry.json", state=state)
    client = TestClient(app)
    x_test, _ = load_xy(ROOT / "data/processed/test.csv")
    if cfg.benchmark_batch_size > len(x_test):
        repeats = int(np.ceil(cfg.benchmark_batch_size / len(x_test)))
        x_bench = np.tile(x_test, (repeats, 1))
    else:
        x_bench = x_test
    benchmark = _benchmark(client, x_bench, cfg)
    benchmark.to_csv(ROOT / "reports/LAB7/parallel_benchmark.csv", index=False)

    model = joblib.load(ROOT / "models/registry/model_v2.joblib")
    drift_metrics, drill, evidently_status = _drift_drill(model, cfg)
    drift_metrics.to_csv(ROOT / "reports/LAB7/drift_metrics.csv", index=False)
    drill.to_csv(ROOT / "reports/LAB7/drift_drills.csv", index=False)
    (ROOT / "reports/LAB7/evidently_status.txt").write_text(
        f"evidently_available={evidently_available()}\nresult={evidently_status}\n",
        encoding="utf-8",
    )
    failure = _worker_failure_fallback(client)
    json_dump(ROOT / "reports/LAB7/worker_failure_negative.json", failure)

    grouped = benchmark.groupby("requested_jobs")["throughput_items_s"].mean().reset_index()
    best_row = grouped.loc[grouped["throughput_items_s"].idxmax()]
    all_cores = max(1, os.cpu_count() or 1)
    all_core_rows = benchmark[benchmark["effective_jobs"] == all_cores]
    summary = {
        "cpu_count": all_cores,
        "all_cores_benchmarked": bool(len(all_core_rows)),
        "all_cores_mean_throughput_items_s": float(all_core_rows["throughput_items_s"].mean()) if len(all_core_rows) else None,
        "best_requested_jobs": int(best_row["requested_jobs"]),
        "best_mean_throughput_items_s": float(best_row["throughput_items_s"]),
        "clean_false_alerts": int(drift_metrics.loc[~drift_metrics["injected_drift"], "alert"].sum()),
        "first_drift_alert_window": (
            int(drift_metrics.loc[drift_metrics["alert"], "window"].min())
            if drift_metrics["alert"].any()
            else None
        ),
        "worker_failure_fallback": failure,
        "evidently_status": evidently_status,
    }
    json_dump(ROOT / "reports/LAB7/summary.json", summary)
    with tracked_run("lab7_parallel_and_drift", experiment="lab7_monitoring") as run:
        run.log_params(cfg.model_dump())
        run.log_metrics({
            "best_mean_throughput_items_s": summary["best_mean_throughput_items_s"],
            "clean_false_alerts": summary["clean_false_alerts"],
            "first_drift_alert_window": summary["first_drift_alert_window"] or -1,
            "worker_fallback_pass": float(failure["pass"]),
        })
        run.log_artifact(ROOT / "reports/LAB7/parallel_benchmark.csv")
        run.log_artifact(ROOT / "reports/LAB7/drift_metrics.csv")
        json_dump(ROOT / "reports/LAB7/run_meta.json", {"run_id": run.run_id, "tracking_backend": run.backend})

    print(benchmark.groupby("requested_jobs")["throughput_items_s"].agg(["mean", "std"]).to_string())
    print(drift_metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
