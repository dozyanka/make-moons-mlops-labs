"""LAB7 evidence: joblib benchmark, bounded streaming + Evidently, failure drills.

Run (Windows): .venv\\Scripts\\python.exe scripts/run_lab7.py --report-dir demo_output/LAB7
The official reports/LAB7 is only updated by the isolated Git worktree PR installer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.lab7_config import Lab7Config, load_lab7_config
from moons_lab.data import load_xy
from moons_lab.evidently_adapter import evaluate_evidently_window, evidently_available
from moons_lab.generator import GeneratorSettings, generate_dataframe
from moons_lab.lab6_bootstrap import build_registry
from moons_lab.lab7_pipeline import SCENARIOS, collect_window_metrics, summarize_drill
from moons_lab.service import ServiceState, create_app
from moons_lab.tracking import tracked_run
from moons_lab.utils import json_dump, sha256_file


def _predict_via_api(client: TestClient, frame: pd.DataFrame) -> np.ndarray:
    items = frame[["x1", "x2"]].to_dict(orient="records")
    result = client.post("/predict/batch", json={"items": items, "n_jobs": 1})
    if result.status_code != 200:
        raise RuntimeError(f"API prediction failed ({result.status_code}): {result.text[:150]}")
    content = result.json()
    if content["count"] != len(items) or len(content["results"]) != len(items):
        raise RuntimeError("API prediction count mismatch")
    return np.asarray([p["probability"] for p in content["results"]], dtype=float)


def _benchmark(client: TestClient, x: np.ndarray, cfg: Lab7Config, *, quick: bool) -> pd.DataFrame:
    cpu_count = max(1, os.cpu_count() or 1)
    if cpu_count > 256:
        raise RuntimeError("Service n_jobs field supports at most 256 logical cores")
    jobs_to_try = list(dict.fromkeys([1, *cfg.jobs, cpu_count]))
    if quick:
        jobs_to_try = list(dict.fromkeys([1, min(2, cpu_count)]))
    jobs_to_try = [j for j in jobs_to_try if 1 <= j <= cpu_count]
    x = np.asarray(x, dtype=float)
    required = cfg.benchmark_batch_size
    if len(x) < required:
        x = np.tile(x, (int(np.ceil(required / len(x))), 1))
    items = [{"x1": float(a), "x2": float(b)} for a, b in x[:required]]
    process = psutil.Process()
    rows = []
    # Verify parallel prediction parity against a deterministic single-worker answer.
    single = client.post("/predict/batch", json={"items": items, "n_jobs": 1})
    if single.status_code != 200:
        raise RuntimeError(single.text)
    expected_probabilities = np.asarray([r["probability"] for r in single.json()["results"]])
    for jobs in jobs_to_try:
        # Warmup is measured separately from the repeated benchmark.
        for _ in range(2):
            warmed = client.post("/predict/batch", json={"items": items[:min(200, len(items))], "n_jobs": jobs})
            if warmed.status_code != 200:
                raise RuntimeError("warmup failed: " + warmed.text[:150])
        for repeat in range(cfg.benchmark_repeats):
            memory_before = process.memory_info().rss
            cpu_before = process.cpu_times()
            fallbacks_before = client.get("/metrics").json()["batch_fallbacks"]
            started = time.perf_counter()
            response = client.post("/predict/batch", json={"items": items, "n_jobs": jobs})
            elapsed = max(1e-9, time.perf_counter() - started)
            cpu_after = process.cpu_times()
            if response.status_code != 200:
                raise RuntimeError(f"benchmark returned HTTP {response.status_code}: {response.text[:150]}")
            fallbacks_after = client.get("/metrics").json()["batch_fallbacks"]
            if fallbacks_after != fallbacks_before:
                raise AssertionError("joblib failed and service silently used sequential fallback; invalid benchmark")
            payload = response.json()
            prediction_count = payload["count"]
            observed = np.asarray([r["probability"] for r in payload["results"]])
            parity = prediction_count == len(items) and np.allclose(
                observed, expected_probabilities, atol=1e-12, rtol=0
            )
            if not parity:
                raise AssertionError(f"n_jobs={jobs}: incorrect predictions or lost rows")
            cpu_elapsed = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
            rows.append({
                "requested_jobs": jobs,
                "effective_jobs": payload["n_jobs"],
                "logical_cores": cpu_count,
                "all_cores": payload["n_jobs"] == cpu_count,
                "repeat": repeat + 1,
                "warmup_requests": 2,
                "batch_size": len(items),
                "elapsed_s": elapsed,
                "batch_latency_ms": elapsed * 1000,
                "latency_ms_per_item_amortized": elapsed * 1000 / len(items),
                "throughput_items_s": len(items) / elapsed,
                "cpu_time_s": cpu_elapsed,
                "cpu_util_pct_machine": 100 * cpu_elapsed / (elapsed * cpu_count),
                "rss_before_mb": memory_before / 1048576,
                "rss_after_mb": process.memory_info().rss / 1048576,
                "predictions_match_single_worker": bool(parity),
                "measurement": "in_process_TestClient_not_external_http",
            })
    return pd.DataFrame(rows)


def _worker_failure(client: TestClient) -> dict:
    """Inject worker-pool exception (not a real OS process kill)."""
    class BrokenParallel:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            raise RuntimeError("injected joblib executor failure")
    initial_fallbacks = client.get("/metrics").json()["batch_fallbacks"]
    with patch("moons_lab.service.Parallel", BrokenParallel):
        answer = client.post("/predict/batch", json={
            "items": [{"x1": 0.1, "x2": 0.2}, {"x1": 0.3, "x2": 0.4}], "n_jobs": 2,
        })
        health = client.get("/health")
    final_fallbacks = client.get("/metrics").json()["batch_fallbacks"]
    outcome = {
        "fault": "simulated joblib Parallel exception",
        "batch_status": answer.status_code, "health_during_fault": health.status_code,
        "predictions": answer.json().get("count", 0) if answer.status_code == 200 else 0,
        "fallback_counter_delta": final_fallbacks - initial_fallbacks,
    }
    outcome["pass"] = (outcome["batch_status"] == 200 and outcome["health_during_fault"] == 200
                       and outcome["predictions"] == 2 and outcome["fallback_counter_delta"] == 1)
    return outcome


def _make_report(report: Path, cfg: Lab7Config, benchmark: pd.DataFrame,
                 drills: pd.DataFrame, summary: dict, run_id: str, backend: str) -> None:
    code_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip() if (ROOT / ".git").exists() else "archive_without_git"
    lines = [
        "# ЛР7. Параллельный FastAPI-сервис и мониторинг дрейфа",
        "",
        "**Вариант:** 09, make_moons (нелинейность).",
        f"**Дата UTC:** {datetime.now(timezone.utc).isoformat()}; автор: обучающийся проекта (Владимир).",
        f"**Коммит кода до публикации:** `{code_sha}`.",
        f"**SHA-256 исходной модели ЛР2:** `{summary['source_model_sha256']}`.",
        f"**SHA-256 конфигурации ЛР7:** `{summary['config_sha256']}`.",
        "**Источник настроек:** `configs/lab7.json` (Pydantic `Lab7Config`).",
        f"**Прогон MLflow:** `{run_id}`; backend: `{backend}`.",
        "**Источник данных:** управляемый генератор ЛР5; координаты x1/x2, предсказания v2 через FastAPI.",
        "",
        "## 1. Методика параллельного бенчмарка",
        "", "`joblib.Parallel(prefer=threads)`; доступные логические ядра определяются `os.cpu_count()`; "
        "измерения включают 2 прогревочных запроса на конфигурацию, затем независимые повторы.",
        f"Размер пакета {cfg.benchmark_batch_size}; повторов {cfg.benchmark_repeats}; режим: TestClient в одном процессе, "
        "поэтому это **не внешняя сетевая задержка**. p50/p95/p99 по повторам приведены для одного пакетного HTTP-вызова.",
        "", "| Исполнители | Средняя производительность, элементов/с | Средняя задержка пакета, мс | p50 / p95 / p99 пакета, мс |",
        "|---:|---:|---:|---:|",
    ]
    grouped = benchmark.groupby("effective_jobs", sort=True)
    for jobs, group in grouped:
        latency = group["batch_latency_ms"].to_numpy()
        q = np.quantile(latency, [.5, .95, .99])
        lines.append(f"| {jobs} | {group['throughput_items_s'].mean():.2f} | "
                     f"{np.mean(latency):.2f} | {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} |")
    mean = grouped["throughput_items_s"].mean()
    speedup = float(mean.max() / mean.loc[1])
    lines.extend(["", f"Максимальное измеренное ускорение относительно одного исполнителя: "
                  f"{speedup:.3f}× (при {int(mean.idxmax())} исполнителях). "
                  "Ускорение может быть сублинейным либо отрицательным из-за накладных расходов Python/HTTP/задач joblib.",
                  "", "## 2. Оконное наблюдение Evidently и PSI / KS / CSI", "",
                  f"Окно: {cfg.window_size} наблюдений; шаг: {cfg.window_step}; "
                  f"окон на сценарий: {cfg.n_windows}. Новый блок поступает из генератора ЛР5, "
                  "окно хранится в `deque(maxlen=window_size)` без накопления всего потока.",
                  f"Дрейф вводится на окне {cfg.drift_start_window} (отсчёт с нуля); "
                  f"порог max(PSI(x1), PSI(x2)) > {cfg.psi_alert_threshold}, "
                  f"или CSI(вероятность) > {cfg.csi_alert_threshold}; "
                  f"срок обнаружения ≤ {cfg.detection_deadline_windows} окон после начала дрейфа.",
                  "Evidently `DataDriftPreset()` рассчитывается **на каждом окне** по "
                  "`x1`, `x2`, `probability`; отдельные HTML-снимки сохраняются для последних окон сценариев.",
                  "", "| Сценарий | Первое окно с алертом | Задержка, окон | Ложных алертов | Вердикт |",
                  "|---|---:|---:|---:|---|",
                  ])
    for _, row in drills.iterrows():
        lines.append(f"| {row['scenario']} | {row['first_alert_window']} | "
                     f"{row['detection_delay_windows']} | {row['false_alerts']} | "
                     f"{'PASS' if row['accepted'] else 'FAIL'} |")
    lines.extend(["", "## 3. Негативный контроль исполнителя", "",
                  "Отказ joblib воспроизводится инъекцией исключения в `Parallel`: сервис "
                  "переходит к последовательному расчёту, не теряя ответ. "
                  "Это **моделирование отказа пула**, а не убийство OS-процесса.",
                  f"Результат: `{json.dumps(summary['worker_failure'], ensure_ascii=False)}`.",
                  "", "## 4. Машиночитаемые доказательства", "",
                  "- `parallel_benchmark.csv` — каждый прогрев/повтор, число ядер, CPU, RSS, задержки, скорость, совпадение результатов.",
                  "- `drift_metrics.csv` — PSI/KS/CSI, доля чужеродных входов, алерты и факт расчёта Evidently для каждого окна.",
                  "- `drift_drills.csv` — контроль чистого потока, сдвига средних и смены доли классов.",
                  "- `evidently_{clean,mean_shift,class_ratio_shift}.html` — визуальные отчёты Evidently.",
                  "- `worker_failure_negative.json`, `summary.json`, `run_meta.json`, `lab7_tests.log` — контроли и трассировка.",
                  "", "## 5. Воспроизводимость и ограничения", "",
                  "Прогон использует фиксированные seed из конфигурации; при повторе совпадают значения генератора "
                  "и метрики окон в пределах численной точности окружения. Производительность зависит от загруженности CPU.",
                  "Все значения получены из сохранённых машинных CSV/JSON. "
                  "Финальные сетевые испытания по настоящим портам нужно показывать отдельно от TestClient.",
                  "", "**Итог:** " + ("все дриллы пройдены." if all(drills["accepted"]) and summary["worker_failure"]["pass"]
                                      else "есть непройденные дриллы; работа не принята."),
                  ""])
    (report / "lab7_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/LAB7")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "demo_output/LAB7_runtime")
    parser.add_argument("--quick", action="store_true", help="Developer smoke only; not publishable evidence")
    args = parser.parse_args()
    if not evidently_available():
        raise RuntimeError("Evidently is mandatory for LAB7; install with pip install -e '.[monitoring]'")
    cfg: Lab7Config = load_lab7_config( ROOT / "configs/lab7.json")  # type: ignore[assignment]
    if args.quick:
        cfg = cfg.model_copy(update={"benchmark_batch_size": 300, "benchmark_repeats": 2,
                                     "reference_size": 1000, "window_size": 400,
                                     "window_step": 200, "n_windows": 5, "drift_start_window": 2})
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    reg_path = build_registry(ROOT, args.runtime_dir / "registry", reset=True)
    app = create_app(reg_path, state=ServiceState())
    client = TestClient(app)
    x, _ = load_xy(ROOT / "data/processed/test.csv")
    benchmark = _benchmark(client, x, cfg, quick=args.quick)
    benchmark.to_csv(report / "parallel_benchmark.csv", index=False, lineterminator="\n")
    reference = generate_dataframe(GeneratorSettings(
        mode="similar", n_samples=cfg.reference_size, seed=cfg.seed, noise=.22,
    ))
    reference["probability"] = _predict_via_api(client, reference)
    all_metrics = []
    drill_rows = []
    html_reports = []
    for scenario in SCENARIOS:
        def evidently_window(name: str, index: int, window: pd.DataFrame) -> dict:
            html = (report / f"evidently_{name}.html") if index == cfg.n_windows - 1 else None
            result = evaluate_evidently_window(reference, window, html_path=html)
            if html is not None:
                html_reports.append(html)
            return result

        metrics, last = collect_window_metrics(cfg, scenario, reference,
                                               lambda frame: _predict_via_api(client, frame),
                                               on_window=evidently_window)
        all_metrics.append(metrics)
        drill_rows.append(summarize_drill(metrics, scenario, cfg))
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(report / "drift_metrics.csv", index=False, lineterminator="\n")
    drills = pd.DataFrame(drill_rows)
    drills.to_csv(report / "drift_drills.csv", index=False, lineterminator="\n")
    failure = _worker_failure(client)
    json_dump(report / "worker_failure_negative.json", failure)
    cpu = max(1, os.cpu_count() or 1)
    best_by_jobs = benchmark.groupby("effective_jobs")["throughput_items_s"].mean()
    summary = {
        "variant": 9,
        "logical_cores": cpu,
        "all_cores_benchmarked": bool((benchmark["effective_jobs"] == cpu).any()) if not args.quick else None,
        "best_effective_jobs": int(best_by_jobs.idxmax()),
        "speedup_max_vs_1": float(best_by_jobs.max() / best_by_jobs.loc[1]),
        "window_count": len(combined), "evidently_evaluations": int(combined["evidently_evaluated"].sum()),
        "scenarios": list(SCENARIOS), "clean_false_alerts": int(drills.loc[drills.scenario == "clean", "false_alerts"].iloc[0]),
        "worker_failure": failure, "all_drills_pass": bool(drills["accepted"].all()),
        "source_model_sha256": sha256_file(ROOT / "reports/LAB2/ml_model.joblib"),
        "config_sha256": sha256_file(ROOT / "configs/lab7.json"),
        "quick": args.quick,
    }
    json_dump(report / "summary.json", summary)
    if not (summary["all_drills_pass"] and failure["pass"]):
        raise AssertionError("LAB7 negative/drift drill failed; inspect drift_drills.csv")
    if not args.quick and not summary["all_cores_benchmarked"]:
        raise AssertionError("LAB7: no full-core benchmark")
    with tracked_run("lab7_parallel_stream_evidently", experiment="lab7_monitoring") as run:
        if run.backend != "mlflow":
            raise RuntimeError("LAB7 needs real MLflow backend, not a fallback")
        _make_report(report, cfg, benchmark, drills, summary, run.run_id, run.backend)
        run.log_params(cfg.model_dump())
        run.log_metrics({
            "best_throughput_items_s": float(best_by_jobs.max()),
            "speedup_max_vs_1": summary["speedup_max_vs_1"],
            "clean_false_alerts": summary["clean_false_alerts"],
            "all_drills_pass": float(summary["all_drills_pass"]),
            "all_cores_benchmarked": float(bool(summary["all_cores_benchmarked"])),
        })
        for artifact in ["parallel_benchmark.csv", "drift_metrics.csv", "drift_drills.csv",
                         "summary.json", "worker_failure_negative.json", "lab7_report.md",
                         *[p.name for p in html_reports]]:
            run.log_artifact(report / artifact)
        json_dump(report / "run_meta.json", {"run_id": run.run_id, "tracking_backend": run.backend})
    print(benchmark.groupby("effective_jobs")["throughput_items_s"].agg(["mean", "std"]).to_string())
    print(drills.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("LAB7 PASSED:", report)


if __name__ == "__main__":
    main()
