"""LAB6 full drill in an isolated output directory (does not touch LAB1–5)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab6Config, load_config
from moons_lab.lab6_bootstrap import build_registry
from moons_lab.monitor_service import MonitorState, Observation, create_monitor_app
from moons_lab.registry import ModelRegistry
from moons_lab.service import ServiceState, create_app
from moons_lab.tracking import tracked_run
from moons_lab.utils import json_dump, sha256_file


def contract_checks(client: TestClient, monitor: TestClient) -> pd.DataFrame:
    cases = [
        ("health", "GET", "/health", None, 200),
        ("predict_valid", "POST", "/predict", {"x1": .2, "x2": .1}, 200),
        ("predict_missing", "POST", "/predict", {"x1": .2}, 422),
        ("predict_wrong_type", "POST", "/predict", {"x1": "bad", "x2": .1}, 422),
        ("predict_numeric_string", "POST", "/predict", {"x1": "0.2", "x2": .1}, 422),
        ("predict_out_of_range", "POST", "/predict", {"x1": 999, "x2": .1}, 422),
        ("predict_nonfinite", "POST", "/predict", {"x1": 1e100, "x2": .1}, 422),
        ("predict_extra", "POST", "/predict", {"x1": .2, "x2": .1, "fake": 1}, 422),
        ("batch", "POST", "/predict/batch", {"items": [{"x1": .2, "x2": .1}]}, 200),
        ("metrics", "GET", "/metrics", None, 200),
        ("models", "GET", "/models", None, 200),
        ("runtime", "GET", "/runtime", None, 200),
        ("runtime_patch", "PATCH", "/runtime", {"metrics_enabled": False}, 200),
        ("unknown_version", "POST", "/promote", {"version": "missing"}, 404),
        ("stage_preparing", "PATCH", "/models/v1/status", {"status": "preparing"}, 200),
        ("reject_preparing", "POST", "/promote", {"version": "v1"}, 409),
        ("stage_canary", "PATCH", "/models/v1/status", {"status": "canary"}, 200),
    ]
    rows = []
    for name, method, uri, payload, expected in cases:
        response = client.request(method, uri, json=payload)
        rows.append({"case": name, "method": method, "path": uri, "expected_status": expected,
                     "actual_status": response.status_code, "pass": response.status_code == expected})
    client.patch("/runtime", json={"metrics_enabled": True})
    r = monitor.get("/health")
    rows.append({"case": "monitor_health", "method": "GET", "path": "monitor:/health",
                 "expected_status": 200, "actual_status": r.status_code, "pass": r.status_code == 200})
    r = monitor.get("/metrics")
    rows.append({"case": "monitor_metrics", "method": "GET", "path": "monitor:/metrics",
                 "expected_status": 200, "actual_status": r.status_code, "pass": r.status_code == 200})
    return pd.DataFrame(rows)


def endpoint_load(client: TestClient, monitor: TestClient, n: int) -> pd.DataFrame:
    cases = [
        ("/health", lambda: client.get("/health")),
        ("/predict", lambda: client.post("/predict", json={"x1": .2, "x2": .1})),
        ("/predict/batch", lambda: client.post("/predict/batch", json={"items": [{"x1": .2, "x2": .1}]})),
        ("/metrics", lambda: client.get("/metrics")),
        ("/models", lambda: client.get("/models")),
        ("/runtime", lambda: client.get("/runtime")),
        ("/promote", lambda: client.post("/promote", json={"version": "v2", "reason": "load"})),
        ("/rollback", lambda: client.post("/rollback")),
        ("monitor:/health", lambda: monitor.get("/health")),
        ("monitor:/metrics", lambda: monitor.get("/metrics")),
        ("monitor:/plot.png", lambda: monitor.get("/plot.png")),
    ]
    results = []
    for name, call in cases:
        start = time.perf_counter()
        codes = [call().status_code for _ in range(n)]
        elapsed = time.perf_counter() - start
        results.append({"endpoint": name, "requests": n, "successes": sum(c == 200 for c in codes),
                        "errors": sum(c != 200 for c in codes), "elapsed_s": elapsed,
                        "rps": n / max(elapsed, 1e-9)})
    # Known production version after control-drill.
    if client.get("/models").json()["active"] != "v2":
        client.post("/promote", json={"version": "v2", "reason": "after load"})
    return pd.DataFrame(results)


def switch_under_load(client: TestClient, workers: int = 8, per_worker: int = 120) -> dict:
    statuses, versions = [], []
    lock = threading.Lock()
    start = threading.Event()
    def worker(w: int) -> None:
        start.wait()
        for i in range(per_worker):
            response = client.post("/predict", json={"x1": .15 + i * .0001, "x2": .18})
            with lock:
                statuses.append(response.status_code)
                if response.status_code == 200:
                    versions.append(response.json()["model_version"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(worker, i) for i in range(workers)]
        start.set()
        time.sleep(.03)
        promote = client.post("/promote", json={"version": "v1", "reason": "switch under load"})
        time.sleep(.08)
        restored = client.post("/rollback")
        for task in tasks:
            task.result()
    count = workers * per_worker
    return {"expected_requests": count, "total_requests": len(statuses),
            "successful_requests": sum(code == 200 for code in statuses),
            "lost_or_failed": count - sum(code == 200 for code in statuses),
            "success_fraction": sum(code == 200 for code in statuses) / count,
            "versions_observed": sorted(set(versions)),
            "promote_status": promote.status_code,
            "rollback_status": restored.status_code,
            "active_after_rollback": client.get("/models").json()["active"]}


def hash_negative_control(root: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="lab6-bad-hash-") as tmp:
        registry_file = build_registry(root, Path(tmp) / "registry")
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
        candidate = registry_file.parent / payload["versions"]["v1"]["path"]
        with candidate.open("ab") as file:
            file.write(b"tamper-for-negative-control")
        try:
            ModelRegistry(registry_file)
        except ValueError as exc:
            if "hash mismatch" in str(exc):
                return "PASS: corrupted candidate refused (SHA-256 mismatch)"
            raise
        raise AssertionError("corrupted model was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/LAB6")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "demo_output/LAB6_runtime")
    parser.add_argument("--quick", action="store_true", help="use fewer requests for developer smoke test")
    args = parser.parse_args()
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    cfg: Lab6Config = load_config(6, ROOT / "configs/lab6.json")  # type: ignore[assignment]
    registry_path = build_registry(ROOT, args.runtime_dir / "registry", reset=True)  # clean test run, not production
    monitor_state = MonitorState(log_path=args.runtime_dir / "monitor_observations.jsonl")
    monitor = TestClient(create_monitor_app(monitor_state))
    sink = lambda x1, x2, p: monitor_state.add(Observation(x1=x1, x2=x2, probability=p))
    service = TestClient(create_app(registry_path, state=ServiceState(), monitor_sink=sink))

    contracts = contract_checks(service, monitor)
    load = endpoint_load(service, monitor, 15 if args.quick else max(100, cfg.load_test_requests))
    drill = switch_under_load(service, per_worker=30 if args.quick else 120)
    monitor_state.save_plot(report / "monitor_snapshot.png")
    negative = hash_negative_control(ROOT)
    monitor_metrics = monitor.get("/metrics").json()
    registry_snapshot = service.get("/models").json()
    service_metrics = service.get("/metrics").json()
    contract_count = int(contracts["pass"].sum())
    contracts.to_csv(report / "endpoint_contracts.csv", index=False)
    load.to_csv(report / "endpoint_load.csv", index=False, float_format="%.9f")
    json_dump(report / "promotion_drill.json", drill)
    json_dump(report / "registry.json", registry_snapshot)
    json_dump(report / "monitor_metrics.json", monitor_metrics)
    json_dump(report / "service_metrics.json", service_metrics)
    (report / "hash_negative_control.log").write_text(negative + "\n", encoding="utf-8")
    log = [f"contracts={contract_count}/{len(contracts)}", f"load_requests={int(load['requests'].sum())}",
           f"load_errors={int(load['errors'].sum())}",
           f"promote_rollback_success={drill['promote_status'] == 200 and drill['rollback_status'] == 200}",
           f"drill_requests={drill['total_requests']}", f"drill_errors={drill['lost_or_failed']}",
           f"drill_fraction={drill['success_fraction']:.6f}",
           f"versions_observed={drill['versions_observed']}",
           f"monitor_records={monitor_metrics['count']}", negative]
    (report / "service_tests.log").write_text("\n".join(log) + "\n", encoding="utf-8")

    if (not contracts["pass"].all() or int(load["errors"].sum()) or drill["lost_or_failed"]
            or drill["success_fraction"] < .999 or drill["active_after_rollback"] != "v2"):
        raise AssertionError("LAB6 contract, load, switch or rollback failed")
    with tracked_run("lab6_model_service", experiment="lab6_service") as run:
        run.log_params({**cfg.model_dump(), "versions": "v1,v2", "mode": "TestClient in-process drill"})
        run.log_metrics({"contracts_pass_rate": contract_count / len(contracts),
                         "load_error_count": int(load["errors"].sum()),
                         "drill_success_fraction": drill["success_fraction"],
                         "drill_lost_requests": drill["lost_or_failed"]})
        run.log_artifact(report / "service_tests.log")
        json_dump(report / "run_meta.json", {"run_id": run.run_id, "tracking_backend": run.backend})
    run_meta = json.loads((report / "run_meta.json").read_text(encoding="utf-8"))
    (report / "lab6_report.md").write_text(f"""# ЛР6. FastAPI-сервис модели с бесшовным переключением

**Вариант:** 09 — make_moons. **Автор:** Владимир Шаталов, ЕТ-122.

## 1. Архитектура и версии

Отдельный model-serving FastAPI и независимый monitoring FastAPI. Реестр содержит
**v1** (линейный baseline ЛР2) и **v2** (нелинейная ML-модель ЛР2). Артефакты
копируются побайтово в изолированный runtime. Исходные файлы ЛР1–5 не изменяются.
При смене версии модель проверяется по SHA-256 и загружается до изменения
активного указателя. Заблокированные на короткое время запросы продолжаются.

| Версия | Test accuracy ЛР2 | SHA-256 | Итоговый статус |
|---|---:|---|---|
| v1 | {registry_snapshot['versions']['v1']['metrics']['test_accuracy']:.6f} | `{registry_snapshot['versions']['v1']['sha256']}` | {registry_snapshot['versions']['v1']['status']} |
| v2 | {registry_snapshot['versions']['v2']['metrics']['test_accuracy']:.6f} | `{registry_snapshot['versions']['v2']['sha256']}` | {registry_snapshot['versions']['v2']['status']} |

Источник чисел: `reports/LAB2/comparison.json`, `registry.json`.

## 2. Контракты API

| Адрес | Метод | Назначение |
|---|---|---|
| `/health` | GET | Состояние сервиса и активная версия |
| `/predict` | POST | Одна точка, ответ prediction/probability/version |
| `/predict/batch` | POST | Пакет точек |
| `/metrics` | GET | Ошибки, задержка, p95, uptime |
| `/models` | GET | Реестр версий с хешами и статусами |
| `/promote` | POST | Переключить на проверенную версию |
| `/rollback` | POST | Откат на предыдущую версию |
| `/runtime` | GET, PATCH | Изменить параметры без перезапуска |

Неполные, нечисловые и выходящие за пределы диапазона данные получают **422**.
Подробные случаи: `endpoint_contracts.csv` ({contract_count}/{len(contracts)} пройдено).

## 3. Проверка нагрузки и переключения

| Показатель | Значение |
|---|---:|
| Контрактные проверки | {contract_count}/{len(contracts)} |
| Нагрузка на каждый endpoint | {int(load['requests'].min())} запросов |
| Всего нагрузочных запросов | {int(load['requests'].sum())} |
| Ошибок нагрузки | {int(load['errors'].sum())} |
| Запросов при переключении | {drill['total_requests']} |
| Успешных при переключении | {drill['successful_requests']} |
| Доля успеха | {drill['success_fraction']:.6%} |
| Потерянных | {drill['lost_or_failed']} |
| Активная после отката | {drill['active_after_rollback']} |

Доказательства: `endpoint_load.csv`, `promotion_drill.json`, `service_tests.log`.
Измерение выполнено внутри одного процесса через FastAPI TestClient; внешний
сетевой стресс-тест на развёрнутых Uvicorn-процессах этим не заменяется.

## 4. Мониторинг и негативные контроли

Независимый monitoring FastAPI: `/health`, `/ingest`, `/metrics`, `/plot.png`.
Принято {monitor_metrics['count']} наблюдений. Снимок: `monitor_snapshot.png`;
числа: `monitor_metrics.json`; структурированный журнал формируется в runtime.

Хеш-негативный контроль: **{negative}**. Проверки 422 и неизвестной версии
в `endpoint_contracts.csv`. Дополнительные модульные тесты — `tests/test_lab6_service.py`.

## 5. MLflow / журнал прогона

Run ID: `{run_meta['run_id']}`. Backend: `{run_meta['tracking_backend']}`.
Параметры, метрики и `service_tests.log` записаны через `tracked_run()`.

## 6. Выводы

1. Два действительных артефакта ЛР2 зарегистрированы с SHA-256.
2. Схемы запросов отвергают невалидные признаки с HTTP 422.
3. Переключение и откат прошли под конкурентными запросами без потерь.
4. Независимый монитор собирает наблюдения и строит график.
5. Негативный тест повреждённой модели обнаруживает несовпадение SHA-256.
""", encoding="utf-8")
    # Append provenance to the newly generated report without changing the original
    # published LAB1–5 artifacts.  The Git SHA names the source snapshot being run.
    import subprocess
    from datetime import datetime, timezone
    git_state = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True, check=False)
    source_commit = git_state.stdout.strip() if git_state.returncode == 0 else "no-git-checkout"
    training = ROOT / "data/processed/train.csv"
    training_hash = sha256_file(training) if training.is_file() else "not available"
    with (report / "lab6_report.md").open("a", encoding="utf-8") as out:
        out.write(f"\n## 7. Воспроизводимость и уточнения после аудита\n\n"
                  f"Дата UTC: {datetime.now(timezone.utc).isoformat()}. "
                  f"Исходный коммит: `{source_commit}`.\n\n"
                  f"SHA-256 `data/processed/train.csv`: `{training_hash}`.\n\n"
                  "Проверки TestClient и сетевой Uvicorn-нагрузки являются разными "
                  "сценариями; внешний тест: `scripts/lab6_network_drill.py`.\n\n"
                  "Реестр сохраняет состояние между обычными запусками; "
                  "этот лабораторный прогон явно сбрасывает только свою копию реестра.\n")
    print(contracts.to_string(index=False))
    print(load.to_string(index=False))
    print(json.dumps(drill, indent=2, ensure_ascii=False))
    print(negative)
    print(f"LAB6 PASSED: {contract_count} contracts, {int(load['requests'].sum())} load requests, "
          f"{drill['total_requests']} switching requests, no losses; report={report}")


if __name__ == "__main__":
    main()
