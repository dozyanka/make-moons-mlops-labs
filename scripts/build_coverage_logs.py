from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    2: {
        "tests": ["test_ml.py", "test_metrics.py", "test_data.py", "test_config.py", "test_tracking_utils.py"],
        "modules": ["ml_pipeline.py", "metrics.py", "data.py", "config.py", "tracking.py", "utils.py"],
        "note": "MLflow online-backend не принудительно тестируется в offline runner; обучение full/chunks, сериализация, утечка, метрики, конфиги и fallback tracking покрыты pytest.",
    },
    3: {
        "tests": ["test_dl.py", "test_metrics.py", "test_data.py", "test_config.py"],
        "modules": ["dl_pipeline.py", "metrics.py", "data.py", "config.py"],
        "note": "CUDA-ветвь device parity невозможно исполнить на CPU-only runner; CPU full/stream, checkpoint, seed, данные и метрики покрыты pytest.",
    },
    4: {
        "tests": ["test_validator.py", "test_config.py"],
        "modules": ["validator.py", "config.py"],
        "note": "Проверены full/chunks статистики, сериализация, OOD rejection и интерпретируемое объяснение решения validator.explain().",
    },
    5: {
        "tests": ["test_generator.py", "test_config.py"],
        "modules": ["generator.py", "generator_service.py", "config.py"],
        "note": "Проверены схема, seed, similar/random, каждая drift-ручка и online PATCH без перезапуска процесса.",
    },
    6: {
        "tests": ["test_service.py", "test_monitor.py", "test_config.py"],
        "modules": ["service.py", "registry.py", "monitor_service.py", "config.py"],
        "note": "Внешний TCP-процесс uvicorn не поднимается в unit-тестах: API проверяется FastAPI TestClient. Контракты, registry, promote/rollback, hash rejection, runtime settings и monitoring state покрыты.",
    },
    7: {
        "tests": ["test_service.py", "test_drift.py", "test_monitor.py", "test_config.py"],
        "modules": ["service.py", "drift.py", "monitor_service.py", "config.py"],
        "note": "Реальный crash ОС-процесса не инициируется: отказ joblib пула моделируется mock и проверяется serial fallback. Drift-метрики и batch API покрыты.",
    },
}


def coverage_rows(full_log: str, modules: list[str]) -> list[str]:
    rows = []
    for line in full_log.splitlines():
        if any(f"/{name}" in line or line.strip().startswith(f"src/moons_lab/{name}") for name in modules):
            rows.append(line.rstrip())
    return rows


def main() -> None:
    log_path = ROOT / "reports/LAB8/all_tests.log"
    full_log = log_path.read_text(encoding="utf-8")
    total = next((ln for ln in full_log.splitlines() if ln.startswith("TOTAL")), "TOTAL: see all_tests.log")
    passed = next((ln for ln in full_log.splitlines() if " passed in " in ln), "pytest result: see all_tests.log")

    for lab, case in CASES.items():
        rows = coverage_rows(full_log, case["modules"])
        text = [
            f"LAB{lab} pytest/coverage scope",
            "",
            "Тестовые файлы:",
            *[f"- tests/{name}" for name in case["tests"]],
            "",
            "Строки покрытия соответствующих модулей из полного pytest-прогона:",
            "Name                                 Stmts   Miss Branch BrPart  Cover   Missing",
            *rows,
            "",
            f"Project total: {total}",
            f"Project test result: {passed}",
            "",
            "Пояснение по интеграционным/непокрываемым ветвям:",
            case["note"],
            "",
            "Полный исходный coverage-лог: reports/LAB8/all_tests.log",
            "",
        ]
        (ROOT / f"reports/LAB{lab}/pytest_coverage.log").write_text("\n".join(text), encoding="utf-8")

    (ROOT / "reports/LAB8/testing_scope.md").write_text(
        "# Pytest scope for LAB8\n\n"
        "ЛР8 использует полный набор `tests/`; численный результат и project-wide coverage находятся в `all_tests.log`. "
        "CLI-процессы, внешний uvicorn, CUDA и online-only backends (MLflow/Evidently) относятся к интеграционным проверкам и отдельно документированы в отчёте.\n",
        encoding="utf-8",
    )
    print("per-lab coverage logs built from full pytest log")


if __name__ == "__main__":
    main()
