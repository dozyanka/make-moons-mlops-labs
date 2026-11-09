# ЛР7 — параллельный инференс и оконный Evidently

Вариант 09: `make_moons`. Реализованы `joblib`/`/predict/batch`, сравнение
1 / 2 / 4 / все логические ядра, прогрев и повторные измерения,
ограниченное окно с параметрами `window_size`/`window_step`,
PSI/KS и CSI для предсказаний, Evidently `DataDriftPreset()` на каждом окне,
два дрейфовых дрилла и негативный контроль отказа пула.

Прогон для преподавателя: `RUN_DEMO.cmd` → **20**. Отчёт: **21**.
Или напрямую: `.venv\\Scripts\\python.exe scripts\\run_lab7.py --report-dir demo_output\\LAB7`.

Тесты: `.venv\\Scripts\\python.exe -m pytest -q tests/test_lab7_complete.py tests/test_drift.py`.
Исходные артефакты лабораторных 1–6 не модифицируются.
