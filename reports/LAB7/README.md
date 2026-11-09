# ЛР7 — доказательства проверки

`lab7_report.md`, `parallel_benchmark.csv`, `drift_metrics.csv`, `drift_drills.csv`,
`evidently_clean.html`, `evidently_mean_shift.html`,
`evidently_class_ratio_shift.html`, `worker_failure_negative.json`,
`run_meta.json`, `summary.json` и `lab7_tests.log` автоматически получаются
при запуске `scripts/run_lab7.py` и pytest через установщик №39.

Для воспроизведения в Windows запускайте пункт **20** в `RUN_DEMO.cmd`.
Новый прогон записывается в `demo_output/LAB7` без изменения опубликованного
комплекта. Номер прогона MLflow сохранён в `run_meta.json`.

**Важно:** измерение `TestClient` не равно внешнему сетевому стресс-тесту;
отказ исполнителя имитируется исключением joblib, а не завершением процесса ОС.
Машинно полученные измерения сохраняются без ручного редактирования.
