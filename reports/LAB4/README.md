# ЛР4 — сохранённые результаты валидатора

Основной отчёт: [`lab4_report.md`](lab4_report.md).

| Файл | Назначение |
|---|---|
| [`lab4_report.md`](lab4_report.md) | Условия, full/chunks, пороги, таблицы, MLflow Run ID, выводы |
| [`validator_metrics.csv`](validator_metrics.csv) | Precision, Recall, false reject rate, ROC-AUC, время и память по режимам |
| `validator_params.npz` | NumPy-параметры режима full (SHA-256 в отчёте) |
| `validator_params_chunk.npz` | NumPy-параметры режима chunks |
| [`validator_negative.log`](validator_negative.log) | Ожидаемые и фактические результаты чужеродных входов |
| [`negative_summary.json`](negative_summary.json) | Сводка с явно отмеченными ошибочными приёмами перевёрнутых меток |
| [`comparison.json`](comparison.json) | Сравнение статистик и метрик full/chunks |
| [`repeat_verification.json`](repeat_verification.json) | Независимые повторы и SHA-256 сохранённых NPZ |
| [`reproducibility.json`](reproducibility.json) | Проверка воспроизводимости |
| [`config_negative_test.log`](config_negative_test.log) | Невалидные Pydantic-конфигурации |
| [`pytest_coverage.log`](pytest_coverage.log) | Прогон pytest и покрытие |
| [`coverage_details.json`](coverage_details.json) | Машиночитаемое покрытие |
| [`source_manifest.json`](source_manifest.json) | Git-коммит исходного кода и хеши исходников |

**Важно:** `mlruns/` локальный и не коммитится в Git. Сохранённые Run ID доступны в отчёте; новые появляются после запуска ЛР4 на компьютере проверяющего.

Воспроизведение через [общее меню](../../DEMO/README.md) — пункт **10** — сохраняет новые артефакты отдельно в `demo_output/LAB4` и восстанавливает эти опубликованные файлы. Для пяти готовых примеров входов используйте пункт **11**.
