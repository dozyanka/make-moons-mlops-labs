# Карта сдачи по лабораторным — вариант 09 make_moons

В финальном разложенном архиве каждая ЛР имеет собственную папку и отдельный ZIP для загрузки. Ниже — минимальный набор, который должен быть виден преподавателю.

| ЛР | Что отправлять/показывать | Главные доказательства |
|---|---|---|
| 1 | отчёт, `eda_stats.csv`, графики, pre-commit/hash logs, manifest, ссылка GitHub | nonlinear scatter, 24 chunks, блокировка секрета/большого файла, data hash |
| 2 | отчёт, `ml_metrics.csv`, обе joblib-модели, hyperparam table, negative logs, pytest coverage | full vs 24 chunks, memory/time, bootstrap CI, leakage control, MLflow run IDs |
| 3 | отчёт, `dl_metrics.csv`, `.pt` checkpoints, loss curves, memory compare, seed/parity logs, pytest coverage | PyTorch full vs stream, reproducibility, CPU/GPU parity при наличии CUDA |
| 4 | отчёт, validator metrics/params, `validator_negative.log`, comparison, pytest coverage | OOD reject recall, false reject rate, разумный `reason` для отказа, full≈chunks |
| 5 | отчёт, synthetic metrics/config, reproducibility, live update log, pytest coverage | PATCH config без рестарта, similar/random/drifted, seed reproducibility |
| 6 | отчёт, registry, service logs, endpoint contracts/load, promotion drill, monitor snapshot, C4 diagram, pytest coverage | 422, promote/rollback, zero lost requests, runtime config, понятные metrics |
| 7 | отчёт, parallel benchmark, drift metrics/drills, worker failure, Evidently status/report, pytest coverage | все CPU cores, throughput/CPU/RSS, drift detected, no false alerts |
| 8 | отчёт, `end_to_end.csv`, alerts, install log, manifest, defense checklist, all tests, review+comment | clean install, repeat real/synthetic, alerts, project-wide pytest/coverage |

## Общие вложения

- ссылка на GitHub-репозиторий;
- скринкаст 3–5 минут по `reports/SCREENCAST_PLAN.md`;
- `reports/llm_review.md` и `reports/llm_review_comment.md`;
- `defense_checklist.md`;
- при сдаче ЛР6/итоговой — `diagrams/c4_make_moons_architecture.png` и `.drawio`.
