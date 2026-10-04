# Дополнительные требования с лекции — где они реализованы

Этот файл фиксирует требования, записанные на лекции, и конкретное место в варианте 09 `make_moons`.

| Требование с лекции | Где выполнено | Что показать |
|---|---|---|
| ЛР2: обучение на полном датасете и на >10 чанках, сохранить и продумать хранение | `data/chunks/` содержит 24 CSV; `run_lab2.py`; `ml_model.joblib`, `ml_model_chunk.joblib`; `ml_metrics.csv` | таблицу full/chunks, список 24 чанков, загрузку сохранённой модели |
| ЛР3: то же самое на PyTorch | `run_lab3.py`, `IterableDataset/DataLoader`, `dl_model.pt`, `dl_model_chunk.pt` | full/stream, loss curves, memory comparison |
| Нужен код, который запускается и собирает модель | `scripts/run_lab2.py`, `run_lab3.py`, CLI `train_ml`, `train_dl`, `RUN_ME_WINDOWS.ps1` | запустить команду и показать созданный model file |
| ЛР4: простой model-validator; невалидные данные должны получать разумный ответ | `ValidatorModel.explain()`, `validator_negative.log` | причины `too_far...` / `low_model_confidence`, expected vs actual |
| ЛР5: синтетические данные и изменение параметров online без рестарта | `generator_service.py`: GET/PATCH `/config`, POST `/generate`; `live_update_demo.log` | PATCH shift/noise и сразу новый POST generate в том же процессе |
| С ЛР2 код покрывается pytest; непокрываемое объяснить | `tests/`, `reports/LAB2..LAB7/pytest_coverage.log`, `LAB8/all_tests.log` | coverage + пояснения CUDA/uvicorn/MLflow/Evidently |
| ЛР6: сервис; runtime изменения; логи/метрики; понятные параметры инженеру и аналитику | `/runtime`, `/metrics`, отдельный monitor service, `metric_descriptions`, model registry | uptime, error_rate, latency, online accuracy, promote/rollback |
| ЛР7: распараллеливание на всех ядрах CPU | `/predict/batch`; benchmark автоматически добавляет `os.cpu_count()` и пишет `uses_all_available_cores` | строку benchmark с effective_jobs = cpu_count |
| ЛР8: запустить и проверить всё; максимизировать requests/s; объяснить предел; учитывать время/CPU/память | `parallel_benchmark.csv`, `end_to_end.csv`, `system_resources.json`, отчёт ЛР8 | лучший throughput и объяснение overhead, CPU/RSS/elapsed |
| Все ЛР: репозиторий, отчёт, скринкаст, неправильные режимы | Git/CI, `labN_report.md/.docx`, `SCREENCAST_PLAN.md`, negative logs | GitHub URL, 3–5 мин видео, один negative case |
| Рецензия модели + критический комментарий | `reports/llm_review.md`, `reports/llm_review_comment.md` | оба файла приложить к итоговой сдаче |

## Важное перед сдачей

1. Выполнить полный прогон на своей машине с интернетом, чтобы получить настоящий MLflow/Evidently.
2. Если есть CUDA — повторить ЛР3 для фактического CPU/GPU parity. Если CUDA нет, оставить честное объяснение в отчёте.
3. Вставить URL своего GitHub-репозитория во все отчёты: `python scripts/set_repo_url.py https://github.com/USER/REPO`.
4. Не удалять CSV/JSON/log-файлы: численные выводы в отчётах должны подтверждаться машиночитаемыми артефактами.
