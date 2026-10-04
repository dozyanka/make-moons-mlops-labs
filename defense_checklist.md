# Defense checklist — вариант 09

Исходные данные SHA-256: `1e23d56137d9a39b1c1ae61dc067319c14f6591da41b6b9d6873e72d7366e981`. Кодовый commit, использованный при формировании отчётов: `d5a60b7b88d31aff32ce8fd1254d39fcfd60cc21`.

| ЛР | Обязательные артефакты | Хеш/воспроизводимость | Негативные контроли | Статус перед финальной сдачей |
|---|---|---|---|---|
| 1 | report, eda_stats, precommit log, hash manifest | SHA-256 сверяется | secret/large/syntax/tamper | готово |
| 2 | report, ml_metrics, ml_model, config log | full=chunks, seed fixed | bad configs, leakage-safe split | rerun with real MLflow online |
| 3 | report, dl_metrics, dl_model, memory compare | repeat exact, tolerance <=0.005 | bad configs, seed demo | CUDA parity rerun if GPU available |
| 4 | report, validator metrics/params/negative log | full≈chunks ~1e-14 | OOD rejection + bad configs | готово |
| 5 | report, synthetic metrics/config/tests | identical hash for same seed | drift knobs + bad config | готово |
| 6 | report, service log, registry | model SHA-256 checked | 422, bad hash, rollback | готово |
| 7 | report, benchmark, drift metrics | repeated warmed measurements | drift/clean/worker failure | rerun with Evidently online |
| 8 | report, alerts, install log, manifest | e2e repeats exact | invalid config, alert drills | готово после online dependency run |

## Раскрытие использования ИИ

ИИ использовался для проектирования структуры репозитория, генерации черновиков кода, тестов и отчётов. Все численные результаты в отчётах получены исполняемыми скриптами и сохранены в CSV/JSON/логах; они не были придуманы вручную. Перед сдачей обучающийся должен самостоятельно просмотреть код, выполнить `RUN_ME_WINDOWS.ps1`, проверить итоговые логи и записать скринкаст.

## Раскрытие использования ИИ для ЛР1

Для ЛР1 использовался ChatGPT как консультант и инструмент подготовки черновиков кода, PowerShell-команд и текста отчёта. Все команды запускались локально, результаты EDA, тестов, хешей и негативных контролей были получены фактическим выполнением проекта на компьютере обучающегося. Итоговые результаты проверены локальными тестами и pre-commit.
