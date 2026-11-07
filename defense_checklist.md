# Defense checklist — вариант 09

Актуальные данные рабочего стола: SHA-256 исходного CSV `dd38bd033f2300777fb0420e4c7bf895946435fa3b76c4d495a69e120b54be91`. Ранее сформированные отчёты могут ссылаться на другие коммиты; для ЛР4 актуальный базовый хеш указан в `reports/LAB4/lab4_report.md`.

| ЛР | Обязательные артефакты | Хеш/воспроизводимость | Негативные контроли | Статус перед финальной сдачей |
|---|---|---|---|---|
| 1 | report, eda_stats, precommit log, hash manifest | SHA-256 сверяется | secret/large/syntax/tamper | готово |
| 2 | report, ml_metrics, ml_model, config log | full=chunks, seed fixed | bad configs, leakage-safe split | rerun with real MLflow online |
| 3 | report, dl_metrics, dl_model, memory compare | repeat exact, tolerance <=0.005 | bad configs, seed demo | CUDA parity rerun if GPU available |
| 4 | report, validator metrics/params/negative log | full≈chunks, NPZ reload, SHA-256 | mean shift/outliers/categories/claimed-label-flip, bad configs | реализовано; MLflow/скринкаст проверить на ПК |
| 5 | report, synthetic metrics/config/tests | identical hash for same seed | drift knobs + bad config | ещё не проверено в текущем репозитории |
| 6 | report, service log, registry | model SHA-256 checked | 422, bad hash, rollback | ещё не проверено в текущем репозитории |
| 7 | report, benchmark, drift metrics | repeated warmed measurements | drift/clean/worker failure | ещё не проверено в текущем репозитории |
| 8 | report, alerts, install log, manifest | e2e repeats exact | invalid config, alert drills | ещё не проверено в текущем репозитории |

## Раскрытие использования ИИ

ИИ использовался для проектирования структуры репозитория, генерации черновиков кода, тестов и отчётов. Все численные результаты в отчётах получены исполняемыми скриптами и сохранены в CSV/JSON/логах; они не были придуманы вручную. Перед сдачей обучающийся должен самостоятельно просмотреть код, выполнить `RUN_ME_WINDOWS.ps1`, проверить итоговые логи и записать скринкаст.

## Раскрытие использования ИИ для ЛР1

Для ЛР1 использовался ChatGPT как консультант и инструмент подготовки черновиков кода, PowerShell-команд и текста отчёта. Все команды запускались локально, результаты EDA, тестов, хешей и негативных контролей были получены фактическим выполнением проекта на компьютере обучающегося. Итоговые результаты проверены локальными тестами и pre-commit.

## Раскрытие использования ИИ для ЛР4

ChatGPT (GPT-6) применялся для реализации и тестирования валидатора, генерации кода отчёта и инструкции запуска. Математический метод — проверяемое расстояние Махаланобиса и правило confidence; все численные результаты формируются исполняемым кодом. Локальные прогоны без установленного MLflow помечены `offline_diagnostic` и не считаются регистрацией в MLflow. Финальная MLflow-регистрация, запись скринкаста и критическая рецензия подтверждаются студентом после проверки на своём ПК.
