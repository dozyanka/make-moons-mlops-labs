# Defense checklist — вариант 09

Актуальные данные рабочего стола: SHA-256 исходного CSV `dd38bd033f2300777fb0420e4c7bf895946435fa3b76c4d495a69e120b54be91`. Ранее сформированные отчёты могут ссылаться на другие коммиты; для ЛР4 актуальный базовый хеш указан в `reports/LAB4/lab4_report.md`.

| ЛР | Обязательные артефакты | Хеш/воспроизводимость | Негативные контроли | Статус перед финальной сдачей |
|---|---|---|---|---|
| 1 | report, eda_stats, precommit log, hash manifest | SHA-256 сверяется | secret/large/syntax/tamper | готово |
| 2 | report, ml_metrics, ml_model, config log | full=chunks, seed fixed | bad configs, leakage-safe split | rerun with real MLflow online |
| 3 | report, dl_metrics, dl_model, memory compare | repeat exact, tolerance <=0.005 | bad configs, seed demo | CUDA parity rerun if GPU available |
| 4 | report, validator metrics/params/negative log | full≈chunks, NPZ reload, SHA-256 | mean shift/outliers/categories/claimed-label-flip, bad configs | DEMO 10-13 и MLflow проверены на Windows; pytest 36 passed |
| 5 | report, synthetic metrics/config/tests | identical hash for same seed | drift knobs + bad config | ещё не проверено в текущем репозитории |
| 6 | report, service log, registry | model SHA-256 checked | 422, bad hash, rollback | ещё не проверено в текущем репозитории |
| 7 | report, benchmark, drift metrics | repeated warmed measurements | drift/clean/worker failure | ещё не проверено в текущем репозитории |
| 8 | report, alerts, install log, manifest | e2e repeats exact | invalid config, alert drills | ещё не проверено в текущем репозитории |
