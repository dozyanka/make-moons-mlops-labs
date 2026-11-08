# ЛР6 — FastAPI-сервис модели с переключением и независимым мониторингом

Вариант 09. Используются **две настоящие ранее обученные модели ЛР2**:
`reports/LAB2/linear_baseline.joblib` (v1) и `reports/LAB2/ml_model.joblib` (v2).
Вся запись выполняется только в `demo_output/` или специально выбранную папку.

## Проверка и сдача

```powershell
.\RUN_DEMO.cmd
# 16 — полный прогон ЛР6; 19 — отчёт
```

Или напрямую:

```powershell
.\.venv\Scripts\python.exe scripts/run_lab6.py --report-dir demo_output/LAB6 --runtime-dir demo_output/LAB6_runtime
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py tests/test_lab6_service.py
```

Прогон выполняет контрактные проверки, **не менее 100 запросов к каждому основному адресу**, одновременные предсказания при promote/rollback, проверку повреждённой модели, проверку отдельного мониторинга и сохранение отчётов. `TestClient` — внутри процесса; настоящие сетевые серверы запускаются ниже.

## Запуск двух независимых серверов

Терминал 1 — мониторинг:

```powershell
.\.venv\Scripts\python.exe scripts/serve_lab6.py --service monitor --port 8001
```

Терминал 2 — ML API и отправка наблюдений в мониторинг:

```powershell
.\.venv\Scripts\python.exe scripts/serve_lab6.py --service model --port 8000 --monitor-url http://127.0.0.1:8001
```

Интерактивная документация: http://127.0.0.1:8000/docs и http://127.0.0.1:8001/docs.
Пример JSON для `POST /predict`: `{"x1":0.2,"x2":0.1}`.
Для отката: `POST /rollback` без тела. Продвижение версии:
`POST /promote` c `{"version":"v1","reason":"demonstration"}`.

Реестр версий пересоздаётся только в изолированной папке `demo_output/LAB6_runtime/registry/`. Исходные артефакты ЛР1–5 и контрольные SHA-256 ЛР4 не изменяются.

## Файлы

- `src/moons_lab/registry.py`: реестр, контроль SHA-256, загрузка до атомарного переключения, история.
- `src/moons_lab/service.py`: FastAPI, Pydantic, метрики, promote/rollback, sink мониторинга.
- `src/moons_lab/monitor_service.py`: отдельное FastAPI-приложение, текстовый JSONL-лог, PNG-график.
- `src/moons_lab/lab6_bootstrap.py`: два оригинальных артефакта ЛР2 → runtime.
- `scripts/run_lab6.py`: тестовый сценарий и воспроизводимый отчёт.
- `scripts/serve_lab6.py`: запуск двух отдельных Uvicorn-процессов.
- `reports/LAB6/`: опубликованные результаты; `demo_output/LAB6/` — новые запуски.

Результаты сетевой эксплуатации Uvicorn не выдаются за результаты `TestClient`: это две разные методики проверки.
