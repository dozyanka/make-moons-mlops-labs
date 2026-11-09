# ЛР8 — вариант 09 `make_moons`

Работа развивается **только поверх** публикации ЛР7 `b9357cf`.
Ранее опубликованные артефакты и зафиксированный `src/moons_lab/config.py` не изменяются.

## Проверки

Короткий запуск (не является полным доказательством):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_lab8_full.py --cov=moons_lab.lab8_config --cov=moons_lab.lab8_observability --cov=moons_lab.lab8_live --cov-report=term-missing
.\.venv\Scripts\python.exe scripts/run_lab8.py --quick
```

**Полный прогон** (настоящие часы, не ускоренная симуляция):

```powershell
.\.venv\Scripts\python.exe scripts/run_lab8.py --full --install-check
```

Проводятся: 5 минут разогрева, пять этапов по минимум 10 минут и 10 000 HTTP-запросов, C1–C5.
Запуск не отмечается как успешный, если длительность, p95, успешность, алерты или
отказы не удовлетворяют порогам. В случае превышения порога результат `FAIL`
сохраняется в JSON и отчёте; данные не подменяются.

Испытания идут на `127.0.0.1:18780` (модель) и `127.0.0.1:18781` (монитор).
Проверьте, что эти порты свободны. Все модели в `reports/LAB2` и `reports/LAB3`
используются только для чтения. Runtime-версии — отдельные копии в `demo_output/`.

## Показатели и файлы

В папке `demo_output/LAB8`: `end_to_end.csv`, `summary.json`, `lab8_report.md`,
`alerts_sample.jsonl`, `stream_windows.csv`, `incidents_log.jsonl`, `faults.json`,
CSV запросов и системных ресурсов для каждого этапа, графики CPU/RAM,
`install.log`, `artifacts_manifest.json`, `defense_checklist.md`.

Важное ограничение: исходный `reports/LAB6/service_metrics.json` основан на
измерениях внутри TestClient. Сравнивать с ним настоящую сетевую задержку
следует с пояснением различий методик. Для сопоставимого сетевого эталона
укажите `--baseline-ms` и приложите протокол измерения ЛР6.

### Оптимизация DL и сопоставимый HTTP-бенчмарк (патч №48)

Маршрут ЛР8 использует постоянный пул потоков, а не повторное создание `joblib.Parallel`
для каждого пакета. PyTorch-модель и нормализация загружаются один раз, после чего
вызовы `torch.inference_mode()` не изменяют состояние весов. Исходный код ЛР3 и ЛР6
не меняется. `n_jobs` в ответе отражает число реально назначенных задач (с учётом
размера пакета и доступных логических ядер), а `/lab8/parallel-stats` сохраняет
число потоков, фактически выполнивших вычисления. `/lab8/parallel-proof`
демонстрирует способность пула одновременно обслуживать по одному потоку на
каждое логическое ядро; **этот тест не доказывает 100% загрузку всех ядер**.

Для честного сравнения методик запустите:

```powershell
.\.venv\Scripts\python.exe scripts/lab8_compare_latency.py --samples 200 --warmup 20 --rps 18
```

В `demo_output/LAB8_comparison` сохраняются `matched_http_latency.csv` и
`matched_http_latency.json`. Оба сервера используют **реальные HTTP-запросы** с
одинаковыми входными данными, размером пакета, n_jobs и целевым RPS:
базовый сервер — неизменённый ЛР6 на модели ML v2; оптимизированный — ЛР8
на модели DL v3_dl. Результат сравнивается как между серверами, так и с
опубликованным порогом ЛР6. **Основной порог 1.5×9.367 мс не повышается**:
сравнительный бенчмарк диагностический и не заменяет полный прогон.

В `--quick` дрейфовый поток теперь содержит не меньше реального окна
наблюдения (500 точек по умолчанию). Это позволяет выполнить негативную
проверку обнаружения дрейфа даже в коротком прогоне, но не заменяет
требуемых 10 минут и 10 000 запросов на этап.

`defense_checklist.md` — заготовка, которая должна быть дополнена фактическими
ссылками на скринкасты, рецензию БЯМ, комментарий и ручную приёмку; автоматические
галочки в эти поля не ставятся.


### LAB8 HTTP latency methodology

The LAB8 load client uses one persistent HTTP/1.1 loopback connection **per load stage**,
closing it before advancing to the next stage. Each request is still timed from
JSON serialization/HTTP dispatch to complete response processing; there is no
HTTP retry after a connection failure. Stage CSV/JSON label the transport
`persistent_http_1_1_loopback`. This eliminates artificial Windows TCP connect
overhead from opening a new connection for each very small model request.
It does **not** lower the published LAB6 acceptance limit (14.050785 ms).
Published LAB6 latency was measured with FastAPI TestClient, so it is identified
as a different measurement method. `scripts/lab8_compare_latency.py` measures
the untouched LAB6 HTTP route and optimized LAB8 route with the same transport,
pace and batch shape for a separate, reproducible diagnostic. The runtime still
records failures, time and actual stage durations and never substitutes smoke
evidence for >=600s/10000-request stage trials.
