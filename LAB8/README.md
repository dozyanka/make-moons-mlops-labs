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

`defense_checklist.md` — заготовка, которая должна быть дополнена фактическими
ссылками на скринкасты, рецензию БЯМ, комментарий и ручную приёмку; автоматические
галочки в эти поля не ставятся.
