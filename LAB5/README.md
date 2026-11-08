# Лабораторная работа №5 — генератор синтетических данных

Вариант 09 (`sklearn.datasets.make_moons`), профиль P1.

**Основной запуск на Windows:** откройте `RUN_DEMO.cmd` из корня проекта,
выберите пункт **14**. Результаты повторного прогона находятся в
`demo_output/LAB5/`. Пункт **15** открывает опубликованный отчёт.

**Воспроизведение независимо от DEMO:**

```powershell
.\.venv\Scripts\python.exe scripts/run_lab5.py --report-dir demo_output/LAB5 --data-dir demo_output/LAB5/data
```

**Проверка отдельно:**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_generator.py tests/test_lab5_workflow.py -q --cov=moons_lab.generator --cov-report=term-missing --cov-fail-under=80 --basetemp=demo_output/pytest_runtime/lab5_manual
```

**Опубликованные материалы:**

- `reports/LAB5/lab5_report.md` — методика, таблицы PSI/KS и выводы;
- `reports/LAB5/synthetic_metrics.csv` — показатели для каждого признака и режима;
- `reports/LAB5/generator_config.json` — все режимы и ручки с seed;
- `reports/LAB5/generator_tests.log` — результаты pytest-cov;
- `reports/LAB5/reproducibility.json` — SHA-256 выборок и повторов;
- `data/synthetic/{similar,random,drifted}.csv` — сохранённые примеры.

`data/raw/moons.csv` — только для чтения, его SHA-256 проверяется перед
запуском. ЛР4 использует неизменные `reports/LAB2/ml_model.joblib`,
`reports/LAB4/`, `src/moons_lab/validator.py` и `scripts/run_lab4.py`.
