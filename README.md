# make-moons-mlops-labs

Учебный сквозной MLOps-проект по варианту 09.

**Вариант:** 09  
**Данные:** `sklearn.datasets.make_moons`  
**Основная проблема:** нелинейность классов

## Статус

Лабораторная работа 1 завершена.

Реализованы:

- воспроизводимая генерация исходного набора данных;
- SHA-256 исходного CSV;
- разбиение train / validation / test;
- подготовка чанков;
- EDA средствами pandas, matplotlib и seaborn;
- визуализация нелинейности;
- pre-commit проверки;
- негативные контроли;
- GitHub Actions CI.

## ЛР1: параметры данных

- `n_samples = 24000`
- `noise = 0.22`
- `seed = 42`
- train: 16800
- validation: 3600
- test: 3600
- train chunks: 24
- chunk size: 700

SHA-256 raw-файла:

`dd38bd033f2300777fb0420e4c7bf895946435fa3b76c4d495a69e120b54be91`

## Запуск ЛР1

```powershell
.\.venv\Scripts\Activate.ps1
python .\scripts\run_lab1.py
python -m pytest .\tests\test_data.py .\tests\test_eda.py -q
pre-commit run --all-files
```

## Структура проекта

- `configs/` — конфигурации;
- `data/` — данные;
- `reports/` — отчёты и численные артефакты;
- `scripts/` — сценарии запуска;
- `src/moons_lab/` — код проекта;
- `tests/` — тесты;
- `.github/` — CI и шаблон pull request.
