# make-moons-mlops-labs

Учебный сквозной MLOps-проект по варианту 09.

**Вариант:** 09  
**Данные:** `sklearn.datasets.make_moons`  
**Основная проблема:** нелинейность классов

## Статус

Лабораторные работы 1, 2 и 3 завершены.

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

## ЛР2

Классический ML для make_moons сравнивается в двух режимах: full dataset и 24 чанка.

Pipeline: StandardScaler -> RBFSampler -> SGDClassifier.

Итоговые метрики, bootstrap-интервалы, время, память и MLflow run ID находятся в eports/LAB2/.

## ЛР3

PyTorch MLP 2 -> 32 -> 32 -> 1 обучается в двух режимах: full dataset и streaming по 24 чанкам.

Метрики, bootstrap-интервалы, кривые обучения, память/время, reproducibility, seed negative-control и device parity находятся в eports/LAB3/.
