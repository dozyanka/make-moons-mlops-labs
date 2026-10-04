# ЛР2 — требования задания по пунктам и что показывать

## Цель

Построить классический ML pipeline варианта 09 и сравнить одинаковую модель
при обучении на полном train-наборе и при работе по 24 чанкам.

## Пункт 1. Pydantic-конфигурация и отказ на плохом конфиге

**Что сделано:** настройки ЛР2 вынесены в `configs/lab2.json` и валидируются
Pydantic-схемой. Проверяются chunk size, gamma, passes и другие параметры.

**Как показать:** открыть `configs/lab2.json` и
`reports/LAB2/config_negative_test.log`.

Там сохранены три отклонённые конфигурации:
- неположительный gamma;
- слишком маленький chunk size;
- passes меньше 1.

## Пункт 2. sklearn pipeline и подбор только по validation

**Что сделано:** основной pipeline:

`StandardScaler -> RBFSampler -> SGDClassifier`.

RBF-преобразование добавляет нелинейность, нужную для двух полумесяцев.
`gamma` подбирается только на validation. Лучшее значение в выполненном
прогоне — 1.4.

**Как показать:** открыть `reports/LAB2/hyperparam_search.csv` и
`reports/LAB2/leakage_negative_test.log`.

## Пункт 3. Full dataset: метрики, bootstrap CI, baseline

**Что сделано:** full-модель обучена на train и финально проверена на test.
Сохранены Accuracy, F1, ROC-AUC и bootstrap confidence intervals.
Для сравнения сохранён линейный LogisticRegression baseline.

**Как показать:**

```powershell
python .\defense_demo\lab2_demo.py --mode full
```

Demo загружает `defense_demo/models/lab2_full.joblib`, делает prediction на
том же test CSV и сверяет метрики с `reports/LAB2/ml_metrics.csv`.

Результат работы:
- accuracy ≈ 0.9553;
- F1 ≈ 0.9553;
- ROC-AUC ≈ 0.9928.

Линейный baseline test accuracy ≈ 0.8731.

## Пункт 4. Chunks, память и время

**Что сделано:** тот же ML pipeline обработан по 24 train-чанкам.
Сохранена отдельная chunk-модель и измерены память/время.

**Как показать:**

```powershell
python .\defense_demo\lab2_demo.py --mode chunks
python .\defense_demo\stream_memory_demo.py
```

В основном прогоне full/chunks дали одинаковые метрики и одинаковые
предсказания, а chunk-режим использовал меньше измеренной пиковой памяти.

`stream_memory_demo.py` отдельно показывает принцип bounded-memory:
увеличивается число обработанных чанков, но одновременно материализуется
только один CSV-чанк.

## Пункт 5. MLflow и модель с хешем

**Что сделано:** full и chunks зарегистрированы в MLflow. Сохранены run ID,
модели `.joblib` и SHA-256 в `model_manifest.json`.

**Как показать:** открыть `mlflow_runs.csv`, `model_manifest.json`.
Для UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Пункт 6. Таблица сравнения

**Что сделано:** `ml_metrics.csv` содержит обе строки full/chunks:
метрики, bootstrap CI, peak memory, elapsed time, run ID и tracking backend.

**Как показать:** открыть `ml_metrics.csv` и `lab2_report.docx`.

## Негативные контроли

- три неправильных Pydantic-конфига отклоняются;
- leakage-control не позволяет подбирать параметры по test;
- повторяемость обеспечивается seed=42 и проверяется повторными CI-прогонами;
- bounded-memory демонстрируется chunk/stream обработкой.

## Артефакты задания

- `reports/LAB2/lab2_report.md`
- `reports/LAB2/ml_metrics.csv`
- `reports/LAB2/ml_model.joblib`
- `reports/LAB2/ml_model_chunk.joblib`
- `reports/LAB2/config_negative_test.log`
- `reports/LAB2/leakage_negative_test.log`
- `reports/LAB2/model_manifest.json`
- `reports/LAB2/pytest_coverage.log`
