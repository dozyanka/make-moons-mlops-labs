# ЛР3 — требования задания по пунктам и что показывать

## Цель

Построить воспроизводимый PyTorch pipeline MLP `2 -> 32 -> 32 -> 1` и
сравнить обучение с полным train-набором в памяти и streaming по чанкам.

## Пункт 1. Pydantic-настройки DL

**Что сделано:** `configs/lab3.json` фиксирует:
- seed 42;
- batch size 1024;
- epochs 6;
- learning rate 0.01;
- weight decay 0.0001;
- patience 2;
- bootstrap rounds 120;
- device.

**Как показать:** открыть `configs/lab3.json` и
`reports/LAB3/config_negative_test.log`.

Плохие batch size, learning rate и device отклоняются.

## Пункт 2. Full source и streaming Dataset/DataLoader

**Что сделано:** full-режим держит train в памяти; streaming-режим читает
24 сохранённых чанка без материализации всего train одновременно.

**Как показать:** открыть `src/moons_lab/dl_pipeline.py`, показать
Dataset/DataLoader/streaming-код и папку `data/chunks/`.

Для демонстрации bounded-memory:

```powershell
python .\defense_demo\stream_memory_demo.py
```

## Пункт 3. Обучение двух режимов, seed, early stopping, кривые

**Что сделано:** MLP обучена в full и streaming режимах с seed=42.
Настроен early stopping с patience=2. История train/validation сохранена
в CSV, а кривые — в `figures/loss_curves.png`.

**Как показать:** открыть:
- `full_history.csv`;
- `chunk_history.csv`;
- `figures/loss_curves.png`.

## Пункт 4. Метрики, bootstrap CI и сравнение с ЛР2

**Что сделано:** для full/streaming рассчитаны Accuracy, F1, ROC-AUC и
bootstrap confidence intervals. DL сравнивается с классическим ML из ЛР2.

**Как показать:**

```powershell
python .\defense_demo\lab3_demo.py --mode full
python .\defense_demo\lab3_demo.py --mode streaming
```

В выполненном прогоне:
- DL full accuracy ≈ 0.9528;
- DL streaming accuracy ≈ 0.9531;
- ML ЛР2 full accuracy ≈ 0.9553.

То, что классическая RBF+SGD чуть лучше MLP, не является ошибкой:
цель ЛР3 — воспроизводимый DL pipeline и streaming, а не обязательная победа DL.

## Пункт 5. Воспроизводимость и CPU/GPU parity

**Что сделано:** повтор full CPU с seed=42 дал:
- prediction max abs diff = 0;
- accuracy diff = 0;
- допуск задания 0.005 выполнен.

Негативный контроль seed=43 дал заметно другие prediction.

После установки CUDA PyTorch выполнен CPU/GPU parity на RTX 5070:
max abs diff ≈ `1.788e-07`, что значительно меньше лимита `1e-5`.

**Как показать:** открыть:
- `reproducibility.json`;
- `seed_negative_test.json`;
- `device_parity.csv`;
- `environment.json`.

## Пункт 6. MLflow, checkpoints, хеши, память и время

**Что сделано:** оба режима зарегистрированы в MLflow. Сохранены:
- `dl_model.pt`;
- `dl_model_chunk.pt`;
- SHA-256 в `model_manifest.json`;
- `dl_metrics.csv`;
- `memory_compare.csv`.

**Как показать:** открыть `mlflow_runs.csv`, `model_manifest.json`,
`memory_compare.csv`.

В выполненном прогоне:
- full peak memory ≈ 68.33 MiB;
- streaming peak memory ≈ 0.764 MiB;
- full time ≈ 4.40 s;
- streaming time ≈ 0.49 s.

## Негативные контроли

- другой seed меняет prediction;
- плохая Pydantic-конфигурация отклоняется;
- streaming не держит весь train в памяти;
- pytest для DL проходит, coverage `dl_pipeline.py` ≈ 93%.

## Артефакты задания

- `reports/LAB3/lab3_report.md`
- `reports/LAB3/dl_metrics.csv`
- `reports/LAB3/dl_model.pt`
- `reports/LAB3/dl_model_chunk.pt`
- `reports/LAB3/memory_compare.csv`
- `reports/LAB3/figures/loss_curves.png`
- `reports/LAB3/reproducibility.json`
- `reports/LAB3/device_parity.csv`
- `reports/LAB3/model_manifest.json`
- `reports/LAB3/pytest_coverage.log`
