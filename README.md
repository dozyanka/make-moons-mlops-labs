# make-moons-mlops-labs

**Вариант:** 09  
**Данные:** `sklearn.datasets.make_moons`  
**Основная проблема:** нелинейность классов

## Статус

Лабораторные работы 1, 2 и 3 завершены.

В проекте реализованы:

- воспроизводимая генерация исходного набора данных;
- SHA-256 исходного CSV;
- разбиение train / validation / test;
- подготовка данных чанками;
- EDA и визуализация нелинейности;
- классический ML в режимах full и chunks;
- PyTorch-модель в режимах full и streaming;
- bootstrap-интервалы для метрик;
- измерение времени и памяти;
- MLflow;
- проверки воспроизводимости;
- негативные контроли;
- CPU/GPU parity;
- pytest;
- pre-commit проверки;
- GitHub Actions CI.

## Основные параметры данных

Для ЛР1-ЛР3 используется один воспроизводимый набор данных:

- `n_samples = 24000`
- `noise = 0.22`
- `seed = 42`
- train: 16800
- validation: 3600
- test: 3600
- train chunks: 24
- chunk size: 700

SHA-256 исходного CSV:

`dd38bd033f2300777fb0420e4c7bf895946435fa3b76c4d495a69e120b54be91`

## ЛР1

Подготовка данных и исследовательский анализ.

Выполнены:

- генерация `make_moons`;
- проверка воспроизводимости данных;
- разбиение train / validation / test;
- подготовка 24 train-чанков;
- EDA;
- анализ распределений и корреляции;
- визуализация двух нелинейно разделимых классов;
- линейный диагностический baseline;
- SHA-256 и проверка изменения данных;
- негативные проверки pre-commit;
- автоматические тесты.

Основные результаты находятся в:

`reports/LAB1/`

Запуск:

```powershell
python .\scripts\prepare_data.py
python .\scripts\run_lab1.py
```

## ЛР2

Классический ML для `make_moons`.

Одна модель обучается в двух режимах:

- full dataset;
- chunks.

Pipeline:

`StandardScaler -> RBFSampler -> SGDClassifier`

Параметр `gamma` выбирается по validation-набору.

Выполнены:

- обучение full и chunks;
- подбор `gamma`;
- сравнение с линейным baseline;
- Accuracy, F1 и ROC-AUC;
- bootstrap confidence intervals;
- сравнение времени и памяти;
- проверка совпадения предсказаний;
- сохранение моделей `.joblib`;
- MLflow;
- негативные проверки конфигурации и leakage;
- pytest и coverage.

Основные результаты находятся в:

`reports/LAB2/`

Запуск:

```powershell
python .\scripts\prepare_data.py
python .\scripts\run_lab2.py
```

## ЛР3

Deep Learning на PyTorch.

Используется MLP:

`2 -> 32 -> 32 -> 1`

Модель обучается в двух режимах:

- full dataset;
- streaming по 24 чанкам.

Выполнены:

- обучение PyTorch MLP;
- full и streaming режимы;
- Accuracy, F1 и ROC-AUC;
- bootstrap confidence intervals;
- loss curves;
- сравнение времени и памяти;
- сохранение `.pt` checkpoints;
- проверка воспроизводимости с одинаковым seed;
- negative-control с другим seed;
- CPU/GPU parity;
- MLflow;
- pytest и coverage.

Основные результаты находятся в:

`reports/LAB3/`

Запуск:

```powershell
python .\scripts\prepare_data.py
python .\scripts\run_lab3.py
```

## Структура проекта

- `configs/` — конфигурации лабораторных работ;
- `data/` — исходные, обработанные данные и чанки;
- `reports/` — отчёты, метрики, графики и остальные артефакты;
- `scripts/` — сценарии запуска и формирования результатов;
- `src/moons_lab/` — основной код проекта;
- `tests/` — автоматические тесты;
- `defense_demo/` — готовые модели и демонстрационные сценарии;
- `DEMO/` — отдельный launcher и инструкция для проверки проекта;
- `.github/` — GitHub Actions и настройки репозитория.

## Live defense demos

Готовые модели ЛР1-ЛР3, команды для демонстрации и заметки по каждому пункту задания находятся в `defense_demo/`.

Запуск всех сохранённых моделей:

```powershell
python .\defense_demo\run_all.py
```

## Тесты

Запуск тестов проекта:

```powershell
python -m pytest -q
```

Подробная инструкция для автоматической проверки ЛР1-ЛР3 находится в:

`DEMO/README.md`
