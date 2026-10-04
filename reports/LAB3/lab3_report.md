# Лабораторная работа 3. PyTorch: full vs streaming

## 1. Паспорт работы

Автор: Vladimir

Дата: 2026-10-05

Вариант: 09 — `make_moons`, проблема нелинейности.

Хеш коммита-основания: `eb604ec6cf1e344045aa6e436455770e36a997e4`.

Архитектура: MLP `2 -> 32 -> 32 -> 1`.

Число параметров по checkpoint: 1185.

PyTorch: `2.14.0+cu130`.

CUDA available: `True`.

GPU: `NVIDIA GeForce RTX 5070`.

## 2. Настройки обучения

- seed: 42
- batch size: 1024
- epochs: 6
- learning rate: 0.01
- weight decay: 0.0001
- early stopping patience: 2
- bootstrap rounds: 120
- configured training device: `cpu`

В работе сравниваются два источника данных:
1. full — весь train-набор материализуется в памяти;
2. streaming/chunks — train читается по 24 сохранённым чанкам через потоковый загрузчик.

## 3. Метрики

| Метрика | Full | Streaming |
|---|---:|---:|
| Accuracy | 0.952778 | 0.953056 |
| F1 | 0.952778 | 0.952912 |
| ROC-AUC | 0.992540 | 0.992604 |
| Best epoch | 6 | 6 |

Разница accuracy: 0.000278.

Максимальное абсолютное расхождение вероятностных предсказаний full/stream:
0.096559.

## 4. Bootstrap confidence intervals

Full accuracy CI:
`[0.946653; 0.960021]`.

Streaming accuracy CI:
`[0.946361; 0.960833]`.

Full ROC-AUC CI:
`[0.990613; 0.994013]`.

Streaming ROC-AUC CI:
`[0.990674; 0.994020]`.

## 5. Память и время

| Режим | Peak memory, MiB | Time, s |
|---|---:|---:|
| Full | 68.333910 | 4.377913 |
| Streaming | 0.756774 | 0.502916 |

В этом конкретном прогоне streaming использовал на 98.89% меньше измеренной пиковой памяти.

Изменение времени streaming относительно full: -88.51%.

Эти значения являются измерением данной реализации и данного запуска, а не универсальным свойством streaming.

## 6. Early stopping и кривые

Настроено правило early stopping с `patience=2`.

В текущем прогоне лучший epoch full: 6.

Лучший epoch streaming: 6.

Кривые обучения сохранены в:

`figures/loss_curves.png`.

## 7. Воспроизводимость

Режим:
`full_cpu_deterministic`.

Повтор с тем же seed:
- prediction max abs diff: 0.000000;
- accuracy diff: 0.000000;
- допустимое расхождение: 0.005000;
- pass: `True`.

Негативный контроль seed:
- reference seed: 42;
- alternate seed: 43;
- prediction max abs diff: 0.230987;
- accuracy abs diff: 0.001111;
- pass: `True`.

## 8. Device parity

CPU/GPU prediction max absolute difference: 1.7881393432617188e-07. Required tolerance: 1e-5.

## 9. Сравнение с классическим ML из ЛР2

ML full accuracy из ЛР2: 0.955278.

DL full accuracy: 0.952778.

Разница DL - ML: -0.250 процентного пункта.

В данном прогоне классическая RBF+SGD модель ЛР2 немного точнее MLP, что является допустимым результатом: цель ЛР3 — построить и исследовать DL-пайплайн, а не гарантированно превзойти классический ML.

## 10. Checkpoints и MLflow

`dl_model.pt` SHA-256:
`e739b5620b312f4c5104df0c5bed1980ada64e164d77c886eb32d55b664b6cc3`

`dl_model_chunk.pt` SHA-256:
`9571051ccab02b5cf97ef10f67da662b7f5064f6c1586a58df66df044ff925c8`

Full MLflow run:
`049c0140dbf446588f441d8ec994c587`

Streaming MLflow run:
`070df53cc7194382aeed5377f42e35da`

Backend:
`sqlite:///mlflow.db`.

## 11. Негативные конфиги

`config_negative_test.log` подтверждает отказ для:
1. слишком маленького batch size;
2. неположительного learning rate;
3. недопустимого device.

## 12. Выводы

1. Full accuracy составляет 0.9528.
2. Streaming accuracy составляет 0.9531.
3. Расхождение accuracy между режимами составляет 0.0003.
4. Full ROC-AUC составляет 0.9925.
5. Streaming ROC-AUC составляет 0.9926.
6. Full peak memory составляет 68.3339 MiB.
7. Streaming peak memory составляет 0.7568 MiB.
8. Экономия измеренной пиковой памяти streaming составляет 98.89%.
9. Воспроизводимость с seed 42 подтверждена: `True`.
10. Негативный seed-control подтверждён: `True`.
11. DL full отличается от ML ЛР2 на -0.250 процентного пункта.
