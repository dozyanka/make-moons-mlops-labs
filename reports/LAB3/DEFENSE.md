# Защита ЛР3

## Что сказать

В ЛР3 задача make_moons решается нейросетью PyTorch MLP 2→32→32→1.

Есть два режима:
- full — train целиком в памяти;
- streaming — данные читаются по 24 чанкам через потоковый загрузчик.

Full accuracy: 0.9528.

Streaming accuracy: 0.9531.

Full peak memory: 68.3339 MiB.

Streaming peak memory: 0.7568 MiB.

Full time: 4.3779 s.

Streaming time: 0.5029 s.

## Что показать

1. `configs/lab3.json`.
2. `src/moons_lab/dl_pipeline.py` — MLP и streaming loader.
3. `data/chunks/` — 24 чанка.
4. `dl_metrics.csv`.
5. `memory_compare.csv`.
6. `figures/loss_curves.png`.
7. `dl_model.pt` и `dl_model_chunk.pt`.
8. `reproducibility.json`.
9. `seed_negative_test.json`.
10. `device_parity.csv`.
11. `pytest_coverage.log`.
12. MLflow UI.

## Воспроизводимость

Повтор с seed=42 дал prediction diff 0 и metric diff 0.

При seed=43 prediction max abs diff стал:
0.230987.

Это демонстрирует, зачем фиксировать seed.

## CPU/GPU

CUDA available: True.

GPU: NVIDIA GeForce RTX 5070.

CPU/GPU prediction max absolute difference: 1.7881393432617188e-07. Required tolerance: 1e-5.

## Сравнение с ЛР2

Классический ML ЛР2 accuracy:
0.9553.

DL full accuracy:
0.9528.

То есть в этом запуске классический RBF+SGD немного точнее MLP. Это нормальный результат; ЛР3 проверяет DL-пайплайн, streaming и воспроизводимость.
