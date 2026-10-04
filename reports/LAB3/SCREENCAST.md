# Скринкаст ЛР3 — 3–5 минут

## 0:00–0:30
Показать GitHub/ветку ЛР3 и сказать: PyTorch MLP 2→32→32→1, full vs streaming.

## 0:30–1:00
Открыть `configs/lab3.json` и `data/chunks/` — 24 чанка.

## 1:00–1:40
Показать `loss_curves.png` и объяснить epochs/early stopping.

## 1:40–2:20
Показать `dl_metrics.csv`.

Full accuracy: 0.9528.

Streaming accuracy: 0.9531.

## 2:20–2:50
Показать `memory_compare.csv`.

Full: 68.3339 MiB / 4.3779 s.

Streaming: 0.7568 MiB / 0.5029 s.

## 2:50–3:20
Показать `reproducibility.json` и `seed_negative_test.json`.

## 3:20–3:45
Показать `device_parity.csv` и объяснить результат.

## 3:45–4:10
Показать две `.pt` модели и `model_manifest.json`.

## 4:10–4:30
Показать `pytest_coverage.log`, MLflow и зелёный CI.
