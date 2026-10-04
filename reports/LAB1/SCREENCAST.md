# Скринкаст ЛР1 — план на 3–5 минут

## 0:00–0:30
Открыть GitHub. Назвать вариант 09, make_moons и проблему нелинейности.

## 0:30–1:00
Показать `hash_manifest.json`: 24000 объектов, seed 42, noise 0.22, SHA-256.

## 1:00–1:40
Показать `eda_stats.csv` и `schema.csv`: пропусков 0, дубликатов 0, классы 50/50.

## 1:40–2:30
Показать `nonlinearity_scatter.png`.
Объяснить два полумесяца и линейную границу.
Linear accuracy: 0.8679.

Быстро показать остальные четыре графика.

## 2:30–3:10
Показать `precommit_block.log`: три реальные попытки commit заблокированы.

## 3:10–3:35
Показать `hash_negative_test.log` и `reproducibility.log`.

## 3:35–4:00
Запустить:
`python -m pytest .\\tests\\test_data.py .\\tests\\test_eda.py -q`

Показать `3 passed`.

## 4:00–4:20
Показать зелёные GitHub Actions и активный `Protect main`.

## 4:20–4:35
Открыть `lab1_report.docx`.
