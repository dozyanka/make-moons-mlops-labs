# Защита ЛР2

## Что сказать

В ЛР2 одна и та же нелинейная классическая ML-модель обучается двумя способами: на полном train-наборе и через 24 сохранённых чанка.

Pipeline: StandardScaler → RBFSampler → SGDClassifier.

`gamma` выбирается только по validation. Лучшее gamma: 1.4.

Линейный baseline accuracy: 0.8731.

Нелинейная модель accuracy: 0.9553.

Это показывает, что для make_moons нелинейное представление работает заметно лучше линейного baseline.

## Что показать

1. `data/chunks/` — 24 CSV-файла.
2. `hyperparam_search.csv` — выбор gamma по validation.
3. `ml_metrics.csv` — full vs chunks, accuracy/F1/ROC-AUC, bootstrap CI, память, время.
4. `ml_model.joblib` и `ml_model_chunk.joblib`.
5. `model_manifest.json` — SHA-256 моделей.
6. `config_negative_test.log`.
7. `leakage_negative_test.log`.
8. `pytest_coverage.log`.
9. MLflow UI.

## MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Открыть `http://127.0.0.1:5000`.

Full run ID: 266a11fa19084bbd89cbd9dde34b5957

Chunks run ID: 92db92877da945769ba1fc79b7b3ad6e

## Главный результат

Full:
- accuracy 0.9553
- F1 0.9553
- ROC-AUC 0.9928
- memory 1.8372 MiB
- time 0.3320 s

Chunks:
- accuracy 0.9553
- F1 0.9553
- ROC-AUC 0.9928
- memory 1.5651 MiB
- time 0.5426 s

В этом детерминированном прогоне full и chunks дали одинаковые predictions.
Разница заключается в способе подачи train-данных и профиле ресурсов.
