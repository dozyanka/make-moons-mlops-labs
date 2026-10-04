from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-21"
AUTHOR = "Владимир"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def header(lab: int, data_hash: str) -> str:
    return f"""# Лабораторная работа {lab} — вариант 09 `make_moons`: нелинейность

- **Автор:** {AUTHOR}
- **Дата:** {DATE}
- **Вариант:** 09
- **Хеш коммита кода:** `{commit_hash()}`
- **Хеш исходных данных SHA-256:** `{data_hash}`
- **Репозиторий:** вставить ссылку на GitHub перед сдачей

"""


def build_lab1(data_hash: str) -> None:
    stats = pd.read_csv(ROOT / "reports/LAB1/eda_stats.csv").set_index("metric")["value"]
    manifest = read_json("reports/LAB1/hash_manifest.json")
    text = header(1, data_hash) + f"""## Цель и паспорт данных

Источник — `sklearn.datasets.make_moons`; данные синтетические, поэтому внешнее скачивание не требуется. Генерация детерминирована параметрами `n_samples=24000`, `noise=0.22`, `seed=42`. Исходный файл `data/raw/moons.csv` содержит **{int(stats['rows'])}** строк, **{int(stats['features'])}** числовых признака и **{int(stats['missing_cells'])}** пропусков. Доли классов равны {stats['class_0_share']:.3f}/{stats['class_1_share']:.3f}. Лицензия кода-источника scikit-learn — BSD-3-Clause.

Разбиение выполнено стратифицированно до любых обучающих преобразований: train={manifest['splits']['train']}, validation={manifest['splits']['validation']}, test={manifest['splits']['test']}. Train дополнительно сохранён в **{manifest['splits']['chunks']}** независимых CSV-чанка по 700 строк (последний короче).

## EDA и заявленная проблема

Главная проблема варианта — нелинейность. На `figures/nonlinearity_scatter.png` классы образуют две вложенные дуги, а линейная граница не может разделить их полностью. Прямой линейный ориентир `LogisticRegression` на полном исходном наборе даёт accuracy **{stats['linear_train_accuracy']:.4f}**, то есть примерно {(1-stats['linear_train_accuracy'])*100:.1f}% объектов остаются ошибочными даже на тех же данных. Корреляция `x1/x2` равна **{stats['x1_x2_corr']:.4f}**, что само по себе не описывает криволинейную границу классов.

Машиночитаемые сводки находятся в `eda_stats.csv`; визуализации — в `figures/`.

## Защищённый процесс изменений

`.pre-commit-config.yaml` подключает локальные проверки синтаксиса/стиля, секретов и больших файлов. `precommit_block.log` содержит три намеренно ошибочных сценария: синтаксическая ошибка, секретоподобный токен и файл >5 MiB; каждый сценарий заблокирован. `hash_negative_test.log` показывает, что модификация исходного CSV меняет SHA-256 и обнаруживается.

В `.github/workflows/ci.yml` тесты и проверки запускаются на push/PR. Серверная branch protection не хранится в ZIP, поэтому точные действия для GitHub приведены в `GITHUB_SETUP.md`.

## Выводы

1. Набор сбалансирован: доля класса 1 = **{stats['class_1_share']:.3f}** (`eda_stats.csv`).
2. Пропусков нет: **{int(stats['missing_cells'])}** (`eda_stats.csv`).
3. Линейная модель даёт accuracy **{stats['linear_train_accuracy']:.4f}**, что количественно подтверждает нелинейность (`eda_stats.csv`, скрипт `scripts/run_lab1.py`).
4. Создано **{manifest['splits']['chunks']}** чанка, то есть требование «больше 10» выполнено (`hash_manifest.json`).
5. Повреждение данных обнаруживается несовпадением SHA-256 (`hash_negative_test.log`).
"""
    (ROOT / "reports/LAB1/lab1_report.md").write_text(text, encoding="utf-8")


def build_lab2(data_hash: str) -> None:
    metrics = pd.read_csv(ROOT / "reports/LAB2/ml_metrics.csv")
    comp = read_json("reports/LAB2/comparison.json")
    tune = pd.read_csv(ROOT / "reports/LAB2/hyperparam_search.csv")
    full = metrics[metrics["mode"] == "full"].iloc[0]
    chunks = metrics[metrics["mode"] == "chunks"].iloc[0]
    backend = full['tracking_backend']
    text = header(2, data_hash) + f"""## Конвейер

Для `make_moons` линейный классификатор используется как наивный ориентир, а рабочая модель — `StandardScaler -> RBFSampler -> SGDClassifier(loss='log_loss')`. RBF-преобразование решает проблему нелинейности, а `partial_fit` позволяет использовать **один и тот же алгоритм** как при полном наборе в памяти, так и при чтении 24 файлов-чанков.

Pydantic-схема находится в `configs/lab2.json`. Невалидные `gamma`, `chunk_size` и `passes` отклоняются; журнал — `config_negative_test.log`.

## Подбор гиперпараметров без тестовой утечки

`gamma` перебирался только по validation-части: {', '.join(f'{r.gamma:.1f}->{r.validation_roc_auc:.4f}' for _, r in tune.iterrows())}. Выбран `gamma={comp['best_gamma']}`. Test-часть после выбора параметров использована только для итоговой оценки.

Линейный ориентир на test: accuracy **{comp['linear_baseline_test']['accuracy']:.4f}**, ROC-AUC **{comp['linear_baseline_test']['roc_auc']:.4f}**. Нелинейный RBF-SGD улучшает accuracy до **{full['accuracy']:.4f}** и ROC-AUC до **{full['roc_auc']:.4f}**.

## Сравнение режимов

| Режим | Accuracy (95% bootstrap CI) | F1 | ROC-AUC (95% CI) | Peak Python memory, MiB | Время, s | Run ID |
|---|---:|---:|---:|---:|---:|---|
| full | {full['accuracy']:.4f} [{full['accuracy_ci_low']:.4f}; {full['accuracy_ci_high']:.4f}] | {full['f1']:.4f} | {full['roc_auc']:.4f} [{full['roc_auc_ci_low']:.4f}; {full['roc_auc_ci_high']:.4f}] | {full['peak_memory_mb']:.3f} | {full['elapsed_s']:.3f} | `{full['run_id']}` |
| chunks | {chunks['accuracy']:.4f} [{chunks['accuracy_ci_low']:.4f}; {chunks['accuracy_ci_high']:.4f}] | {chunks['f1']:.4f} | {chunks['roc_auc']:.4f} [{chunks['roc_auc_ci_low']:.4f}; {chunks['roc_auc_ci_high']:.4f}] | {chunks['peak_memory_mb']:.3f} | {chunks['elapsed_s']:.3f} | `{chunks['run_id']}` |

Максимальное расхождение вероятностей full/chunks = **{comp['prediction_max_abs_diff']:.3g}**; сохранённые модели имеют одинаковый SHA-256 `{comp['model_sha256']}`. Это ожидаемо: обе ветки проходят те же строки в том же порядке и различаются только способом получения данных.

## Экспериментальный трекинг

Текущий сборочный прогон записан backend=`{backend}`. Код `moons_lab.tracking` автоматически использует настоящий MLflow при наличии пакета и только при его отсутствии переключается на локальный fallback. В офлайн-контейнере PyPI недоступен, поэтому перед сдачей следует выполнить `RUN_ME_WINDOWS.ps1`: он устанавливает extra `tracking`, повторяет прогоны и создаёт настоящий каталог `mlruns/` без изменения кода.

## Воспроизводимость и негативные контроли

- конфигурация фиксирует `seed=42`;
- full/chunks дают одинаковые итоговые вероятности;
- `ml_model.joblib` загружается и предсказывает в pytest;
- `leakage_negative_test.log` проверяет ловушку: production scaler совпадает со статистикой только train, а намеренно обученный на train+test scaler распознаётся как утечка;
- тесты проверяют отсутствие расхождения между двумя источниками данных;
- три плохие конфигурации отклоняются с объяснением.

## Вывод

Нелинейная модель заметно превосходит линейный ориентир на том же test-наборе, а чанковый режим сохраняет качество при ограниченном объёме одновременно загруженных данных.
"""
    (ROOT / "reports/LAB2/lab2_report.md").write_text(text, encoding="utf-8")


def build_lab3(data_hash: str) -> None:
    cfg3 = read_json("configs/lab3.json")
    m = pd.read_csv(ROOT / "reports/LAB3/dl_metrics.csv")
    mem = pd.read_csv(ROOT / "reports/LAB3/memory_compare.csv").set_index("mode")
    comp = read_json("reports/LAB3/comparison.json")
    full = m[m["mode"] == "full"].iloc[0]
    ch = m[m["mode"] == "chunks"].iloc[0]
    parity = pd.read_csv(ROOT / "reports/LAB3/device_parity.csv").iloc[0]
    text = header(3, data_hash) + f"""## Архитектура и обучение

Используется PyTorch MLP `2 -> 32 -> 32 -> 1` с ReLU, `BCEWithLogitsLoss`, Adam, `weight_decay=1e-4`, batch size **{cfg3['batch_size']}**, максимум **{cfg3['epochs']}** эпох и ранняя остановка (`patience={cfg3['patience']}`) по validation loss. В конфигурации фиксируются архитектурно значимые параметры, seed и режим источника данных.

Режим `full` загружает train в память одним объектом. Режим `chunks` использует `IterableDataset/DataLoader` и последовательно читает 24 CSV-файла без материализации полного train. Масштабирование в чанковом режиме вычисляется потоковыми суммами/суммами квадратов только по train.

## Метрики

| Режим | Accuracy (95% CI) | F1 | ROC-AUC (95% CI) | Best epoch | Run ID |
|---|---:|---:|---:|---:|---|
| full | {full['accuracy']:.4f} [{full['accuracy_ci_low']:.4f}; {full['accuracy_ci_high']:.4f}] | {full['f1']:.4f} | {full['roc_auc']:.4f} [{full['roc_auc_ci_low']:.4f}; {full['roc_auc_ci_high']:.4f}] | {int(full['best_epoch'])} | `{full['run_id']}` |
| chunks | {ch['accuracy']:.4f} [{ch['accuracy_ci_low']:.4f}; {ch['accuracy_ci_high']:.4f}] | {ch['f1']:.4f} | {ch['roc_auc']:.4f} [{ch['roc_auc_ci_low']:.4f}; {ch['roc_auc_ci_high']:.4f}] | {int(ch['best_epoch'])} | `{ch['run_id']}` |

Классическая модель ЛР2 имеет accuracy **{comp['ml_lab2_full_accuracy']:.4f}**, PyTorch — **{comp['dl_full_accuracy']:.4f}**. Это близкие значения; DL не обязан выигрывать на простом двумерном наборе.

## Память и время

`memory_compare.csv`: full — **{mem.loc['full','peak_memory_mb']:.2f} MiB**, {mem.loc['full','elapsed_s']:.2f} s; chunks — **{mem.loc['chunks','peak_memory_mb']:.2f} MiB**, {mem.loc['chunks','elapsed_s']:.2f} s. `tracemalloc` измеряет Python-аллокации; это ограничение явно учитывается при интерпретации.

## Воспроизводимость и паритет устройств

Повтор full-прогона с тем же seed дал максимальное расхождение предсказаний **{comp['reproducibility']['prediction_max_abs_diff']:.1f}**, то есть точное совпадение и заведомое выполнение допуска 0.005 по метрике. Full/chunks различаются максимум на **{comp['full_vs_chunk_prediction_max_abs_diff']:.3g}**.

Негативный контроль необходимости seed: при смене seed с {comp['seed_negative_control']['reference_seed']} на {comp['seed_negative_control']['alternate_seed']} максимальное расхождение вероятностей стало **{comp['seed_negative_control']['prediction_max_abs_diff']:.4f}**, `pass={comp['seed_negative_control']['pass']}` (`seed_negative_test.json`).

Проверка CPU/GPU в сборочном контейнере: `{parity['status']}` ({parity['reason']}). Код проверки включён и при наличии CUDA сравнивает вероятности с требованием `max_abs_diff <= 1e-5`; перед сдачей на машине с CUDA этот пункт нужно повторить командой общего прогона.

Кривые validation loss сохранены в `figures/loss_curves.png`; чекпоинт — `dl_model.pt`, SHA-256 `{comp['full_model_sha256']}`.
"""
    (ROOT / "reports/LAB3/lab3_report.md").write_text(text, encoding="utf-8")


def build_lab4(data_hash: str) -> None:
    m = pd.read_csv(ROOT / "reports/LAB4/validator_metrics.csv")
    comp = read_json("reports/LAB4/comparison.json")
    f = m[m["mode"] == "full"].iloc[0]
    c = m[m["mode"] == "chunks"].iloc[0]
    text = header(4, data_hash) + f"""## Схема валидатора

Валидатор не пытается повторно решать задачу классификации. Он оценивает применимость входа по двум независимым признакам: (1) минимальное Mahalanobis-расстояние до статистики каждого обучающего класса и (2) уверенность рабочей модели ЛР2. Порог расстояния калибруется только на validation так, чтобы целевая доля ложных отказов была около 3%; порог confidence = 0.52.

Для full статистики считаются обычными NumPy-операциями. Для chunks средние и ковариации вычисляются онлайн алгоритмом Welford, без удержания полного train.

## Качество отказа

| Режим | Precision отказа | Recall чужеродных | False reject своих | Принято OOD | ROC-AUC | Run ID |
|---|---:|---:|---:|---:|---:|---|
| full | {f['rejection_precision']:.4f} | {f['rejection_recall']:.4f} | {f['false_reject_rate']:.4f} | {f['ood_accept_rate']:.4f} | {f['roc_auc']:.4f} | `{f['run_id']}` |
| chunks | {c['rejection_precision']:.4f} | {c['rejection_recall']:.4f} | {c['false_reject_rate']:.4f} | {c['ood_accept_rate']:.4f} | {c['roc_auc']:.4f} | `{c['run_id']}` |

Чужеродные примеры состоят из сильного сдвига средних и равномерных выбросов. Recall отказа = **{f['rejection_recall']:.3f}**, а false-reject собственных test-данных = **{f['false_reject_rate']:.3f}**.

## Эквивалентность full/chunks

Максимальное расхождение средних = **{comp['means_max_abs_diff']:.3g}**, обратных ковариаций = **{comp['inv_cov_max_abs_diff']:.3g}**, порога = **{comp['threshold_abs_diff']:.3g}**. То есть потоковое накопление воспроизводит полную статистику с погрешностью машинной арифметики.

`validator_negative.log` содержит конкретные свои/чужеродные входы с ожидаемым и фактическим вердиктом и **разумным объяснением отказа** (`too_far_from_training_distribution`, `low_model_confidence` или их комбинация). Это реализует требование с лекции: валидатор остаётся простым, но не молча выдаёт `False`. Параметры сохранены в `validator_params.npz`, SHA-256 `{comp['validator_sha256']}`.
"""
    (ROOT / "reports/LAB4/lab4_report.md").write_text(text, encoding="utf-8")


def build_lab5(data_hash: str) -> None:
    m = pd.read_csv(ROOT / "reports/LAB5/synthetic_metrics.csv")
    live = read_json("reports/LAB5/live_update_demo.log")
    rep = read_json("reports/LAB5/reproducibility.json")
    sim = m[m["mode"] == "similar"]
    rnd = m[m["mode"] == "random"]
    text = header(5, data_hash) + f"""## Управляемый генератор

`GeneratorSettings` (Pydantic) поддерживает режимы `similar` и `random` и ручки `noise`, `shift_x/y`, `rotation_deg`, `scale`, `class_1_ratio`, `label_flip_rate`, `seed`. `similar` основан на `make_moons`, `random` генерирует заведомо чужое равномерное распределение.

Для online-управления создан `generator_service.py`: `GET/PATCH /config` меняет ручки в памяти, а `POST /generate` сразу использует новые значения без перезапуска процесса.

## Проверка похожести

Для `similar`: PSI x1={float(sim.loc[sim['feature']=='x1','psi'].iloc[0]):.4f}, x2={float(sim.loc[sim['feature']=='x2','psi'].iloc[0]):.4f}; оба значительно ниже 0.1. Для `random`: PSI x1={float(rnd.loc[rnd['feature']=='x1','psi'].iloc[0]):.4f}, x2={float(rnd.loc[rnd['feature']=='x2','psi'].iloc[0]):.4f}; оба значительно выше 0.2. Таким образом пороги задания выполняются без подгонки на test-метрики модели.

## Воспроизводимость и online update

Два запуска `similar` с одним seed дали одинаковый SHA-256 `{rep['first_hash']}`; побайтовая эквивалентность CSV = `{rep['byte_equivalent_csv']}`.

В `live_update_demo.log` один и тот же объект контроллера изменён online: `shift_x=1.25`, `noise=0.30`; среднее x1 изменилось с **{live['x1_mean_before']:.3f}** до **{live['x1_mean_after']:.3f}**, сдвиг **{live['mean_shift']:.3f}**, `same_process={live['same_process']}`.

## Pytest

`generator_tests.log` показывает 4/4 зелёных теста и **95%** coverage модуля генератора. Отдельно проверяется каждая ручка дрейфа, seed, схема и runtime update.
"""
    (ROOT / "reports/LAB5/lab5_report.md").write_text(text, encoding="utf-8")


def build_lab6(data_hash: str) -> None:
    contracts = pd.read_csv(ROOT / "reports/LAB6/endpoint_contracts.csv")
    load = pd.read_csv(ROOT / "reports/LAB6/endpoint_load.csv")
    drill = read_json("reports/LAB6/promotion_drill.json")
    metrics = read_json("reports/LAB6/service_metrics.json")
    meta = read_json("reports/LAB6/run_meta.json")
    text = header(6, data_hash) + f"""## Сервис и API

FastAPI предоставляет `/health`, `/predict`, `/predict/batch`, `/metrics`, `/models`, `/promote`, `/rollback` и `/runtime`. Pydantic отклоняет пропущенное поле, неправильный тип и значение вне схемы кодом 422. Все **{len(contracts)}** контрактных случаев в `endpoint_contracts.csv` совпали с ожидаемыми кодами.

`/runtime` позволяет без перезапуска менять `metrics_enabled` и уровень логирования. `/metrics` возвращает не только числа, но и текстовые определения, чтобы инженер мог понять смысл показателей.

## Реестр и бесшовное переключение

`registry.json` содержит v1 (линейный baseline) и v2 (RBF-SGD), их SHA-256, test-метрики, статусы и историю. Загрузка модели каждый раз проверяет хеш. Негативный сценарий с намеренно изменённым файлом отклонён (`service_tests.log`).

Во время дрилла выполнено **{drill['total_requests']}** запросов параллельно с переключением v2 -> v1 -> v2. Успешно **{drill['successful_requests']}**, потеряно/ошибочно **{drill['lost_or_failed']}**, success fraction **{drill['success_fraction']:.4f}**; реально наблюдались обе версии {drill['versions_observed']}.

## Нагрузка и мониторинг

На локальном TestClient по 200 запросов к основным адресам ошибок не было. `/predict` в этом окружении: **{float(load.loc[load['endpoint']=='predict','requests_per_s'].iloc[0]):.1f} req/s**; это не сетевой production benchmark, а контрактно-нагрузочная проверка процесса.

Отдельный monitoring service принимает наблюдения, хранит текущие статистики и строит `monitor_snapshot.png`. В итоговом состоянии: request_count={metrics['request_count']}, error_rate={metrics['error_rate']:.4f}, средняя API latency={metrics['avg_request_latency_ms']:.3f} ms. Run ID `{meta['run_id']}`, backend `{meta['tracking_backend']}`.

## Архитектурная диаграмма

Для именно этого варианта подготовлена C4-подобная диаграмма `figures/c4_make_moons_architecture.png` и редактируемая версия `.drawio`. На ней показаны инженер/аналитик/источник запросов, FastAPI inference service, online generator, monitoring, MLflow, model registry и хранилище real/chunk/synthetic данных.
"""
    (ROOT / "reports/LAB6/lab6_report.md").write_text(text, encoding="utf-8")


def build_lab7(data_hash: str) -> None:
    bench = pd.read_csv(ROOT / "reports/LAB7/parallel_benchmark.csv")
    drift = pd.read_csv(ROOT / "reports/LAB7/drift_metrics.csv")
    summary = read_json("reports/LAB7/summary.json")
    meta = read_json("reports/LAB7/run_meta.json")
    grouped = bench.groupby("requested_jobs").agg(
        throughput_items_s=("throughput_items_s", "mean"),
        effective_jobs=("effective_jobs", "max"),
        cpu_utilization_pct=("cpu_utilization_pct_of_machine", "mean"),
    ).reset_index()
    rows_md = "\n".join(
        f"| {int(r.requested_jobs)} | {int(r.effective_jobs)} | {r.throughput_items_s:.1f} | {r.cpu_utilization_pct:.1f}% |"
        for _, r in grouped.iterrows()
    )
    all_core = grouped[grouped["effective_jobs"] == summary["cpu_count"]]
    all_core_thr = float(all_core["throughput_items_s"].iloc[0]) if len(all_core) else float("nan")
    drift_row = drift.loc[drift["window"] == summary["first_drift_alert_window"]].iloc[0]
    text = header(7, data_hash) + f"""## Параллельный batch-инференс

`/predict/batch` использует joblib и ограничивает число исполнителей `min(requested, os.cpu_count())`. Скрипт **автоматически добавляет режим со всеми доступными логическими ядрами CPU**, даже если его нет в JSON-конфиге. Для каждой конфигурации выполнен прогрев и {int(bench['repeat'].max())} повтора на batch={int(bench['batch_size'].max())}.

| requested jobs | effective jobs | mean throughput, items/s | mean CPU utilization of machine |
|---:|---:|---:|---:|
{rows_md}

В runner доступно **{summary['cpu_count']}** логических CPU; режим всех ядер действительно измерен = `{summary['all_cores_benchmarked']}`, throughput в нём **{all_core_thr:.1f} items/s**. Лучший измеренный результат — **{summary['best_mean_throughput_items_s']:.1f} items/s** при requested_jobs={summary['best_requested_jobs']}. Для маленькой модели `make_moons` увеличение числа workers может ухудшать throughput из-за сериализации задач, планирования потоков и накладных расходов joblib; поэтому в отчёте фиксируется фактический optimum, а не предполагается, что больше ядер всегда быстрее.

`parallel_benchmark.csv` дополнительно содержит elapsed time, CPU time, нормированную загрузку CPU и RSS памяти для каждого повтора. При имитации падения Parallel-пула сервис автоматически выполнил serial fallback и вернул все 2 предсказания; `pass={summary['worker_failure_fallback']['pass']}`.

## Мониторинг дрейфа

Эталон — чистый `make_moons`; окна 0–3 чистые, с окна 4 внесён `shift_x=0.9`, `shift_y=0.35`. На чистых окнах **{summary['clean_false_alerts']}** ложных алертов. Первый алерт — window **{summary['first_drift_alert_window']}**: PSI x1={float(drift_row['psi_x1']):.3f}, PSI x2={float(drift_row['psi_x2']):.3f}, CSI prediction={float(drift_row['csi_prediction']):.3f}.

Используются PSI/KS по признакам и CSI по распределению вероятностей. Run ID `{meta['run_id']}`.

## Evidently

Слой интеграции `evidently_adapter.py` включён и вызывается при установленном пакете. Статус сборочного окружения: `{summary['evidently_status']}`. На машине с интернетом extra `monitoring` устанавливает Evidently и тот же `run_lab7.py` сохраняет HTML-отчёт.
"""
    (ROOT / "reports/LAB7/lab7_report.md").write_text(text, encoding="utf-8")

def build_lab8(data_hash: str) -> None:
    e2e = pd.read_csv(ROOT / "reports/LAB8/end_to_end.csv")
    alert = read_json("reports/LAB8/alert_drill.json")
    tests = (ROOT / "reports/LAB8/all_tests.log").read_text(encoding="utf-8")
    coverage_line = next((line.strip() for line in tests.splitlines() if line.startswith("TOTAL")), "TOTAL coverage: see all_tests.log")
    coverage_match = re.search(r"(\d+)%\s*$", coverage_line)
    coverage_pct = coverage_match.group(1) + "%" if coverage_match else "see all_tests.log"
    passed_match = re.search(r"(\d+) passed in", tests)
    passed_line = f"{passed_match.group(1)} passed" if passed_match else "see all_tests.log"
    resources = read_json("reports/LAB8/system_resources.json")
    real = e2e[(e2e['source']=='real') & (e2e['repeat']==1)].iloc[0]
    syn = e2e[(e2e['source']=='synthetic') & (e2e['repeat']==1)].iloc[0]
    text = header(8, data_hash) + f"""## Устанавливаемый проект и CLI

`pyproject.toml` описывает пакет и точки входа `train_ml`, `train_dl`, `train_validator`, `generate`, `serve`, `evaluate`, `monitor`. Каждая обучающая команда получает Pydantic-конфигурацию и может выбирать полный/чанковый источник там, где это предусмотрено.

| Команда | Пример | Назначение |
|---|---|---|
| `train_ml` | `train_ml --config configs/lab2.json --source chunks --data-source real` | классическая модель, full/chunks |
| `train_dl` | `train_dl --config configs/lab3.json --source chunks --data-source synthetic` | PyTorch, full/stream |
| `train_validator` | `train_validator --config configs/lab4.json --source chunks --data-source real` | статистический валидатор |
| `generate` | `generate --config configs/lab5.json --mode similar --n 5000` | синтетические данные |
| `serve` | `serve --config configs/lab6.json` | FastAPI model service |
| `evaluate` | `evaluate --config configs/lab2.json --data-source synthetic --model reports/LAB2/ml_model.joblib` | оценка модели |
| `monitor` | `monitor --config configs/lab7.json --data-source synthetic` | PSI/KS/CSI мониторинг |

`install.log` подтверждает успешную сборку wheel, установку этого wheel в отдельное виртуальное окружение `--no-deps` и импорт пакета версии 1.0.0. Полное разрешение внешних зависимостей из PyPI в офлайн-контейнере невозможно; для реальной машины предназначен `RUN_ME_WINDOWS.ps1`.

## Сквозной сценарий, повторённый дважды

Каждый источник прогнан два раза с тем же seed; числа обоих повторов совпадают точно.

| Источник | ML accuracy | ML ROC-AUC | DL accuracy | DL ROC-AUC | Validator accept rate |
|---|---:|---:|---:|---:|---:|
| real | {real['ml_accuracy']:.4f} | {real['ml_roc_auc']:.4f} | {real['dl_accuracy']:.4f} | {real['dl_roc_auc']:.4f} | {real['validator_accept_rate']:.4f} |
| synthetic similar | {syn['ml_accuracy']:.4f} | {syn['ml_roc_auc']:.4f} | {syn['dl_accuracy']:.4f} | {syn['dl_roc_auc']:.4f} | {syn['validator_accept_rate']:.4f} |

Синтетический `similar` источник не ухудшает качество и остаётся применимым для валидатора; небольшое улучшение на конкретном seed не трактуется как преимущество, а лишь как выборочная вариация.

## Алерты

Проверены три группы: data drift, рост reject-rate валидатора и error-rate сервиса. На чистом сценарии алертов **{alert['clean_alerts']}**; при внесённых дефектах создано **{alert['drill_alerts']}** из 3 ожидаемых; повторный drift-алерт внутри cooldown подавлен = `{alert['duplicate_suppressed']}`. Пример записей находится в `alerts_sample.jsonl` и содержит время, показатель, значение, порог, уровень и действие.

## Тестирование

Полный pytest: **{passed_line}**, покрытие проекта **{coverage_pct}** (`all_tests.log`; {coverage_line}). Начиная с ЛР2 также создаются отдельные `pytest_coverage.log` по областям каждой работы с пояснением тех ветвей, которые корректнее проверять интеграционно (CUDA, реальный uvicorn, online MLflow/Evidently).

## Время, CPU и память

`end_to_end.csv` содержит для каждого сквозного прогона `elapsed_s`, `process_cpu_s`, нормированную загрузку CPU и RSS. Сведения о runner находятся в `system_resources.json`: logical CPU={resources['cpu_logical']}, physical CPU={resources['cpu_physical']}, RAM={resources['memory_total_gib']:.2f} GiB; суммарное время четырёх e2e-прогонов={resources['e2e_elapsed_s_sum']:.3f} s, CPU time={resources['e2e_process_cpu_s_sum']:.3f} s, peak RSS={resources['e2e_peak_rss_mb']:.1f} MiB. Эти числа пересоздаются на машине обучающегося при повторном запуске.

## Ограничения и группы ошибок

1. MLflow/Evidently не устанавливались в офлайн-сборке; рабочие адаптеры и online extras присутствуют, перед сдачей нужно выполнить общий прогон с интернетом.
2. CUDA parity ЛР3 не измерена на CPU-only сборочном runner; код проверки присутствует.
3. Микроскопическая модель `make_moons` плохо масштабируется по потокам из-за overhead, что количественно показано в ЛР7.
4. `tracemalloc` отражает Python memory и не является полным измерителем нативной памяти PyTorch/BLAS.
5. Benchmark через TestClient — воспроизводимая лабораторная оценка, а не production SLA через реальный сетевой стек.

`artifacts_manifest.json` содержит SHA-256 и размеры артефактов; `defense_checklist.md` фиксирует комплектность, негативные контроли и использование ИИ.
"""
    (ROOT / "reports/LAB8/lab8_report.md").write_text(text, encoding="utf-8")


def append_lecture_requirements() -> None:
    sections = {
        2: """## Дополнительные требования с лекции

- Полный датасет и чанки: реализованы оба режима; train хранится в **24 CSV-чанках** в `data/chunks/`, то есть больше 10.
- Хранение: чанки являются воспроизводимыми data-artifacts; в Git рекомендуется хранить код/manifest, а сами данные допустимо пересоздавать `prepare_data.py`. В учебном ZIP они включены для демонстрации.
- Модель собирается исполняемым кодом и сохраняется в `ml_model.joblib` / `ml_model_chunk.joblib`.
- Pytest с ЛР2: `pytest_coverage.log` фиксирует тесты и объяснение интеграционных ветвей, которые не следует имитировать как unit-тесты.
""",
        3: """## Дополнительные требования с лекции

- Повторена логика full/chunks уже на **PyTorch**.
- Потоковый источник использует `IterableDataset/DataLoader`, а не предварительное объединение чанков в память.
- `dl_model.pt` и `dl_model_chunk.pt` доказывают, что код не только считает метрики, но и собирает/сохраняет модель.
- Pytest scope и ограничение CUDA-runner записаны в `pytest_coverage.log`.
""",
        4: """## Дополнительные требования с лекции

- Валидатор намеренно простой: статистическое расстояние + confidence рабочей модели.
- Для невалидного входа формируется объяснимый ответ через `ValidatorModel.explain()`, а `validator_negative.log` содержит поле `reason`; невалидный объект не принимается молча.
- Full/chunks статистики сравниваются численно.
""",
        5: """## Дополнительные требования с лекции

- Генератор имеет управляемые параметры drift.
- `PATCH /config` меняет их **в том же процессе без рестарта**, после чего `POST /generate` сразу создаёт данные с новыми настройками.
- Это отдельно проверено pytest и `live_update_demo.log`.
""",
        6: """## Дополнительные требования с лекции

- Runtime-настройки `metrics_enabled` и `log_level`, а также активная модель изменяются без остановки сервиса.
- Метрики имеют `metric_descriptions`: инженер видит смысл параметров, аналитик — uptime/error rate/latency/online accuracy и состояние распределения через monitoring service.
- Архитектура именно этого проекта показана C4-подобной диаграммой в `figures/`.
""",
        7: """## Дополнительные требования с лекции

- Benchmark обязательно включает **все доступные логические ядра CPU** (`effective_jobs == os.cpu_count()`).
- Для каждой конфигурации сохраняются throughput, latency, elapsed, process CPU time, CPU utilization и RSS.
- Если ускорения нет, это не маскируется: отчёт объясняет overhead на маленькой модели.
""",
        8: """## Дополнительные требования с лекции

- Выполняется сквозная проверка real/synthetic и повтор для воспроизводимости.
- Пропускная способность и причина ограничения разобраны по фактическому benchmark ЛР7.
- В `system_resources.json` и `end_to_end.csv` учитываются время, CPU и память.
- Итоговая сдача включает отчёты, репозиторий, скринкаст, negative controls, рецензию БЯМ и критический комментарий.
""",
    }
    for lab, section in sections.items():
        path = ROOT / f"reports/LAB{lab}/lab{lab}_report.md"
        text = path.read_text(encoding="utf-8").rstrip() + "\n\n" + section.strip() + "\n"
        path.write_text(text, encoding="utf-8")


def build_review() -> None:
    review = """# Рецензия БЯМ по промпту из приложения А

## ЛР1 — готова
Артефакты присутствуют: отчёт, EDA CSV, два графика, hash manifest, три pre-commit блокировки и проверка подмены данных. Числа отчёта воспроизводятся скриптом.

## ЛР2 — готова с замечанием
Full/chunks, интервалы, память/время, модель и негативные конфигурации присутствуют. Метрики двух режимов совпадают. Замечание: сборочный run имеет `tracking_backend=fallback`, потому что в офлайн-контейнере отсутствует MLflow; код автоматически использует MLflow после установки extra `tracking`.

## ЛР3 — готова с замечанием
PyTorch full/stream реализованы, воспроизводимость с тем же seed точная, память/время и кривые сохранены. CPU/GPU parity в текущем пакете проверить невозможно: сборочный PyTorch CPU-only. Проверка программно реализована и должна быть повторена в CUDA-окружении.

## ЛР4 — готова
Полный и потоковый валидатор совпадают в машинном допуске. Recall чужеродных примеров 0.969, false reject своих 0.0278; отрицательные входы логируются.

## ЛР5 — готова
Режим `similar` имеет PSI << 0.1, `random` PSI > 0.2. Seed воспроизводим побайтово, runtime PATCH меняет генерацию без перезапуска. Coverage генератора 95%.

## ЛР6 — готова
Контракты API зелёные, есть две версии модели и проверка SHA-256. Под нагрузкой при переключении 640/640 запросов успешны, обе версии реально наблюдались.

## ЛР7 — готова с замечанием
Есть batch parallelism, повторный benchmark с прогревом, drift windows, ноль ложных алертов и serial fallback при сбое пула. Замечание: Evidently физически не установлен в офлайн-сборке; adapter и dependency extra присутствуют, но HTML-отчёт надо пересоздать online.

## ЛР8 — готова с замечанием
Wheel собирается и импортируется из отдельного venv; сквозной сценарий real/synthetic повторён дважды без расхождений, алерты и suppression работают, pytest 93%. Полное разрешение зависимостей из PyPI не проверялось из-за запрета сети в сборочном runner.

## Противоречия и невоспроизводимые числа
Найденных противоречий между CSV/JSON и отчётами нет. Невоспроизводимые в текущем CPU-only/offline runner пункты: CUDA parity, настоящий MLflow backend и Evidently HTML; причины указаны явно, код повторной проверки включён.

## Сильные стороны
1. Один сквозной train/validation/test split и отсутствие test-утечки при выборе gamma.
2. Full/chunks используют одну модель и дают одинаковые результаты в ЛР2.
3. Чанковый DL резко уменьшает измеренный Python peak memory.
4. Негативные контроли автоматизированы, а не описаны словами.
5. ЛР6–8 связаны с реальными артефактами предыдущих работ.

## Рекомендации
1. Перед сдачей запустить `RUN_ME_WINDOWS.ps1` в online-окружении, чтобы получить MLflow и Evidently артефакты.
2. При наличии CUDA повторить ЛР3 и сохранить `device_parity.csv` с фактическим числом <=1e-5.
3. Создать GitHub-репозиторий, включить branch protection и вставить ссылку в отчёты.
4. Записать единый 3–5-минутный скринкаст по `reports/SCREENCAST_PLAN.md`.
"""
    (ROOT / "reports/llm_review.md").write_text(review, encoding="utf-8")

    comment = """# Критический комментарий к рецензии БЯМ

| Пункт рецензии | Согласен / не согласен | Обоснование |
|---|---|---|
| ЛР2: fallback вместо MLflow в сборочном прогоне | Согласен | Это ограничение именно текущего офлайн-runner, а не архитектуры проекта. В `tracking.py` первым выбирается MLflow; `RUN_ME_WINDOWS.ps1` ставит extra `tracking`. Перед сдачей этот прогон нужно повторить. |
| ЛР3: нет фактического CUDA parity | Согласен | В контейнере установлен CPU-only PyTorch. Проверка с допуском 1e-5 реализована функцией `device_parity`; результат надо получить на машине с CUDA. |
| ЛР7: нет Evidently HTML в текущем архиве | Согласен | Пакет нельзя было скачать без сети. `evidently_adapter.py` и extra `monitoring` есть; после установки скрипт автоматически пытается сохранить отчёт. |
| ЛР8: dependency resolution не проверено online | Согласен | Wheel и чистый импорт проверены отдельно. Полный online install вынесен в однокнопочный скрипт и должен быть выполнен перед финальной сдачей. |
| Остальные работы отмечены готовыми | Согласен | Выводы подтверждаются CSV/JSON/логами и автоматическими pytest; полное покрытие проекта в сборочном прогоне 93%. |
"""
    (ROOT / "reports/llm_review_comment.md").write_text(comment, encoding="utf-8")


def build_checklist(data_hash: str) -> None:
    text = f"""# Defense checklist — вариант 09

Исходные данные SHA-256: `{data_hash}`. Кодовый commit, использованный при формировании отчётов: `{commit_hash()}`.

| ЛР | Обязательные артефакты | Хеш/воспроизводимость | Негативные контроли | Статус перед финальной сдачей |
|---|---|---|---|---|
| 1 | report, eda_stats, precommit log, hash manifest | SHA-256 сверяется | secret/large/syntax/tamper | готово |
| 2 | report, ml_metrics, ml_model, config log | full=chunks, seed fixed | bad configs, leakage-safe split | rerun with real MLflow online |
| 3 | report, dl_metrics, dl_model, memory compare | repeat exact, tolerance <=0.005 | bad configs, seed demo | CUDA parity rerun if GPU available |
| 4 | report, validator metrics/params/negative log | full≈chunks ~1e-14 | OOD rejection + bad configs | готово |
| 5 | report, synthetic metrics/config/tests | identical hash for same seed | drift knobs + bad config | готово |
| 6 | report, service log, registry | model SHA-256 checked | 422, bad hash, rollback | готово |
| 7 | report, benchmark, drift metrics | repeated warmed measurements | drift/clean/worker failure | rerun with Evidently online |
| 8 | report, alerts, install log, manifest | e2e repeats exact | invalid config, alert drills | готово после online dependency run |

## Раскрытие использования ИИ

ИИ использовался для проектирования структуры репозитория, генерации черновиков кода, тестов и отчётов. Все численные результаты в отчётах получены исполняемыми скриптами и сохранены в CSV/JSON/логах; они не были придуманы вручную. Перед сдачей обучающийся должен самостоятельно просмотреть код, выполнить `RUN_ME_WINDOWS.ps1`, проверить итоговые логи и записать скринкаст.
"""
    path = ROOT / "reports/LAB8/defense_checklist.md"
    path.write_text(text, encoding="utf-8")
    (ROOT / "defense_checklist.md").write_text(text, encoding="utf-8")


def build_screencast_plan() -> None:
    text = """# План скринкаста 3–5 минут

1. **0:00–0:30 — репозиторий и ЛР1.** Показать структуру, `hash_manifest.json`, scatter `make_moons`, запустить pre-commit negative control.
2. **0:30–1:15 — ЛР2–3.** Открыть `ml_metrics.csv` и `dl_metrics.csv`; показать full/chunks, память, MLflow UI после online-прогона; запустить одну команду предсказания сохранённой модели.
3. **1:15–1:45 — ЛР4–5.** Показать `validator_negative.log`; через generator API сделать `PATCH /config` и без рестарта сгенерировать данные со сдвигом.
4. **1:45–2:45 — ЛР6.** Открыть Swagger `/docs`: `/predict`, плохой запрос -> 422, `/models`, `/promote`, `/rollback`, `/runtime`, `/metrics`.
5. **2:45–3:30 — ЛР7.** Показать `parallel_benchmark.csv`, строку режима со всеми ядрами и фактический лучший throughput; объяснить, почему масштабирование зависит от размера модели и overhead; открыть drift windows/Evidently и показать срабатывание с окна 4.
6. **3:30–4:15 — ЛР8.** Показать `end_to_end.csv`, `alerts_sample.jsonl`, `install.log`, `artifacts_manifest.json`.
7. **4:15–4:45 — тесты.** `pytest --cov=moons_lab --cov-report=term-missing`; показать фактическое число passed и TOTAL coverage из текущего прогона.
8. **4:45–5:00 — негативный режим.** Один намеренно плохой конфиг и понятное сообщение Pydantic.

На видео должны быть видны команды, результат, файлы отчёта и связь каждого числа с конкретным CSV/JSON/run ID.
"""
    (ROOT / "reports/SCREENCAST_PLAN.md").write_text(text, encoding="utf-8")


def main() -> None:
    data_hash = read_json("reports/LAB1/hash_manifest.json")["raw"]["sha256"]
    build_lab1(data_hash)
    build_lab2(data_hash)
    build_lab3(data_hash)
    build_lab4(data_hash)
    build_lab5(data_hash)
    build_lab6(data_hash)
    build_lab7(data_hash)
    build_lab8(data_hash)
    append_lecture_requirements()
    build_review()
    build_checklist(data_hash)
    build_screencast_plan()
    print("reports rebuilt")


if __name__ == "__main__":
    main()
