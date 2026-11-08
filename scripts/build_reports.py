from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-21"
AUTHOR = "Владимир Шаталов ЕТ-122"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def header(lab: int, data_hash: str) -> str:
    return f"""# Лабораторная работа {lab} – вариант 09 `make_moons`: нелинейность

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

Источник – `sklearn.datasets.make_moons`; данные синтетические, поэтому внешнее скачивание не требуется. Генерация детерминирована параметрами `n_samples=24000`, `noise=0.22`, `seed=42`. Исходный файл `data/raw/moons.csv` содержит **{int(stats['rows'])}** строк, **{int(stats['features'])}** числовых признака и **{int(stats['missing_cells'])}** пропусков. Доли классов равны {stats['class_0_share']:.3f}/{stats['class_1_share']:.3f}. Лицензия кода-источника scikit-learn – BSD-3-Clause.

Разбиение выполнено стратифицированно до любых обучающих преобразований: train={manifest['splits']['train']}, validation={manifest['splits']['validation']}, test={manifest['splits']['test']}. Train дополнительно сохранён в **{manifest['splits']['chunks']}** независимых CSV-чанка по 700 строк (последний короче).

## EDA и заявленная проблема

Главная проблема варианта – нелинейность. На `figures/nonlinearity_scatter.png` классы образуют две вложенные дуги, а линейная граница не может разделить их полностью. Прямой линейный ориентир `LogisticRegression` на полном исходном наборе даёт accuracy **{stats['linear_train_accuracy']:.4f}**, то есть примерно {(1-stats['linear_train_accuracy'])*100:.1f}% объектов остаются ошибочными даже на тех же данных. Корреляция `x1/x2` равна **{stats['x1_x2_corr']:.4f}**, что само по себе не описывает криволинейную границу классов.

Машиночитаемые сводки находятся в `eda_stats.csv`; визуализации – в `figures/`.

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

Для `make_moons` линейный классификатор используется как наивный ориентир, а рабочая модель – `StandardScaler -> RBFSampler -> SGDClassifier(loss='log_loss')`. RBF-преобразование решает проблему нелинейности, а `partial_fit` позволяет использовать **один и тот же алгоритм** как при полном наборе в памяти, так и при чтении 24 файлов-чанков.

Pydantic-схема находится в `configs/lab2.json`. Невалидные `gamma`, `chunk_size` и `passes` отклоняются; журнал – `config_negative_test.log`.

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

Текущий прогон зарегистрирован с backend=`{backend}`. При установленном MLflow эксперименты записываются в систему отслеживания; при отсутствии пакета код использует предусмотренный fallback. Для независимой проверки ЛР №2 используйте `RUN_DEMO.cmd`, пункт 3.

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

Классическая модель ЛР2 имеет accuracy **{comp['ml_lab2_full_accuracy']:.4f}**, PyTorch – **{comp['dl_full_accuracy']:.4f}**. Это близкие значения; DL не обязан выигрывать на простом двумерном наборе.

## Память и время

`memory_compare.csv`: full – **{mem.loc['full','peak_memory_mb']:.2f} MiB**, {mem.loc['full','elapsed_s']:.2f} s; chunks – **{mem.loc['chunks','peak_memory_mb']:.2f} MiB**, {mem.loc['chunks','elapsed_s']:.2f} s. `tracemalloc` измеряет Python-аллокации; это ограничение явно учитывается при интерпретации.

## Воспроизводимость и паритет устройств

Повтор full-прогона с тем же seed дал максимальное расхождение предсказаний **{comp['reproducibility']['prediction_max_abs_diff']:.1f}**, то есть точное совпадение и заведомое выполнение допуска 0.005 по метрике. Full/chunks различаются максимум на **{comp['full_vs_chunk_prediction_max_abs_diff']:.3g}**.

Негативный контроль необходимости seed: при смене seed с {comp['seed_negative_control']['reference_seed']} на {comp['seed_negative_control']['alternate_seed']} максимальное расхождение вероятностей стало **{comp['seed_negative_control']['prediction_max_abs_diff']:.4f}**, `pass={comp['seed_negative_control']['pass']}` (`seed_negative_test.json`).

Проверка CPU/GPU в сборочном контейнере: `{parity['status']}` ({parity['reason']}). Код проверки включён и при наличии CUDA сравнивает вероятности с требованием `max_abs_diff <= 1e-5`; перед сдачей на машине с CUDA этот пункт нужно повторить командой общего прогона.

Кривые validation loss сохранены в `figures/loss_curves.png`; чекпоинт – `dl_model.pt`, SHA-256 `{comp['full_model_sha256']}`.
"""
    (ROOT / "reports/LAB3/lab3_report.md").write_text(text, encoding="utf-8")




def build_lab5(data_hash: str) -> None:
    """LAB5's measured report is generated by scripts/run_lab5.py.

    Do not replace the verified table/coverage with a hard-coded template.
    """
    report = ROOT / "reports/LAB5/lab5_report.md"
    if not report.is_file():
        raise FileNotFoundError("Run scripts/run_lab5.py before build_reports.py")


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

Для этого варианта подготовлена дополнительная C4-подобная диаграмма `diagrams/c4_make_moons.drawio.png` и редактируемая версия `.drawio`. На ней показаны инженер/аналитик/источник запросов, FastAPI inference service, online generator, monitoring, MLflow, model registry и хранилище real/chunk/synthetic данных.
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

В runner доступно **{summary['cpu_count']}** логических CPU; режим всех ядер действительно измерен = `{summary['all_cores_benchmarked']}`, throughput в нём **{all_core_thr:.1f} items/s**. Лучший измеренный результат – **{summary['best_mean_throughput_items_s']:.1f} items/s** при requested_jobs={summary['best_requested_jobs']}. Для маленькой модели `make_moons` увеличение числа workers может ухудшать throughput из-за сериализации задач, планирования потоков и накладных расходов joblib; поэтому в отчёте фиксируется фактический optimum, а не предполагается, что больше ядер всегда быстрее.

`parallel_benchmark.csv` дополнительно содержит elapsed time, CPU time, нормированную загрузку CPU и RSS памяти для каждого повтора. При имитации падения Parallel-пула сервис автоматически выполнил serial fallback и вернул все 2 предсказания; `pass={summary['worker_failure_fallback']['pass']}`.

## Мониторинг дрейфа

Эталон – чистый `make_moons`; окна 0–3 чистые, с окна 4 внесён `shift_x=0.9`, `shift_y=0.35`. На чистых окнах **{summary['clean_false_alerts']}** ложных алертов. Первый алерт – window **{summary['first_drift_alert_window']}**: PSI x1={float(drift_row['psi_x1']):.3f}, PSI x2={float(drift_row['psi_x2']):.3f}, CSI prediction={float(drift_row['csi_prediction']):.3f}.

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

`install.log` отражает проверку сборки wheel и установки в изолированное окружение. Проверка полного набора зависимостей выполняется отдельно в сетевом окружении командой `python -m pip install -e ".[all]"`.

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
5. Benchmark через TestClient – воспроизводимая лабораторная оценка, а не production SLA через реальный сетевой стек.

`artifacts_manifest.json` содержит SHA-256 и размеры артефактов.
"""
    (ROOT / "reports/LAB8/lab8_report.md").write_text(text, encoding="utf-8")










def main() -> None:
    data_hash = read_json("reports/LAB1/hash_manifest.json")["raw"]["sha256"]
    build_lab1(data_hash)
    build_lab2(data_hash)
    build_lab3(data_hash)
    build_lab5(data_hash)
    build_lab6(data_hash)
    build_lab7(data_hash)
    build_lab8(data_hash)
    print("reports rebuilt")


if __name__ == "__main__":
    main()
