# Вариант 09 — `make_moons`: нелинейность

Полный сквозной проект для восьми лабораторных работ. Один и тот же вариант используется от EDA и классического ML до PyTorch, валидатора, управляемой синтетики, FastAPI-сервиса, параллельного инференса, мониторинга дрейфа и итоговой сборки.

## Почему архитектура именно такая

`make_moons` — двумерная нелинейно разделимая задача бинарной классификации. Для ЛР2 основной классификатор — `RBFSampler + SGDClassifier(loss="log_loss")`: он даёт нелинейное RBF-преобразование и при этом поддерживает честное инкрементальное обучение через `partial_fit`, поэтому **одна и та же модель** обучается как из полностью загруженного датасета, так и из >10 файлов-чанков. Отдельно считается линейный `LogisticRegression` как наивный ориентир и подтверждение проблемы нелинейности.

## Быстрый запуск на Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\RUN_ME_WINDOWS.ps1
```

Скрипт создаёт `.venv`, ставит проект с дополнительными зависимостями, готовит данные, запускает ЛР1–8 и тесты. Если установлен пакет MLflow, все прогоны регистрируются в `mlruns/`; если MLflow временно недоступен, встроенный fallback не ломает расчёты и сохраняет параметры/метрики в `runs_fallback/`.

После полного прогона:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri .\mlruns --port 5000
```

## Быстрый запуск на Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[all]'
python scripts/run_all.py
pytest --cov=moons_lab --cov-report=term-missing
```


## Где отчёты и что сдавать

- Отчёты и все численные артефакты: `reports/LAB1/` … `reports/LAB8/`.
- Карта «что отправлять в какую лабораторную»: `SUBMISSION_MAP.md`.
- Все дополнительные требования, записанные на лекции: `LECTURE_REQUIREMENTS_MAP.md`.
- Архитектурная диаграмма именно варианта 09: `diagrams/c4_make_moons_architecture.png` + редактируемый `.drawio`.
- Запуск одной выбранной лабораторной с её prerequisites на Windows: `./RUN_LAB_WINDOWS.ps1 -Lab 6` (номер заменить).
- План скринкаста: `reports/SCREENCAST_PLAN.md`.
- Рецензия БЯМ и критический комментарий: `reports/llm_review.md`, `reports/llm_review_comment.md`.

## Структура

- `configs/` — Pydantic-конфигурации ЛР1–8.
- `data/raw/` — детерминированный исходный `make_moons`.
- `data/processed/` — train/validation/test без утечки.
- `data/chunks/` — 24 чанка обучающей выборки.
- `src/moons_lab/` — весь код проекта.
- `tests/` — pytest для вычислительной логики, API и негативных сценариев.
- `reports/LAB1..LAB8/` — обязательные артефакты и отчёты.
- `models/registry/` — версии моделей для promote/rollback.
- `scripts/` — воспроизводимые команды каждой лабораторной.

## Ключевые команды

```bash
python scripts/prepare_data.py
python scripts/run_lab1.py
python scripts/run_lab2.py
python scripts/run_lab3.py
python scripts/run_lab4.py
python scripts/run_lab5.py
python scripts/run_lab6.py
python scripts/run_lab7.py
python scripts/run_lab8.py
pytest --cov=moons_lab --cov-report=term-missing
```

## Важное про сборочный архив

Архив собран в офлайн-контейнере. Доступные зависимости использованы для фактических прогонов. `mlflow` и `evidently` не могли быть скачаны из PyPI в сборочном окружении, поэтому код имеет безопасный fallback и явно фиксирует это в отчётах. На машине с интернетом `RUN_ME_WINDOWS.ps1` ставит extras `tracking` и `monitoring`, после чего те же скрипты работают через настоящие MLflow/Evidently без изменения кода.
