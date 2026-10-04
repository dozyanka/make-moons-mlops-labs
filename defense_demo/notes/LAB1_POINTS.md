# ЛР1 — требования задания по пунктам и что показывать

Вариант 09 относится к профилю P1 — табличные данные. Поэтому для EDA
важны CSV-чанки, доли классов, корреляции и классификационные метрики.

## Цель

Развернуть Git-репозиторий с защищённым порядком изменений и сделать EDA,
на котором видна проблема варианта 09 — нелинейность `make_moons`.

## Пункт 1. Репозиторий, ветки, защита main, шаблон PR, gitignore/gitattributes

**Что сделано:** проект ведётся в GitHub-репозитории
`dozyanka/make-moons-mlops-labs`. Для `main` создан ruleset `Protect main`,
изменения последующих работ проходят через отдельную ветку и Pull Request.
Настроены `.gitignore`, `.gitattributes`, GitHub Actions и PR-шаблон.

**Как показать:** GitHub → Settings → Rules → Rulesets; затем открыть историю
PR и показать PR #1/PR #2. В локальном проекте показать `.gitignore`,
`.gitattributes` и `.github/`.

## Пункт 2. Pre-commit: стиль, секреты, большие файлы, быстрые проверки

**Что сделано:** `.pre-commit-config.yaml` вызывает локальные проверки из
`scripts/precommit_checks.py`. Проверяются синтаксис/базовый стиль Python,
случайно попавшие credentials и файлы больше допустимого размера.
Быстрые проверки проекта дополнительно выполняются pytest и GitHub Actions.

**Как показать:** открыть `.pre-commit-config.yaml`,
`reports/LAB1/precommit_block.log` и `reports/LAB1/precommit_clean.log`.
В block-log сохранены реальные отклонённые commit для плохого Python-файла,
тестового секрета и файла больше 5 MiB.

**Что такое секрет:** пароль, API key, access token, private key или другая
credential-строка, которую нельзя публиковать в Git.

## Пункт 3. Данные и хеш

**Что сделано:** для варианта 09 внешний CSV не скачивается: источник данных —
детерминированный генератор `sklearn.datasets.make_moons`.
Создан `data/raw/moons.csv`, параметры генерации зафиксированы, SHA-256
сохранён в `reports/LAB1/hash_manifest.json`.

**Как показать:** открыть `hash_manifest.json`, затем
`hash_negative_test.log`: при подмене raw-файла хеш меняется и подмена
обнаруживается.

## Пункт 4. EDA

**Что сделано:** проверены объём, поля `x1/x2/y`, типы, пропуски, дубликаты,
finite-values, допустимые значения `y`, распределения, баланс классов и
корреляции.

**Как показать:** открыть `reports/LAB1/eda_stats.csv` и `schema.csv`.

Ключевые числа:
- 24 000 объектов;
- 2 входных признака;
- 0 пропусков;
- 0 дубликатов;
- классы 50% / 50%;
- corr(x1, x2) ≈ -0.399.

## Пункт 5. Графики и проблема варианта

**Что сделано:** в `reports/LAB1/figures/` лежат баланс классов,
распределения признаков, correlation matrix, confusion matrix и главный
`nonlinearity_scatter.png`.

На главном графике LogisticRegression проводит линейную границу между двумя
полумесяцами. Accuracy ≈ 0.8679, поэтому видно, что одна прямая не описывает
нелинейную геометрию.

**Как показать:** открыть `nonlinearity_scatter.png`, затем запустить:

```powershell
python .\defense_demo\lab1_demo.py
```

Demo загружает сохранённую LogisticRegression и получает ту же accuracy,
что записана в `eda_stats.csv`.

## Пункт 6. Машиночитаемые сводки и отчёт

**Что сделано:** `eda_stats.csv`, `schema.csv`, `hash_manifest.json`,
`lab1_report.md`, `lab1_report.docx`. Выводы отчёта привязаны к числам.

**Как показать:** открыть Word/Markdown-отчёт и рядом `eda_stats.csv`.

## Негативные контроли

- секрет → commit заблокирован;
- файл больше лимита → commit заблокирован;
- плохой Python → commit заблокирован;
- подмена raw CSV → SHA-256 не совпадает;
- повторная генерация с seed=42 → исходный SHA-256 повторяется.

## Артефакты задания

- `reports/LAB1/lab1_report.md`
- `reports/LAB1/eda_stats.csv`
- `reports/LAB1/precommit_config.yaml`
- `reports/LAB1/precommit_block.log`
- `reports/LAB1/hash_manifest.json`
