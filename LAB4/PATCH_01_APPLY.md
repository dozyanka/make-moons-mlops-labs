# Патч №01 — только лабораторная №4

Архив: `01_LAB4_Validator_Patch.zip`.

**Адрес папки-назначения:** `C:\Users\shata\Desktop\make-moons-mlops-labs`.

## Установка (Windows)

**Вариант 1:** открыть архив, перетащить находящуюся внутри папку
`make-moons-mlops-labs` на **рабочий стол**, подтвердить объединение папок и
замену только совпадающих файлов из патча. Не удалять существующую папку!

**Вариант 2:** положить архив в «Загрузки», выполнить в PowerShell:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\01_LAB4_Validator_Patch.zip" -DestinationPath "$env:USERPROFILE\Desktop" -Force
```

После распаковки должен существовать файл
`C:\Users\shata\Desktop\make-moons-mlops-labs\LAB4\RUN_LAB4.cmd`.

Для запуска двойной щелчок по `LAB4\RUN_LAB4.cmd`. Никаких прав администратора
не нужно. При наличии интернета установятся зависимости, включая MLflow.

## Что заменяется

Существующие файлы ЛР4: `configs/lab4.json`, `src/moons_lab/config.py`,
`src/moons_lab/validator.py`, `scripts/run_lab4.py`.
Дополняются общий `README.md` и `defense_checklist.md`.
Создаются `LAB4/`, `.github/workflows/lab4-ci.yml`,
`defense_demo/lab4_demo.py`, `defense_demo/notes/LAB4_POINTS.md`,
`tests/test_lab4_workflow.py` и `reports/LAB4/`.

**Не входят:** папки `data/`, `reports/LAB1`, `LAB2`, `LAB3`, `.git/`,
`mlruns/`, модели ЛР2–3; патч **не уничтожает их** и не перегенерирует.

## После установки

1. Запустить `LAB4/RUN_LAB4.cmd` на своём Windows-компьютере.
2. Проверить `reports/LAB4/pytest_coverage.log` (all passed) и Run IDs в
   `reports/LAB4/validator_metrics.csv`: должны быть настоящие ID, а не
   `OFFLINE_NOT_MLFLOW`.
3. Открыть `reports/LAB4/lab4_report.md`, сравнить оба режима, показать отказ
   на чужих данных в `defense_demo/lab4_demo.py`.
4. До одобрения работы **не публиковать патч на GitHub**. Скрипт отправки будет
   отдельным файлом, после твоего подтверждения.

## Безопасность данных

В исходных архивах обнаружены отличия между desktop-версией и GitHub в
файлах данных и ЛР1. Этот патч основан **только** на desktop-версии и ничего
из этих расхождений не исправляет без отдельного согласования.
