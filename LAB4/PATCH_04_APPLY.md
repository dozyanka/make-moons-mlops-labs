# Патч №04: исправление WinError 5 при запуске pytest

## Что произошло

В патче №03 полный эксперимент с MLflow выполнялся, но шесть тестов с
фикстурой `tmp_path` останавливались **до выполнения тела теста**:

- `C:\Users\shata\AppData\Local\Temp\pytest-of-shata` — `PermissionError [WinError 5]`.
- `.pytest_cache` внутри проекта — `PytestCacheWarning [WinError 5]`.

Это права доступа Windows на старые папки, а не ошибка методов валидатора.
Возможно, папки были созданы ранее другим пользователем или при запуске от
администратора, но точную причину по одному traceback установить нельзя.

## Что изменено

`LAB4/RUN_LAB4.ps1` создаёт **новую уникальную директорию для каждого
тестового запуска**: сначала в `reports/LAB4/.pytest_runtime/`, а если там
нет прав, то в `%LOCALAPPDATA%/make-moons-mlops-labs/lab4-tests/`.

Туда перенаправлены:

1. `pytest --basetemp=...` — временные каталоги `tmp_path`;
2. `pytest -o cache_dir=...` — кэш `pytest` вместо старой `.pytest_cache`;
3. `TEMP`, `TMP`, `TMPDIR` — временные файлы сторонних библиотек;
4. `COVERAGE_FILE` — файл `.coverage` вместо старого файла в корне проекта.

После успешного запуска директория удаляется. При ошибке она остаётся для
диагностики, а `reports/LAB4/.pytest_runtime/` исключён из Git.

GitHub Actions также передаёт явные пути `--basetemp` и `cache_dir`, сохраняя
одинаковый способ запуска тестов в Windows и Linux.

## Как применить

1. Из архива `04_LAB4_Pytest_Windows_Permissions_Patch.zip` перенести папку
   `make-moons-mlops-labs` на рабочий стол, подтвердить объединение и замену.
2. **Не удалять** `.venv`, старые данные, MLflow-прогоны, `.pytest_cache` или
   `C:\Users\shata\AppData\Local\Temp\pytest-of-shata`.
3. Запустить `LAB4/RUN_LAB4.cmd` обычным двойным щелчком (**не от администратора**).
4. Проверить, что в конце появились `... passed`, две таблицы coverage,
   `=== LAB4 COMPLETE ===`.
5. В `reports/LAB4/repeat_verification.json` должно быть `"all_pass": true`.
   MLflow-прогоны предыдущих запусков сохранятся. Новый запуск добавит ещё
   четыре прогона, а отчёт обновится ссылками на последние.

Если другие тесты теперь упадут **по новой причине**, приложить последний
фрагмент консоли и `reports/LAB4/pytest_coverage.log`.

Запуск без переустановки пакетов можно сделать из PowerShell, находясь в
корне проекта:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\LAB4\RUN_LAB4.ps1" -SkipInstall
```

Этот патч не публикует файлы в GitHub и не изменяет ЛР1–3.
