# Live defense demo — ЛР1, ЛР2, ЛР3

Эта папка нужна для живой демонстрации уже обученных моделей на тех же данных,
на которых получены результаты в отчётах.

## Самый простой запуск

Двойной клик:

`defense_demo/RUN_ALL_DEMOS.cmd`

или:

```powershell
.\.venv\Scripts\python.exe .\defense_demo\run_all.py
```

Скрипт последовательно показывает ЛР1, ЛР2 full/chunks, ЛР3 full/streaming
и сверяет реальные метрики с CSV из `reports/`.

Если реальные значения не совпадают с отчётом, demo завершится ошибкой.

## Модели

- `models/lab1_logistic.joblib` — диагностическая LogisticRegression из ЛР1;
- `models/lab2_full.joblib` — full-модель ЛР2;
- `models/lab2_chunks.joblib` — chunk-модель ЛР2;
- `models/lab3_full.pt` — full checkpoint ЛР3;
- `models/lab3_streaming.pt` — streaming checkpoint ЛР3.

## Запуск по одной работе

```powershell
python .\defense_demo\lab1_demo.py
python .\defense_demo\lab2_demo.py --mode full
python .\defense_demo\lab2_demo.py --mode chunks
python .\defense_demo\lab3_demo.py --mode full
python .\defense_demo\lab3_demo.py --mode streaming
python .\defense_demo\stream_memory_demo.py
```

## Что показывать преподавателю

Сначала запускается соответствующий demo. Затем открывается файл
`notes/LAB{N}_POINTS.md`, где требования задания разобраны пункт за пунктом.
