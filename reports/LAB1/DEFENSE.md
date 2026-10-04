# Защита ЛР1

## Короткое вступление

У меня вариант 09 — make_moons. Основная проблема — нелинейность.

Внешний CSV для этого варианта не скачивается: make_moons является встроенным генератором scikit-learn. Я фиксирую параметры генерации, сохраняю raw CSV и его SHA-256.

## Основные числа

- объектов: 24000;
- train: 16800;
- validation: 3600;
- test: 3600;
- чанков: 24;
- классы: 50 % / 50 %;
- corr(x1,x2): -0.3989;
- accuracy линейного ориентира: 0.8679.

## Порядок демонстрации

1. GitHub: репозиторий, последние коммиты, Actions, Protect main.
2. `hash_manifest.json`: источник, параметры, размер, SHA-256.
3. `eda_stats.csv` и `schema.csv`.
4. `nonlinearity_scatter.png`: два полумесяца и линейная граница.
5. Остальные графики: баланс, распределения, корреляции, confusion matrix.
6. `precommit_block.log`: три заблокированных git commit.
7. `hash_negative_test.log`: подмена данных обнаружена.
8. `reproducibility.log`: повторная генерация дала тот же SHA-256.
9. `tests.log`: 3 passed.
10. `lab1_report.docx`.

## Что говорить про главный график

Два класса образуют два полумесяца. Одна прямая граница не может полностью разделить такую геометрию. Поэтому заявленная проблема варианта — нелинейность — видна непосредственно на данных.

Accuracy 0.8679 — это не итоговая оценка ML-модели, а диагностический линейный ориентир EDA.

## Команды на защите

```powershell
git log --oneline -5
git status
Get-Content .\reports\LAB1\hash_manifest.json
Get-Content .\reports\LAB1\eda_stats.csv
Get-Content .\reports\LAB1\precommit_block.log
Get-Content .\reports\LAB1\hash_negative_test.log
Get-Content .\reports\LAB1\reproducibility.log
python -m pytest .\tests\test_data.py .\tests\test_eda.py -q
explorer .\reports\LAB1\figures
```
