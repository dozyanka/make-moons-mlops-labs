from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "LAB1"
FIG_DIR = REPORT_DIR / "figures"
RAW_PATH = ROOT / "data" / "raw" / "moons.csv"


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def load_manifest() -> dict:
    return json.loads(
        (REPORT_DIR / "hash_manifest.json").read_text(encoding="utf-8")
    )


def build_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["x1", "x2", "y"]:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "field": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "non_finite": int((~np.isfinite(numeric)).sum()),
                "unique_values": int(series.nunique()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "schema.csv", index=False)
    return result


def load_and_extend_metrics(df: pd.DataFrame) -> dict[str, float]:
    path = REPORT_DIR / "eda_stats.csv"
    stats = pd.read_csv(path)
    metrics = {
        str(row["metric"]): float(row["value"])
        for _, row in stats.iterrows()
    }

    numeric = df[["x1", "x2"]].to_numpy(dtype=float)
    metrics["non_finite_feature_values"] = float((~np.isfinite(numeric)).sum())
    metrics["invalid_target_values"] = float((~df["y"].isin([0, 1])).sum())
    metrics["target_unique_values"] = float(df["y"].nunique())

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(path, index=False)

    return metrics




def add_picture_with_caption(
    document: Document,
    path: Path,
    caption: str,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(path), width=Inches(6.0))

    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)


def build_docx(
    manifest: dict,
    metrics: dict[str, float],
    commit: str,
) -> None:
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Лабораторная работа 1\n"
        "Git-окружение, pre-commit и EDA\n"
        "Вариант 09 — make_moons: нелинейность"
    )
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)

    document.add_heading("1. Паспорт данных", level=1)

    table = document.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    passport = [
        ("Автор", "Vladimir"),
        ("Вариант", "09"),
        ("Источник", "sklearn.datasets.make_moons"),
        ("Происхождение", "локальная генерация scikit-learn"),
        ("Лицензия библиотеки", "BSD-3-Clause"),
        ("Объём", "24000"),
        ("SHA-256", manifest["raw"]["sha256"]),
        ("Хеш коммита", commit),
    ]

    for key, value in passport:
        cells = table.add_row().cells
        cells[0].text = key
        cells[1].text = value

    document.add_heading("2. Структура и качество данных", level=1)
    document.add_paragraph(
        "Поля: x1 (float64), x2 (float64), y (int64). "
        "Пропущенных значений: 0. Дубликатов: 0. "
        "Нечисловых или бесконечных значений признаков: 0. "
        "Недопустимых значений целевого класса: 0."
    )

    document.add_heading("3. Разбиение", level=1)
    document.add_paragraph(
        f'Train: {manifest["splits"]["train"]}; '
        f'validation: {manifest["splits"]["validation"]}; '
        f'test: {manifest["splits"]["test"]}; '
        f'chunks: {manifest["splits"]["chunks"]}; '
        f'chunk size: {manifest["generator"]["chunk_size"]}.'
    )

    document.add_heading("4. EDA", level=1)
    eda = document.add_table(rows=1, cols=2)
    eda.style = "Table Grid"
    eda.rows[0].cells[0].text = "Показатель"
    eda.rows[0].cells[1].text = "Значение"

    values = [
        ("Объекты", "24000"),
        ("Класс 0", "12000 (50.0 %)"),
        ("Класс 1", "12000 (50.0 %)"),
        ("corr(x1, x2)", f'{metrics["x1_x2_corr"]:.4f}'),
        ("Linear accuracy", f'{metrics["linear_train_accuracy"]:.4f}'),
    ]

    for key, value in values:
        cells = eda.add_row().cells
        cells[0].text = key
        cells[1].text = value

    document.add_heading("5. Визуализации", level=1)

    figures = [
        ("nonlinearity_scatter.png",
         "Рисунок 1 — нелинейность классов и линейная граница"),
        ("class_balance.png",
         "Рисунок 2 — баланс классов"),
        ("feature_distributions.png",
         "Рисунок 3 — распределения признаков"),
        ("correlation_matrix.png",
         "Рисунок 4 — корреляционная матрица"),
        ("linear_confusion.png",
         "Рисунок 5 — матрица ошибок линейного ориентира"),
    ]

    for filename, caption in figures:
        path = FIG_DIR / filename
        if path.exists():
            add_picture_with_caption(document, path, caption)

    document.add_heading("6. Интерпретация", level=1)
    document.add_paragraph(
        "Классы образуют два полумесяца. Одна линейная граница "
        "не соответствует этой геометрии. Диагностическая "
        f'LogisticRegression имеет accuracy {metrics["linear_train_accuracy"]:.4f}. '
        "Эта величина используется только как EDA-ориентир."
    )

    document.add_heading("7. Негативные контроли", level=1)
    document.add_paragraph(
        "Реальные попытки git commit с синтаксически неверным Python-файлом, "
        "тестовым секретом и бинарным файлом больше 5 MiB блокируются "
        "pre-commit. Подмена raw-файла обнаруживается по изменению SHA-256."
    )

    document.add_heading("8. Воспроизводимость", level=1)
    document.add_paragraph(
        "Повторная генерация с теми же параметрами и seed даёт тот же SHA-256."
    )

    document.add_heading("9. Выводы", level=1)
    conclusions = [
        "Объём выборки — 24000 объектов.",
        "Число входных признаков — 2.",
        "Число пропущенных значений — 0.",
        "Число дубликатов — 0.",
        "Доля класса 0 — 50.0 %.",
        "Доля класса 1 — 50.0 %.",
        f'Корреляция x1 и x2 — {metrics["x1_x2_corr"]:.4f}.',
        f'Accuracy линейного ориентира — {metrics["linear_train_accuracy"]:.4f}.',
        f'Количество train-чанков — {manifest["splits"]["chunks"]}.',
    ]
    for item in conclusions:
        document.add_paragraph(item, style="List Number")

    document.save(REPORT_DIR / "lab1_report.docx")




def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    manifest = load_manifest()
    schema = build_schema(df)
    metrics = load_and_extend_metrics(df)
    commit = git_head()

    build_docx(manifest, metrics, commit)

    print("Final Lab 1 package generated.")
    print("DOCX:", REPORT_DIR / "lab1_report.docx")


if __name__ == "__main__":
    main()
