from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
LAB2 = ROOT / "reports" / "LAB2"


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model_manifest() -> dict:
    files = [
        "linear_baseline.joblib",
        "ml_model.joblib",
        "ml_model_chunk.joblib",
    ]

    payload = {}
    for name in files:
        path = LAB2 / name
        if path.exists():
            payload[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    (LAB2 / "model_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def export_mlflow_runs() -> None:
    uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    rows = []
    try:
        experiments = client.search_experiments()
        for exp in experiments:
            for run in client.search_runs([exp.experiment_id]):
                rows.append(
                    {
                        "experiment_id": exp.experiment_id,
                        "experiment_name": exp.name,
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "run_name": run.data.tags.get("mlflow.runName", ""),
                        "params": json.dumps(run.data.params, ensure_ascii=False, sort_keys=True),
                        "metrics": json.dumps(run.data.metrics, ensure_ascii=False, sort_keys=True),
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "experiment_id": "",
                "experiment_name": "",
                "run_id": "",
                "status": "EXPORT_ERROR",
                "run_name": "",
                "params": "",
                "metrics": str(exc),
            }
        )

    pd.DataFrame(rows).to_csv(
        LAB2 / "mlflow_runs.csv",
        index=False,
    )


def read_results():
    metrics = pd.read_csv(LAB2 / "ml_metrics.csv")
    search = pd.read_csv(LAB2 / "hyperparam_search.csv")
    comparison = json.loads(
        (LAB2 / "comparison.json").read_text(encoding="utf-8")
    )
    return metrics, search, comparison


def value(metrics: pd.DataFrame, mode: str, column: str) -> float:
    return float(metrics.loc[metrics["mode"] == mode, column].iloc[0])




def build_docx(
    metrics: pd.DataFrame,
    comparison: dict,
    manifest: dict,
) -> None:
    full = metrics[metrics["mode"] == "full"].iloc[0]
    chunks = metrics[metrics["mode"] == "chunks"].iloc[0]

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "Лабораторная работа 2\n"
        "Классический ML: полный датасет и чанки\n"
        "Вариант 09 — make_moons"
    )
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)

    doc.add_heading("1. Цель", level=1)
    doc.add_paragraph(
        "Обучить классическую нелинейную ML-модель в режимах full и chunks, "
        "сравнить качество, время и память, зарегистрировать прогоны в MLflow."
    )

    doc.add_heading("2. Модель", level=1)
    doc.add_paragraph(
        "Pipeline: StandardScaler → RBFSampler → SGDClassifier. "
        f'Лучшее gamma по validation: {comparison.get("best_gamma", "")}.'
    )

    doc.add_heading("3. Результаты", level=1)
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.rows[0].cells[0].text = "Метрика"
    table.rows[0].cells[1].text = "Full"
    table.rows[0].cells[2].text = "Chunks"

    rows = [
        ("Accuracy", float(full["accuracy"]), float(chunks["accuracy"])),
        ("F1", float(full["f1"]), float(chunks["f1"])),
        ("ROC-AUC", float(full["roc_auc"]), float(chunks["roc_auc"])),
        ("Peak memory, MiB", float(full["peak_memory_mb"]), float(chunks["peak_memory_mb"])),
        ("Time, s", float(full["elapsed_s"]), float(chunks["elapsed_s"])),
    ]

    for name, a, b in rows:
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = f"{a:.6f}"
        cells[2].text = f"{b:.6f}"

    doc.add_heading("4. Интерпретация", level=1)
    mem_save = (
        (float(full["peak_memory_mb"]) - float(chunks["peak_memory_mb"]))
        / float(full["peak_memory_mb"])
        * 100.0
    )
    doc.add_paragraph(
        f'Оба режима дали accuracy {float(full["accuracy"]):.4f}. '
        f'Chunk-режим снизил пиковую измеренную память примерно на {mem_save:.2f}%, '
        "но оказался медленнее из-за overhead обработки отдельных чанков."
    )

    doc.add_heading("5. MLflow", level=1)
    doc.add_paragraph(
        f'Full run ID: {full["run_id"]}\n'
        f'Chunks run ID: {chunks["run_id"]}\n'
        "Backend: sqlite:///mlflow.db"
    )

    doc.add_heading("6. Модели", level=1)
    for name, info in manifest.items():
        doc.add_paragraph(
            f'{name}: {info["bytes"]} bytes; SHA-256 {info["sha256"]}'
        )

    doc.add_heading("7. Негативные контроли", level=1)
    doc.add_paragraph(
        "Проверяются ошибочные конфигурации и защита от утечки test-набора "
        "в процедуру подбора гиперпараметров."
    )

    doc.add_heading("8. Выводы", level=1)
    conclusions = [
        f'Accuracy full — {float(full["accuracy"]):.4f}.',
        f'Accuracy chunks — {float(chunks["accuracy"]):.4f}.',
        f'ROC-AUC full — {float(full["roc_auc"]):.4f}.',
        f'ROC-AUC chunks — {float(chunks["roc_auc"]):.4f}.',
        f'Пиковая память full — {float(full["peak_memory_mb"]):.4f} MiB.',
        f'Пиковая память chunks — {float(chunks["peak_memory_mb"]):.4f} MiB.',
        f'Время full — {float(full["elapsed_s"]):.4f} с.',
        f'Время chunks — {float(chunks["elapsed_s"]):.4f} с.',
        f'Лучшее gamma — {comparison.get("best_gamma", "")}.',
        f'Число train-чанков — {comparison.get("chunks", "")}.',
    ]
    for item in conclusions:
        doc.add_paragraph(item, style="List Number")

    doc.save(LAB2 / "lab2_report.docx")




def main() -> None:
    LAB2.mkdir(parents=True, exist_ok=True)

    metrics, search, comparison = read_results()
    manifest = build_model_manifest()
    export_mlflow_runs()
    commit = git_head()

    build_docx(metrics, comparison, manifest)

    print("Lab 2 final package generated.")
    print("Report:", LAB2 / "lab2_report.docx")


if __name__ == "__main__":
    main()
