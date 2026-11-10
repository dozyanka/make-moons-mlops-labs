from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import torch
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
LAB3 = ROOT / "reports" / "LAB3"
FIGURE = LAB3 / "figures" / "loss_curves.png"


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_parameter_count(path: Path) -> int:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        state = obj.get("model_state_dict")
        if state is None:
            state = obj.get("state_dict")
        if state is None:
            state = obj
    else:
        return 0

    return int(
        sum(
            value.numel()
            for value in state.values()
            if torch.is_tensor(value)
        )
    )


def model_manifest() -> dict:
    payload = {}
    for name in ["dl_model.pt", "dl_model_chunk.pt"]:
        path = LAB3 / name
        if path.exists():
            payload[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tensor_parameter_count": checkpoint_parameter_count(path),
            }

    (LAB3 / "model_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def environment_info() -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }
    (LAB3 / "environment.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return info


def export_mlflow_runs() -> None:
    uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    rows = []
    try:
        exp = client.get_experiment_by_name("lab3_dl")
        if exp is not None:
            for run in client.search_runs([exp.experiment_id]):
                rows.append(
                    {
                        "experiment_id": exp.experiment_id,
                        "experiment_name": exp.name,
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "run_name": run.data.tags.get("mlflow.runName", ""),
                        "params": json.dumps(
                            run.data.params,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "metrics": json.dumps(
                            run.data.metrics,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "experiment_id": "",
                "experiment_name": "lab3_dl",
                "run_id": "",
                "status": "EXPORT_ERROR",
                "run_name": "",
                "params": "",
                "metrics": str(exc),
            }
        )

    pd.DataFrame(rows).to_csv(LAB3 / "mlflow_runs.csv", index=False)


def load_data():
    metrics = pd.read_csv(LAB3 / "dl_metrics.csv")
    memory = pd.read_csv(LAB3 / "memory_compare.csv")
    comparison = json.loads(
        (LAB3 / "comparison.json").read_text(encoding="utf-8")
    )
    reproducibility = json.loads(
        (LAB3 / "reproducibility.json").read_text(encoding="utf-8")
    )
    seed_negative = json.loads(
        (LAB3 / "seed_negative_test.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (ROOT / "configs" / "lab3.json").read_text(encoding="utf-8")
    )
    lab2 = pd.read_csv(ROOT / "reports" / "LAB2" / "ml_metrics.csv")
    return (
        metrics,
        memory,
        comparison,
        reproducibility,
        seed_negative,
        config,
        lab2,
    )


def metric(df: pd.DataFrame, mode: str, name: str) -> float:
    return float(df.loc[df["mode"] == mode, name].iloc[0])


def memory_value(df: pd.DataFrame, mode: str, name: str) -> float:
    return float(df.loc[df["mode"] == mode, name].iloc[0])


def parity_text(comparison: dict) -> str:
    parity = comparison.get("device_parity", {})
    if parity.get("status") == "not_available":
        return (
            "CPU/GPU parity could not be measured in this environment because "
            "CUDA was unavailable. This limitation is explicitly recorded in "
            "`device_parity.csv` and `environment.json`."
        )

    diff = parity.get("max_abs_diff")
    return (
        f"CPU/GPU prediction max absolute difference: {diff}. "
        "Required tolerance: 1e-5."
    )




def add_picture(doc: Document, path: Path, caption: str) -> None:
    if not path.exists():
        return

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(6.0))

    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = c.add_run(caption)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)


def build_docx(
    metrics: pd.DataFrame,
    memory: pd.DataFrame,
    comparison: dict,
    reproducibility: dict,
    config: dict,
    lab2: pd.DataFrame,
    manifest: dict,
    env: dict,
) -> None:
    full = metrics[metrics["mode"] == "full"].iloc[0]
    stream = metrics[metrics["mode"] == "chunks"].iloc[0]
    full_mem = memory[memory["mode"] == "full"].iloc[0]
    stream_mem = memory[memory["mode"] == "chunks"].iloc[0]

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        "Лабораторная работа 3\n"
        "PyTorch: полный датасет и streaming\n"
        "Вариант 09 — make_moons"
    )
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)

    doc.add_heading("1. Цель", level=1)
    doc.add_paragraph(
        "Построить воспроизводимый PyTorch-пайплайн MLP 2→32→32→1 "
        "и сравнить обучение на полном train-наборе со streaming по чанкам."
    )

    doc.add_heading("2. Конфигурация", level=1)
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    items = [
        ("Seed", str(config["seed"])),
        ("Batch size", str(config["batch_size"])),
        ("Epochs", str(config["epochs"])),
        ("Learning rate", str(config["learning_rate"])),
        ("Weight decay", str(config["weight_decay"])),
        ("Patience", str(config["patience"])),
        ("PyTorch", str(env["torch_version"])),
        ("CUDA", str(env["cuda_available"])),
        ("GPU", str(env["gpu_name"])),
    ]

    for key, value in items:
        cells = table.add_row().cells
        cells[0].text = key
        cells[1].text = value

    doc.add_heading("3. Метрики full vs streaming", level=1)
    t = doc.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.rows[0].cells[0].text = "Показатель"
    t.rows[0].cells[1].text = "Full"
    t.rows[0].cells[2].text = "Streaming"

    rows = [
        ("Accuracy", float(full["accuracy"]), float(stream["accuracy"])),
        ("F1", float(full["f1"]), float(stream["f1"])),
        ("ROC-AUC", float(full["roc_auc"]), float(stream["roc_auc"])),
        ("Peak memory, MiB", float(full_mem["peak_memory_mb"]), float(stream_mem["peak_memory_mb"])),
        ("Time, s", float(full_mem["elapsed_s"]), float(stream_mem["elapsed_s"])),
    ]

    for name, a, b in rows:
        cells = t.add_row().cells
        cells[0].text = name
        cells[1].text = f"{a:.6f}"
        cells[2].text = f"{b:.6f}"

    doc.add_heading("4. Кривые обучения", level=1)
    add_picture(
        doc,
        FIGURE,
        "Рисунок 1 — train/validation loss для full и streaming режимов",
    )

    doc.add_heading("5. Воспроизводимость", level=1)
    doc.add_paragraph(
        f'Prediction max abs diff при повторе с тем же seed: '
        f'{float(reproducibility["prediction_max_abs_diff"]):.6f}. '
        f'Проверка: {reproducibility["pass"]}.'
    )

    doc.add_heading("6. Device parity", level=1)
    doc.add_paragraph(parity_text(comparison))

    doc.add_heading("7. Сравнение с ЛР2", level=1)
    ml_acc = float(lab2.loc[lab2["mode"] == "full", "accuracy"].iloc[0])
    doc.add_paragraph(
        f'Accuracy классического ML ЛР2: {ml_acc:.4f}. '
        f'Accuracy MLP full: {float(full["accuracy"]):.4f}.'
    )

    doc.add_heading("8. Checkpoints", level=1)
    for name, info in manifest.items():
        doc.add_paragraph(
            f'{name}: {info["bytes"]} bytes; SHA-256 {info["sha256"]}'
        )

    doc.add_heading("9. Вывод", level=1)
    doc.add_paragraph(
        "Оба режима дают практически одинаковое качество. "
        "Streaming существенно снижает измеренную пиковую память. "
        "Повтор с тем же seed детерминирован в текущем CPU-режиме."
    )

    doc.save(LAB3 / "lab3_report.docx")




def main() -> None:
    LAB3.mkdir(parents=True, exist_ok=True)

    (
        metrics,
        memory,
        comparison,
        reproducibility,
        seed_negative,
        config,
        lab2,
    ) = load_data()

    manifest = model_manifest()
    env = environment_info()
    export_mlflow_runs()
    commit = git_head()

    build_docx(
        metrics,
        memory,
        comparison,
        reproducibility,
        config,
        lab2,
        manifest,
        env,
    )

    print("Lab 3 final package generated.")
    print("DOCX:", LAB3 / "lab3_report.docx")


if __name__ == "__main__":
    main()
