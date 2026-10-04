from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "defense_demo"
MODELS = DEMO / "models"
RESULTS = DEMO / "results"


def find_csv(split: str) -> Path:
    preferred = [
        ROOT / "data" / "processed" / f"{split}.csv",
        ROOT / "data" / "splits" / f"{split}.csv",
        ROOT / "data" / f"{split}.csv",
    ]

    for path in preferred:
        if path.exists():
            frame = pd.read_csv(path, nrows=3)
            if {"x1", "x2", "y"}.issubset(frame.columns):
                return path

    candidates = []
    for path in (ROOT / "data").rglob("*.csv"):
        if path.name.lower() != f"{split}.csv":
            continue
        try:
            frame = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        if {"x1", "x2", "y"}.issubset(frame.columns):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"Could not find {split}.csv with x1/x2/y")

    return sorted(candidates, key=lambda p: len(str(p)))[0]


def load_xy(path: Path):
    frame = pd.read_csv(path)
    return frame[["x1", "x2"]].to_numpy(float), frame["y"].to_numpy(int)


def binary_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def read_metric_row(path: Path, mode: str):
    frame = pd.read_csv(path)
    row = frame.loc[frame["mode"] == mode]
    if row.empty:
        raise KeyError(f"Mode {mode!r} was not found in {path}")
    return row.iloc[0]


def print_compare(title: str, actual: dict, expected: dict, tolerance=1e-9):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    passed = True

    for key in ["accuracy", "f1", "roc_auc"]:
        a = float(actual[key])
        e = float(expected[key])
        diff = abs(a - e)
        ok = diff <= tolerance
        passed = passed and ok
        print(
            f"{key:10s} actual={a:.12f} "
            f"report={e:.12f} diff={diff:.3e} "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


def save_result(name: str, payload: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
