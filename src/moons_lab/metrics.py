from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (np.asarray(proba) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    proba: np.ndarray,
    rounds: int = 250,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    values = {"accuracy": [], "f1": [], "roc_auc": []}
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ps = proba[idx]
        if len(np.unique(ys)) < 2:
            continue
        scores = classification_metrics(ys, ps)
        for key, value in scores.items():
            values[key].append(value)
    lo = 100 * alpha / 2
    hi = 100 * (1 - alpha / 2)
    return {
        key: (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))
        for key, vals in values.items()
    }


def prediction_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
