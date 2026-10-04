from __future__ import annotations

import argparse

import joblib
import numpy as np

from common import (
    MODELS,
    RESULTS,
    ROOT,
    binary_metrics,
    find_csv,
    load_xy,
    print_compare,
    read_metric_row,
    save_result,
)


def probability(model, x):
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)

        # Some project wrappers return P(class=1) directly as a 1-D array,
        # while standard sklearn classifiers return an (n, 2) matrix.
        if proba.ndim == 1:
            return proba

        if proba.ndim == 2:
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]

        raise ValueError(
            f"Unexpected predict_proba shape: {proba.shape}"
        )

    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(x), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))

    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "chunks"],
        default="full",
    )
    args = parser.parse_args()

    model_name = (
        "lab2_full.joblib"
        if args.mode == "full"
        else "lab2_chunks.joblib"
    )

    model = joblib.load(MODELS / model_name)
    test_path = find_csv("test")
    x_test, y_test = load_xy(test_path)

    pred = np.asarray(model.predict(x_test)).astype(int)
    prob = probability(model, x_test)
    actual = binary_metrics(y_test, pred, prob)

    row = read_metric_row(
        ROOT / "reports" / "LAB2" / "ml_metrics.csv",
        args.mode,
    )
    expected = {
        "accuracy": float(row["accuracy"]),
        "f1": float(row["f1"]),
        "roc_auc": float(row["roc_auc"]),
    }

    passed = print_compare(
        f"LAB 2 LIVE MODEL — {args.mode}",
        actual,
        expected,
        tolerance=1e-12,
    )
    print("test data:", test_path.relative_to(ROOT))
    print("objects:", len(y_test))

    save_result(
        f"lab2_{args.mode}_result.json",
        {
            "lab": 2,
            "mode": args.mode,
            "model_file": model_name,
            "test_data": str(test_path.relative_to(ROOT)),
            "actual": actual,
            "report": expected,
            "pass": passed,
        },
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

