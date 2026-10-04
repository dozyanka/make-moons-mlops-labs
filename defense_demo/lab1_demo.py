from __future__ import annotations

import joblib
import pandas as pd

from common import MODELS, RESULTS, ROOT


def main():
    raw = pd.read_csv(ROOT / "data" / "raw" / "moons.csv")
    model = joblib.load(MODELS / "lab1_logistic.joblib")

    x = raw[["x1", "x2"]].to_numpy(float)
    y = raw["y"].to_numpy(int)
    pred = model.predict(x)

    accuracy = float((pred == y).mean())

    stats = pd.read_csv(ROOT / "reports" / "LAB1" / "eda_stats.csv")
    expected = float(
        stats.loc[
            stats["metric"] == "linear_train_accuracy",
            "value",
        ].iloc[0]
    )
    diff = abs(accuracy - expected)
    passed = diff <= 1e-12

    print("=" * 72)
    print("LAB 1 LIVE MODEL — LogisticRegression diagnostic baseline")
    print("=" * 72)
    print("data: data/raw/moons.csv")
    print("objects:", len(raw))
    print(f"actual accuracy : {accuracy:.12f}")
    print(f"report accuracy : {expected:.12f}")
    print(f"difference      : {diff:.3e}")
    print("RESULT:", "PASS" if passed else "FAIL")

    payload = {
        "lab": 1,
        "model": "LogisticRegression",
        "actual_accuracy": accuracy,
        "report_accuracy": expected,
        "difference": diff,
        "pass": passed,
    }
    (RESULTS / "lab1_result.json").write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
