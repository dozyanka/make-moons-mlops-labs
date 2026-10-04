from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab2Config, load_config
from moons_lab.data import load_xy
from moons_lab.metrics import bootstrap_intervals, classification_metrics, prediction_max_abs_diff
from moons_lab.ml_pipeline import (
    save_model,
    scaler_leakage_diagnostic,
    train_linear_baseline,
    train_rbf_chunks,
    train_rbf_full,
    tune_gamma,
)
from moons_lab.tracking import tracked_run
from moons_lab.utils import measured_call, sha256_file
from sklearn.preprocessing import StandardScaler


def _positive(model, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x))
    return p[:, 1] if p.ndim == 2 else p


def _config_negative_tests(cfg: Lab2Config) -> None:
    cases = []
    bad_payloads = [
        {**cfg.model_dump(), "gamma_candidates": [1.0, -0.2]},
        {**cfg.model_dump(), "chunk_size": 1},
        {**cfg.model_dump(), "passes": 0},
    ]
    for idx, payload in enumerate(bad_payloads, 1):
        try:
            Lab2Config.model_validate(payload)
            cases.append(f"case_{idx}: ERROR accepted invalid config")
        except ValidationError as exc:
            first = exc.errors()[0]
            cases.append(f"case_{idx}: REJECTED: {first['loc']} -> {first['msg']}")
    (ROOT / "reports/LAB2/config_negative_test.log").write_text("\n".join(cases) + "\n", encoding="utf-8")


def main() -> None:
    cfg: Lab2Config = load_config(2)  # type: ignore[assignment]
    x_val, y_val = load_xy(ROOT / "data/processed/validation.csv")
    x_test, y_test = load_xy(ROOT / "data/processed/test.csv")

    best_gamma, tuning_rows = tune_gamma(
        ROOT / "data/processed/train.csv",
        ROOT / "data/processed/validation.csv",
        chunk_size=cfg.chunk_size,
        candidates=cfg.gamma_candidates,
        n_components=cfg.rbf_components,
        alpha=cfg.alpha,
        passes=cfg.passes,
        seed=cfg.seed,
    )
    pd.DataFrame(tuning_rows).to_csv(ROOT / "reports/LAB2/hyperparam_search.csv", index=False)

    baseline = train_linear_baseline(ROOT / "data/processed/train.csv", cfg.seed)
    baseline_val = classification_metrics(y_val, _positive(baseline, x_val))
    baseline_test = classification_metrics(y_test, _positive(baseline, x_test))
    save_model(baseline, ROOT / "reports/LAB2/linear_baseline.joblib")

    full_measure = measured_call(
        train_rbf_full,
        ROOT / "data/processed/train.csv",
        chunk_size=cfg.chunk_size,
        gamma=best_gamma,
        n_components=cfg.rbf_components,
        alpha=cfg.alpha,
        passes=cfg.passes,
        seed=cfg.seed,
    )
    chunk_measure = measured_call(
        train_rbf_chunks,
        ROOT / "data/chunks",
        gamma=best_gamma,
        n_components=cfg.rbf_components,
        alpha=cfg.alpha,
        passes=cfg.passes,
        seed=cfg.seed,
    )
    full_model = full_measure.value
    chunk_model = chunk_measure.value
    full_prob = full_model.predict_proba(x_test)
    chunk_prob = chunk_model.predict_proba(x_test)

    save_model(full_model, ROOT / "reports/LAB2/ml_model.joblib")
    save_model(chunk_model, ROOT / "reports/LAB2/ml_model_chunk.joblib")

    rows = []
    for mode, model, proba, measure in [
        ("full", full_model, full_prob, full_measure),
        ("chunks", chunk_model, chunk_prob, chunk_measure),
    ]:
        metrics = classification_metrics(y_test, proba)
        cis = bootstrap_intervals(y_test, proba, rounds=cfg.bootstrap_rounds, seed=cfg.seed)
        with tracked_run(f"lab2_{mode}", experiment="lab2_ml") as run:
            run.log_params({
                **cfg.model_dump(),
                "mode": mode,
                "best_gamma": best_gamma,
                "test_used_for_tuning": False,
            })
            run.log_metrics({**metrics, "elapsed_s": measure.elapsed_s, "peak_memory_mb": measure.peak_memory_mb})
            artifact_path = ROOT / "reports/LAB2" / ("ml_model.joblib" if mode == "full" else "ml_model_chunk.joblib")
            run.log_artifact(artifact_path)
            rows.append(
                {
                    "mode": mode,
                    **metrics,
                    "accuracy_ci_low": cis["accuracy"][0],
                    "accuracy_ci_high": cis["accuracy"][1],
                    "f1_ci_low": cis["f1"][0],
                    "f1_ci_high": cis["f1"][1],
                    "roc_auc_ci_low": cis["roc_auc"][0],
                    "roc_auc_ci_high": cis["roc_auc"][1],
                    "peak_memory_mb": measure.peak_memory_mb,
                    "elapsed_s": measure.elapsed_s,
                    "run_id": run.run_id,
                    "tracking_backend": run.backend,
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "reports/LAB2/ml_metrics.csv", index=False)

    x_train, _ = load_xy(ROOT / "data/processed/train.csv")
    good_leak_check = scaler_leakage_diagnostic(full_model.scaler, x_train, x_test)
    intentionally_leaky_scaler = StandardScaler().fit(np.vstack([x_train, x_test]))
    bad_leak_check = scaler_leakage_diagnostic(intentionally_leaky_scaler, x_train, x_test)
    leak_control = {
        "production_scaler": good_leak_check,
        "intentionally_leaky_scaler": bad_leak_check,
        "pass": (not bool(good_leak_check["leak_suspected"])) and bool(bad_leak_check["leak_suspected"]),
    }
    (ROOT / "reports/LAB2/leakage_negative_test.log").write_text(json.dumps(leak_control, indent=2), encoding="utf-8")

    comparison = {
        "best_gamma": best_gamma,
        "linear_baseline_validation": baseline_val,
        "linear_baseline_test": baseline_test,
        "full_test": classification_metrics(y_test, full_prob),
        "chunk_test": classification_metrics(y_test, chunk_prob),
        "prediction_max_abs_diff": prediction_max_abs_diff(full_prob, chunk_prob),
        "model_sha256": sha256_file(ROOT / "reports/LAB2/ml_model.joblib"),
        "chunk_model_sha256": sha256_file(ROOT / "reports/LAB2/ml_model_chunk.joblib"),
        "chunks": len(list((ROOT / "data/chunks").glob("train_chunk_*.csv"))),
    }
    (ROOT / "reports/LAB2/comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _config_negative_tests(cfg)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
