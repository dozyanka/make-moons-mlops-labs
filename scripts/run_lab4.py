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

from moons_lab.config import Lab4Config, load_config
from moons_lab.data import load_xy
from moons_lab.tracking import tracked_run
from moons_lab.utils import measured_call, sha256_file
from moons_lab.validator import (
    calibrate_threshold,
    evaluate_validator,
    fit_chunks,
    fit_full,
    make_ood_samples,
    save_validator,
)


def _positive(model, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x))
    return p[:, 1] if p.ndim == 2 else p


def _config_negative(cfg: Lab4Config) -> None:
    bad = [
        {**cfg.model_dump(), "confidence_threshold": 0.2},
        {**cfg.model_dump(), "target_false_reject_rate": 1.2},
        {**cfg.model_dump(), "ood_samples": 10},
    ]
    lines = []
    for i, payload in enumerate(bad, 1):
        try:
            Lab4Config.model_validate(payload)
            lines.append(f"case_{i}: ERROR accepted")
        except ValidationError as exc:
            lines.append(f"case_{i}: REJECTED: {exc.errors()[0]['msg']}")
    (ROOT / "reports/LAB4/config_negative_test.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg: Lab4Config = load_config(4)  # type: ignore[assignment]
    model = joblib.load(ROOT / "reports/LAB2/ml_model.joblib")
    x_val, _ = load_xy(ROOT / "data/processed/validation.csv")
    x_test, _ = load_xy(ROOT / "data/processed/test.csv")
    p_val = _positive(model, x_val)
    p_test = _positive(model, x_test)
    x_ood = make_ood_samples(cfg.ood_samples, cfg.ood_shift, cfg.seed)
    p_ood = _positive(model, x_ood)

    full_measure = measured_call(fit_full, ROOT / "data/processed/train.csv", cfg.confidence_threshold)
    chunk_measure = measured_call(fit_chunks, ROOT / "data/chunks", cfg.confidence_threshold)
    full = full_measure.value
    chunk = chunk_measure.value
    calibrate_threshold(full, x_val, p_val, cfg.target_false_reject_rate)
    calibrate_threshold(chunk, x_val, p_val, cfg.target_false_reject_rate)
    save_validator(full, ROOT / "reports/LAB4/validator_params.npz")
    save_validator(chunk, ROOT / "reports/LAB4/validator_params_chunk.npz")

    rows = []
    for mode, validator, measure, artifact in [
        ("full", full, full_measure, "validator_params.npz"),
        ("chunks", chunk, chunk_measure, "validator_params_chunk.npz"),
    ]:
        metrics = evaluate_validator(validator, x_test, p_test, x_ood, p_ood)
        with tracked_run(f"lab4_{mode}", experiment="lab4_validator") as run:
            run.log_params({
                **cfg.model_dump(),
                "mode": mode,
                "distance_threshold": validator.distance_threshold,
            })
            run.log_metrics({**metrics, "elapsed_s": measure.elapsed_s, "peak_memory_mb": measure.peak_memory_mb})
            run.log_artifact(ROOT / "reports/LAB4" / artifact)
            rows.append({
                "mode": mode,
                **metrics,
                "distance_threshold": validator.distance_threshold,
                "elapsed_s": measure.elapsed_s,
                "peak_memory_mb": measure.peak_memory_mb,
                "run_id": run.run_id,
                "tracking_backend": run.backend,
            })
    pd.DataFrame(rows).to_csv(ROOT / "reports/LAB4/validator_metrics.csv", index=False)

    sample_own = x_test[:8]
    sample_ood = x_ood[:8]
    log_rows = []
    for kind, x, p in [("own", sample_own, p_test[:8]), ("foreign", sample_ood, p_ood[:8])]:
        details = full.explain(x, p)
        for idx, detail in enumerate(details):
            expected = "accept" if kind == "own" else "reject"
            actual = "accept" if detail["accepted"] else "reject"
            log_rows.append({
                "kind": kind,
                "x1": float(x[idx, 0]),
                "x2": float(x[idx, 1]),
                "expected": expected,
                "actual": actual,
                "reason": detail["reason"],
                "distance": detail["distance"],
                "distance_threshold": detail["distance_threshold"],
                "confidence": detail["confidence"],
                "confidence_threshold": detail["confidence_threshold"],
                "probability": float(p[idx]),
            })
    pd.DataFrame(log_rows).to_csv(ROOT / "reports/LAB4/validator_negative.log", index=False)

    comparison = {
        "means_max_abs_diff": float(np.max(np.abs(full.means - chunk.means))),
        "inv_cov_max_abs_diff": float(np.max(np.abs(full.inv_covariances - chunk.inv_covariances))),
        "threshold_abs_diff": abs(full.distance_threshold - chunk.distance_threshold),
        "validator_sha256": sha256_file(ROOT / "reports/LAB4/validator_params.npz"),
    }
    (ROOT / "reports/LAB4/comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _config_negative(cfg)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
