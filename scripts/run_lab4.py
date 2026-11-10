"""Lab 4 reproducible validator experiment; run from any directory.

Normal invocation requires MLflow. --offline-test is for environments where
MLflow cannot be installed; its explicitly marked runs are NOT MLflow runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from moons_lab.config import Lab4Config, load_config
from moons_lab.data import load_xy
from moons_lab.utils import measured_call, sha256_file
from moons_lab.validator import (
    ValidatorModel,
    calibrate_threshold,
    evaluate_validator,
    fit_chunks,
    fit_full,
    load_validator,
    make_ood_samples,
    save_validator,
)

OUT = ROOT / "reports" / "LAB4"


def positive_proba(model: object, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=np.float64)  # type: ignore[attr-defined]
    if p.ndim == 2 and p.shape == (len(x), 2):
        p = p[:, 1]
    if p.shape != (len(x),) or not np.all(np.isfinite(p)) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("working model returned invalid positive-class probabilities")
    return p


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot save empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (.git missing; use GitHub commit permalink for the downloaded archive)"


def config_negative(cfg: Lab4Config) -> list[dict[str, object]]:
    cases = {
        "incorrect_feature_order": {"features": ["x2", "x1"]},
        "unsupported_distance_method": {"distance_method": "euclidean"},
        "invalid_confidence": {"confidence_threshold": 0.25},
        "invalid_false_reject_rate": {"target_false_reject_rate": 1.2},
        "invalid_chunk_size": {"chunk_size": 0},
        "invalid_cov_regularization": {"covariance_regularization": 0},
    }
    lines = ["case,status,reason"]
    results: list[dict[str, object]] = []
    for name, change in cases.items():
        try:
            Lab4Config.model_validate({**cfg.model_dump(), **change})
        except ValidationError as exc:
            status = "REJECTED"
            reason = str(exc.errors()[0]["msg"]).replace(",", ";").replace("\n", " ")
        else:
            status, reason = "ERROR_ACCEPTED", "invalid settings passed validation"
        lines.append(f"{name},{status},{reason}")
        results.append({"case": name, "status": status, "reason": reason})
    (OUT / "config_negative_test.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if any(r["status"] != "REJECTED" for r in results):
        raise AssertionError("a negative configuration was accepted")
    return results


def negative_controls(
    validator: ValidatorModel, x_test: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
    x_ood: np.ndarray, p_ood: np.ndarray, cfg: Lab4Config,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    n = cfg.negative_examples_per_type
    half = len(x_ood) // 2
    labels = [
        ("own_unlabeled", x_test[:n], p_test[:n], None, "reference_sample"),
        ("shifted_mean", x_ood[:half], p_ood[:half], None, "injected_foreign"),
        ("outliers", x_ood[half:], p_ood[half:], None, "injected_foreign"),
        ("label_flip_with_claimed_label", x_test, p_test, 1 - y_test, "injected_foreign"),
    ]
    log: list[dict[str, object]] = []
    rates: dict[str, float] = {}
    for scenario, x, p, claimed_labels, sample_type in labels:
        decisions = validator.explain(x, p, claimed_labels)
        accepted = np.array([d["accepted"] for d in decisions], dtype=bool)
        rates[scenario + "_reject_rate"] = float((~accepted).mean())
        # All injected examples are logged, including any false acceptance;
        # no unsuccessful negative control disappears from the output.
        for i, decision in enumerate(decisions):
            expected = "reject" if sample_type == "injected_foreign" else "accept"
            actual = "accept" if accepted[i] else "reject"
            log.append({
                "scenario": scenario, "sample_index": i,
                "x1": float(x[i, 0]), "x2": float(x[i, 1]),
                "claimed_label": "" if claimed_labels is None else int(claimed_labels[i]),
                "expected": expected, "actual": actual,
                "verdict_match": bool(actual == expected),
                "reason": str(decision["reason"]),
                "distance": float(decision["distance"]),
                "distance_threshold": float(decision["distance_threshold"]),
                "confidence": float(decision["confidence"]),
                "confidence_threshold": float(decision["confidence_threshold"]),
                "working_model_p1": float(p[i]),
            })
    # Foreign categories are invalid for this tabular numeric variant and are
    # rejected at schema parsing, before the working model is even invoked.
    for i in range(n):
        raw = {"x1": f"unseen_category_{i}", "x2": 0.0}
        check = validator.validate_record(raw)
        log.append({
            "scenario": "foreign_categories", "sample_index": i,
            "x1": raw["x1"], "x2": raw["x2"], "claimed_label": "",
            "expected": "reject", "actual": "reject" if check["accepted"] is False else "ERROR_ACCEPTED",
            "verdict_match": check["accepted"] is False,
            "reason": check["reason"], "distance": "", "distance_threshold": "",
            "confidence": "", "confidence_threshold": "", "working_model_p1": "",
        })
    rates["foreign_categories_reject_rate"] = float(all(
        row["verdict_match"] for row in log if row["scenario"] == "foreign_categories"
    ))
    # Stringified numbers and booleans are invalid external feature types;
    # they must be rejected rather than coerced to a numeric value.
    for i, invalid in enumerate(("0.5", True, False, None)):
        raw = {"x1": invalid, "x2": 0.0}
        check = validator.validate_record(raw)
        log.append({
            "scenario": "wrong_feature_types", "sample_index": i,
            "x1": repr(invalid), "x2": raw["x2"], "claimed_label": "",
            "expected": "reject", "actual": "reject" if check["accepted"] is False else "ERROR_ACCEPTED",
            "verdict_match": check["accepted"] is False,
            "reason": check["reason"], "distance": "", "distance_threshold": "",
            "confidence": "", "confidence_threshold": "", "working_model_p1": "",
        })
    rates["wrong_feature_types_reject_rate"] = float(all(
        row["verdict_match"] for row in log if row["scenario"] == "wrong_feature_types"
    ))
    _csv(OUT / "validator_negative.log", log)
    if rates["foreign_categories_reject_rate"] != 1.0:
        raise AssertionError("foreign category was silently accepted")
    if rates["wrong_feature_types_reject_rate"] != 1.0:
        raise AssertionError("stringified numeric or bool feature was silently accepted")
    if rates["shifted_mean_reject_rate"] < cfg.min_ood_recall:
        raise AssertionError("mean shift is insufficiently detected")
    if rates["outliers_reject_rate"] < cfg.min_ood_recall:
        raise AssertionError("outliers are insufficiently detected")
    if rates["label_flip_with_claimed_label_reject_rate"] < cfg.min_ood_recall:
        raise AssertionError("label flips with explicitly supplied labels are insufficiently detected")
    return log, rates


def _verify_numpy_replay(model: ValidatorModel, path: Path, x: np.ndarray, p: np.ndarray) -> dict[str, object]:
    loaded = load_validator(path)
    np.testing.assert_array_equal(model.accept(x, p), loaded.accept(x, p))
    np.testing.assert_array_equal(model.distances(x), loaded.distances(x))
    copy = OUT / "__npz_replay_tmp.npz"
    try:
        save_validator(loaded, copy)
        hash_equal = sha256_file(copy) == sha256_file(path)
    finally:
        copy.unlink(missing_ok=True)
    if not hash_equal:
        raise AssertionError("NumPy serialization is not bytewise reproducible")
    return {"prediction_equal": True, "npz_sha256_equal_on_repeat": True, "sha256": sha256_file(path)}


def configure_mlflow_file_store() -> None:
    """Opt in to the course-required local mlruns/ backend on newer MLflow.

    MLflow 3.x can reject filesystem tracking stores by default. Our course
    explicitly requires mlruns/, so we enable the documented compatibility
    flag before importing or using MLflow tracking operations. We do not
    migrate, delete, or reinitialize any existing run data.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@contextmanager
def tracking_run(mode: str, offline: bool) -> Iterator[tuple[str, str, object | None]]:
    if offline:
        # Diagnostic-only deterministic ID, clearly NOT a real MLflow run.
        yield f"OFFLINE_NOT_MLFLOW_{mode}", "offline_diagnostic", None
        return
    configure_mlflow_file_store()
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is required: install with pip install -e '.[tracking]' (or use --offline-test for diagnostics)") from exc
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("lab4_validator")
    with mlflow.start_run(run_name=f"lab4_{mode}") as run:
        yield run.info.run_id, "mlflow", mlflow


def code_source_manifest() -> dict[str, object]:
    """Fingerprint real source bytes independently of the Git commit state."""
    tracked = [
        "configs/lab4.json", "src/moons_lab/validator.py", "src/moons_lab/config.py",
        "src/moons_lab/data.py", "src/moons_lab/utils.py", "scripts/run_lab4.py",
        "defense_demo/lab4_demo.py", "LAB4/RUN_LAB4.ps1",
        "LAB4/NORMALIZE_COVERAGE_LOG.ps1", "LAB4/CHECK_MLFLOW_STORE.ps1",
        "tests/test_validator.py", "tests/test_lab4_workflow.py",
        "tests/test_lab4_mlflow_tracking.py", "tests/test_lab4_runner.py",
    ]
    entries = {name: sha256_file(ROOT / name) for name in tracked}
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
    return {"files_sha256": entries, "combined_sha256": fingerprint, "git_head_at_run": git_head()}


def summarize_negatives(log: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    """Count every expected rejection, including failures, without hiding any."""
    result: dict[str, dict[str, int]] = {}
    for row in log:
        scenario = str(row["scenario"])
        entry = result.setdefault(scenario, {"total": 0, "accepted": 0, "rejected": 0, "unexpected_accept": 0})
        entry["total"] += 1
        is_accept = row["actual"] == "accept"
        entry["accepted" if is_accept else "rejected"] += 1
        if row["expected"] == "reject" and is_accept:
            entry["unexpected_accept"] += 1
    return result


def verify_repeated_runs(
    cfg: Lab4Config, paths: dict[str, Path], original: dict[str, ValidatorModel],
    primary_rows: list[dict[str, object]], x_val: np.ndarray, p_val: np.ndarray,
    x_test: np.ndarray, p_test: np.ndarray, x_ood: np.ndarray, p_ood: np.ndarray,
    original_integrity: dict[str, str], offline: bool,
) -> dict[str, object]:
    """Recompute BOTH regimes; compare saved bytes, statistics, metrics and decisions.

    Repeat runs use new MLflow IDs and never overwrite primary .npz files.
    Runtime and tracemalloc memory are not bitwise-reproducible metrics.
    """
    metrics_keys = ("rejection_precision", "rejection_recall", "false_reject_rate", "ood_accept_rate", "roc_auc")
    checks: dict[str, object] = {"seed": cfg.seed, "stats_tolerance": cfg.stats_atol, "modes": ["full", "chunks"]}
    primary_by_mode = {str(r["mode"]): r for r in primary_rows}
    with TemporaryDirectory(prefix="lab4_repeat_") as temporary:
        for mode in ("full", "chunks"):
            if mode == "full":
                secondary = fit_full(paths["train"], cfg.confidence_threshold, cfg.covariance_regularization)
            else:
                secondary = fit_chunks(ROOT / cfg.chunks_dir, cfg.confidence_threshold,
                                       cfg.covariance_regularization, cfg.chunk_size)
            calibrate_threshold(secondary, x_val, p_val, cfg.target_false_reject_rate)
            repeated_path = Path(temporary) / f"validator_{mode}_repeat.npz"
            save_validator(secondary, repeated_path)
            reference = original[mode]
            original_row = primary_by_mode[mode]
            repeated_metrics = evaluate_validator(secondary, x_test, p_test, x_ood, p_ood)
            differences = {key: abs(repeated_metrics[key] - float(original_row[key])) for key in metrics_keys}
            stats_difference = max(
                float(np.max(np.abs(reference.means - secondary.means))),
                float(np.max(np.abs(reference.inv_covariances - secondary.inv_covariances))),
                abs(reference.distance_threshold - secondary.distance_threshold),
            )
            expected_sha = str(original_row["params_sha256"])
            repeat_sha = sha256_file(repeated_path)
            decisions_equal = bool(np.array_equal(reference.accept(x_test, p_test), secondary.accept(x_test, p_test)))
            same_counts = bool(np.array_equal(reference.counts, secondary.counts))
            passed = (expected_sha == repeat_sha and decisions_equal and same_counts and
                      stats_difference <= cfg.stats_atol and max(differences.values()) <= cfg.stats_atol)
            if not passed:
                raise AssertionError(f"{mode} repeat differs: hash={repeat_sha}, stats={stats_difference}, metrics={differences}")
            # Capture a separate MLflow run for the repeat, without touching the primary run.
            with tracking_run(f"{mode}_repeat", offline) as (repeat_id, backend, tracking):
                if tracking is not None:
                    tracking.log_params({"mode": mode, "repeat_of_run_id": str(original_row["run_id"]),
                                         "seed": cfg.seed, "working_model_sha256": original_integrity["model"]})
                    tracking.log_metrics({**repeated_metrics, "repeat_stats_max_abs_diff": stats_difference,
                                          "repeat_metrics_max_abs_diff": max(differences.values())})
                    tracking.log_artifact(str(repeated_path))
                checks[mode] = {
                    "primary_run_id": str(original_row["run_id"]), "repeat_run_id": repeat_id,
                    "tracking_backend": backend, "npz_sha256_equal": expected_sha == repeat_sha,
                    "primary_sha256": expected_sha, "repeat_sha256": repeat_sha,
                    "statistics_max_abs_diff": stats_difference,
                    "metrics_max_abs_diff": max(differences.values()), "metrics_differences": differences,
                    "decisions_equal": decisions_equal, "counts_equal": same_counts,
                    "pass": passed,
                }
    source_hashes_unchanged = all(sha256_file(paths[k]) == v for k, v in original_integrity.items())
    checks["source_hashes_unchanged"] = source_hashes_unchanged
    checks["all_pass"] = bool(source_hashes_unchanged and all(checks[m]["pass"] for m in ("full", "chunks")))
    if not checks["all_pass"]:
        raise AssertionError("LAB4 repeat failed to prove reproducibility")
    return checks



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lab4 validator full/chunks experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lab4.json")
    parser.add_argument("--offline-test", action="store_true", help="diagnostics without MLflow; not submission-valid")
    args = parser.parse_args(argv)
    cfg: Lab4Config = load_config(4, args.config)  # type: ignore[assignment]
    OUT.mkdir(parents=True, exist_ok=True)
    # MLflow must exist BEFORE the experiment starts unless in explicit diagnostic mode.
    if not args.offline_test:
        configure_mlflow_file_store()
        try:
            import mlflow  # noqa: F401
        except ImportError as exc:
            raise SystemExit("MLflow is missing; install with pip install -e '.[tracking]' or use --offline-test") from exc

    config_negative(cfg)
    paths = {key: ROOT / getattr(cfg, field) for key, field in [
        ("train", "train_path"), ("validation", "validation_path"),
        ("test", "test_path"), ("model", "working_model_path")
    ]}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"required prereq missing: {p}; LAB1-2 must be completed first")
    integrity = {key: sha256_file(path) for key, path in paths.items()}
    model = joblib.load(paths["model"])
    x_val, _ = load_xy(paths["validation"])
    x_test, y_test = load_xy(paths["test"])
    p_val, p_test = positive_proba(model, x_val), positive_proba(model, x_test)
    x_ood = make_ood_samples(cfg.ood_samples, cfg.ood_shift, cfg.seed)
    p_ood = positive_proba(model, x_ood)
    measures = [
        ("full", measured_call(fit_full, paths["train"], cfg.confidence_threshold, cfg.covariance_regularization)),
        ("chunks", measured_call(fit_chunks, ROOT / cfg.chunks_dir, cfg.confidence_threshold,
                                 cfg.covariance_regularization, cfg.chunk_size)),
    ]
    full, chunks = measures[0][1].value, measures[1][1].value
    diffs = {
        "means_max_abs_diff": float(np.max(np.abs(full.means - chunks.means))),
        "cov_max_abs_diff": float(np.max(np.abs(np.linalg.inv(full.inv_covariances) - np.linalg.inv(chunks.inv_covariances)))),
        "inv_cov_max_abs_diff": float(np.max(np.abs(full.inv_covariances - chunks.inv_covariances))),
    }
    counts_equal = bool(np.array_equal(full.counts, chunks.counts))
    if not counts_equal or any(v > cfg.stats_atol for v in diffs.values()):
        raise AssertionError(f"full/chunk statistics differ: {diffs}; counts_equal={counts_equal}")

    rows: list[dict[str, object]] = []
    replay: dict[str, object] = {}
    negatives: dict[str, float] = {}
    negatives_summary: dict[str, object] = {}
    for mode, measured in measures:
        validator = measured.value
        calibrate_threshold(validator, x_val, p_val, cfg.target_false_reject_rate)
        params_path = OUT / ("validator_params.npz" if mode == "full" else "validator_params_chunk.npz")
        save_validator(validator, params_path)
        replay[mode] = _verify_numpy_replay(validator, params_path, x_test[:100], p_test[:100])
        metrics = evaluate_validator(validator, x_test, p_test, x_ood, p_ood)
        if metrics["rejection_recall"] < cfg.min_ood_recall or metrics["false_reject_rate"] > cfg.max_test_false_reject_rate:
            raise AssertionError(f"quality criteria were not met for {mode}: {metrics}")
        if mode == "full":
            negative_log, negatives = negative_controls(validator, x_test, y_test, p_test, x_ood, p_ood, cfg)
            negatives_summary = summarize_negatives(negative_log)
            _json(OUT / "negative_summary.json", negatives_summary)
        with tracking_run(mode, args.offline_test) as (run_id, backend, tracking):
            if tracking is not None:
                tracking.log_params({**cfg.model_dump(exclude={"features"}), "features": str(cfg.features),
                                     "mode": mode, "distance_threshold": validator.distance_threshold,
                                     "working_model_sha256": integrity["model"]})
                tracking.log_metrics({**metrics, "elapsed_s": measured.elapsed_s, "peak_memory_mb": measured.peak_memory_mb,
                                      "distance_threshold": validator.distance_threshold})
                tracking.log_artifact(str(params_path))
                tracking.log_artifact(str(OUT / "config_negative_test.log"))
                if mode == "full":
                    tracking.log_artifact(str(OUT / "validator_negative.log"))
            rows.append({
                "mode": mode, "train_n": int(np.sum(validator.counts)), **metrics,
                "distance_threshold": validator.distance_threshold,
                "elapsed_s": measured.elapsed_s, "peak_memory_mb": measured.peak_memory_mb,
                "params_sha256": replay[mode]["sha256"], "run_id": run_id, "tracking_backend": backend,
            })
    _csv(OUT / "validator_metrics.csv", rows)
    tolerance = cfg.stats_atol
    mkeys = ["rejection_precision", "rejection_recall", "false_reject_rate", "ood_accept_rate", "roc_auc"]
    comps: dict[str, object] = {
        **diffs, "count_full": full.counts.tolist(), "count_chunks": chunks.counts.tolist(),
        "counts_equal": counts_equal,
        "threshold_abs_diff": abs(full.distance_threshold - chunks.distance_threshold),
        "metrics_max_abs_diff": max(abs(float(rows[0][k]) - float(rows[1][k])) for k in mkeys),
        "validator_full_sha256": replay["full"]["sha256"],
        "validator_chunks_sha256": replay["chunks"]["sha256"],
    }
    comps["stats_within_tolerance"] = bool(all(d <= tolerance for d in diffs.values()) and counts_equal and comps["threshold_abs_diff"] <= tolerance)
    comps["metrics_within_tolerance"] = bool(comps["metrics_max_abs_diff"] <= tolerance)
    comps["quality_pass"] = bool(all(float(row["rejection_recall"]) >= cfg.min_ood_recall and float(row["false_reject_rate"]) <= cfg.max_test_false_reject_rate for row in rows))
    if not (comps["stats_within_tolerance"] and comps["metrics_within_tolerance"] and comps["quality_pass"]):
        raise AssertionError(f"comparison criteria violated: {comps}")
    _json(OUT / "comparison.json", comps)
    _json(OUT / "reproducibility.json", {
        "seed": cfg.seed, "replay": replay, "statistics": comps,
        "data_and_model_hashes": integrity, "non_mlflow_diagnostic": args.offline_test,
        "full_and_chunk_decisions_equal": bool(np.array_equal(full.accept(x_test, p_test), chunks.accept(x_test, p_test))),
    })
    repeat = verify_repeated_runs(cfg, paths, {"full": full, "chunks": chunks}, rows,
                                  x_val, p_val, x_test, p_test, x_ood, p_ood, integrity,
                                  args.offline_test)
    _json(OUT / "repeat_verification.json", repeat)
    manifest = code_source_manifest()
    _json(OUT / "source_manifest.json", manifest)
    if not args.offline_test:
        import mlflow
        for row in rows:
            with mlflow.start_run(run_id=str(row["run_id"])):
                for name in ("validator_metrics.csv", "comparison.json", "reproducibility.json",
                             "repeat_verification.json", "negative_summary.json", "source_manifest.json"):
                    mlflow.log_artifact(str(OUT / name))
    for row in rows:
        print(f"LAB4 {row['mode']}: recall={row['rejection_recall']:.4f} false_reject={row['false_reject_rate']:.4f} "
              f"roc_auc={row['roc_auc']:.4f} run_id={row['run_id']} backend={row['tracking_backend']}")
    print(f"LAB4: outputs {OUT}; metrics/statistics equivalent = PASS; independent repeats = PASS")
    if args.offline_test:
        print("WARNING: offline diagnostics only; rerun without --offline-test to produce real MLflow run IDs")


if __name__ == "__main__":
    main()
