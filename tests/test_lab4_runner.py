"""Lab4 runner, repeat verification, strict contract and demo integration tests.

The in-process end-to-end test uses its own tiny fixture: no writes to the
user's LAB1-3 outputs or real MLflow history. Real MLflow is checked separately.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons
from sklearn.svm import SVC

from moons_lab.config import Lab4Config
from moons_lab.validator import ValidatorModel, fit_full

ROOT = Path(__file__).resolve().parents[1]


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_script(ROOT / "scripts/run_lab4.py", "lab4_runner_integrated")


def test_strict_numeric_schema_and_finite_values():
    validator = ValidatorModel(np.array([[0., 0.], [1., 1.]]),
                               np.stack([np.eye(2), np.eye(2)]), 2.0, 0.52)
    for raw in ("0.5", True, False, None, object(), float("nan"), float("inf")):
        assert validator.validate_record({"x1": raw, "x2": 1}) == {
            "accepted": False, "reason": "non_numeric_or_non_finite_feature", "distance": None,
        }
    for raw in (0, 0.5, np.float64(0.5), np.int64(1)):
        assert validator.validate_record({"x1": raw, "x2": 0})["accepted"] is None
    assert validator.validate_record({"x1": 0})["reason"] == "invalid_feature_schema"


def test_probability_shape_contract(runner):
    class Correct:
        def predict_proba(self, x):
            return np.c_[np.full(len(x), .2), np.full(len(x), .8)]
    class Invalid:
        def predict_proba(self, x):
            return np.full((len(x), 3), .3)
    rows = np.zeros((3, 2))
    np.testing.assert_array_equal(runner.positive_proba(Correct(), rows), np.full(3, .8))
    with pytest.raises(ValueError, match="invalid positive-class"):
        runner.positive_proba(Invalid(), rows)


def test_output_helpers_and_empty_csv(tmp_path, runner):
    with pytest.raises(ValueError, match="empty CSV"):
        runner._csv(tmp_path / "empty.csv", [])
    runner._csv(tmp_path / "one.csv", [{"x": 1}, {"x": 2}])
    with (tmp_path / "one.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    runner._json(tmp_path / "a.json", {"result": True})
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {"result": True}


def test_negative_summary_does_not_hide_false_acceptances(runner):
    summary = runner.summarize_negatives([
        {"scenario": "label_flip_with_claimed_label", "expected": "reject", "actual": "accept"},
        {"scenario": "label_flip_with_claimed_label", "expected": "reject", "actual": "reject"},
        {"scenario": "own_unlabeled", "expected": "accept", "actual": "accept"},
    ])
    assert summary["label_flip_with_claimed_label"] == {
        "total": 2, "accepted": 1, "rejected": 1, "unexpected_accept": 1,
    }


def test_code_manifest_contains_real_source_bytes(runner):
    result = runner.code_source_manifest()
    assert len(result["combined_sha256"]) == 64
    assert result["files_sha256"]["src/moons_lab/validator.py"]
    assert "git_head_at_run" in result


def test_inprocess_end_to_end_repeat_and_demo(tmp_path, monkeypatch, runner, capsys):
    """Exercise the real offline pipeline without touching desktop lab outputs."""
    (tmp_path / "chunks").mkdir()
    (tmp_path / "configs").mkdir()
    x, y = make_moons(n_samples=800, noise=.12, random_state=42)
    train = pd.DataFrame({"x1": x[:600, 0], "x2": x[:600, 1], "y": y[:600]})
    holdout = pd.DataFrame({"x1": x[600:, 0], "x2": x[600:, 1], "y": y[600:]})
    train.to_csv(tmp_path / "train.csv", index=False)
    holdout.to_csv(tmp_path / "validation.csv", index=False)
    holdout.to_csv(tmp_path / "test.csv", index=False)
    for i in range(10):
        train.iloc[i * 60:(i + 1) * 60].to_csv(tmp_path / "chunks" / f"train_chunk_{i:03d}.csv", index=False)
    model = SVC(kernel="rbf", C=10, probability=True, random_state=42).fit(x[:600], y[:600])
    joblib.dump(model, tmp_path / "working.joblib")
    base = json.loads((ROOT / "configs/lab4.json").read_text(encoding="utf-8"))
    base.update({
        "train_path": "train.csv", "validation_path": "validation.csv", "test_path": "test.csv",
        "working_model_path": "working.joblib", "chunks_dir": "chunks", "chunk_size": 60,
        "ood_samples": 200, "negative_examples_per_type": 10,
    })
    Lab4Config.model_validate(base)
    (tmp_path / "configs/lab4.json").write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "reports/LAB4")
    # Code fingerprint is separately tested using actual source root.
    monkeypatch.setattr(runner, "code_source_manifest", lambda: {
        "combined_sha256": "a" * 64, "files_sha256": {}, "git_head_at_run": "unit-test"
    })
    runner.main(["--config", str(tmp_path / "configs/lab4.json"), "--offline-test"])
    out = tmp_path / "reports/LAB4"
    check = json.loads((out / "repeat_verification.json").read_text(encoding="utf-8"))
    assert check["all_pass"] is True
    assert all(check[mode]["npz_sha256_equal"] and check[mode]["decisions_equal"] for mode in ("full", "chunks"))
    assert all(check[mode]["repeat_run_id"].startswith("OFFLINE_NOT_MLFLOW_") for mode in ("full", "chunks"))
    with (out / "validator_metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert {row["mode"] for row in csv.DictReader(stream)} == {"full", "chunks"}
    negatives = json.loads((out / "negative_summary.json").read_text(encoding="utf-8"))
    assert negatives["label_flip_with_claimed_label"]["total"] == len(holdout)
    assert "independent repeats = PASS" in capsys.readouterr().out
    # Also exercise the actual demonstration script with the same isolated outputs.
    demo = load_script(ROOT / "defense_demo/lab4_demo.py", "lab4_demo_integrated")
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    demo.main()
    stdout = capsys.readouterr().out
    assert '"kind": "normal"' in stdout
    assert '"kind": "unknown_category"' in stdout
    assert '"accepted": false' in stdout


def test_repeat_fails_loudly_on_changed_sha(tmp_path, runner):
    """Independent second pass must not mask a tampered baseline artefact."""
    x, y = make_moons(n_samples=120, noise=.08, random_state=42)
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    train = tmp_path / "train.csv"
    df.to_csv(train, index=False)
    old_root = runner.ROOT
    runner.ROOT = tmp_path
    try:
        v = fit_full(train, 0.52)
        from moons_lab.validator import calibrate_threshold, evaluate_validator, make_ood_samples
        pv = np.where(y == 1, .99, .01)
        calibrate_threshold(v, x, pv, .1)
        ood = make_ood_samples(100, 4, 42)
        po = np.full(100, .99)
        metrics = evaluate_validator(v, x, pv, ood, po)
        cfg = Lab4Config.model_validate({
            **json.loads((old_root / "configs/lab4.json").read_text(encoding="utf-8")),
            "chunk_size": 60,
        })
        primary = [{"mode": "full", "run_id": "baseline-id", "params_sha256": "0" * 64, **metrics}]
        with pytest.raises(AssertionError, match="repeat differs"):
            runner.verify_repeated_runs(cfg, {"train": train}, {"full": v}, primary,
                                        x, pv, x, pv, ood, po, {"model": "dummy"}, True)
    finally:
        runner.ROOT = old_root
