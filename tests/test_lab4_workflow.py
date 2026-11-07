"""Contract/negative tests for LAB4. No GitHub access or model retraining."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from moons_lab.config import Lab4Config
from moons_lab.data import load_xy
from moons_lab.validator import (
    _class_stats_online,
    calibrate_threshold,
    evaluate_validator,
    fit_chunks,
    fit_full,
    load_validator,
    make_ood_samples,
    save_validator,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Lab4Config.model_validate(json.loads((ROOT / "configs/lab4.json").read_text(encoding="utf-8")))


def _positive(model, x):
    p = np.asarray(model.predict_proba(x))
    return p[:, 1] if p.ndim == 2 else p


@pytest.fixture(scope="module")
def fitted():
    f = fit_full(ROOT / CONFIG.train_path, CONFIG.confidence_threshold, CONFIG.covariance_regularization)
    c = fit_chunks(ROOT / CONFIG.chunks_dir, CONFIG.confidence_threshold, CONFIG.covariance_regularization, CONFIG.chunk_size)
    model = joblib.load(ROOT / CONFIG.working_model_path)
    xv, _ = load_xy(ROOT / CONFIG.validation_path)
    xt, yt = load_xy(ROOT / CONFIG.test_path)
    pv, pt = _positive(model, xv), _positive(model, xt)
    calibrate_threshold(f, xv, pv, CONFIG.target_false_reject_rate)
    calibrate_threshold(c, xv, pv, CONFIG.target_false_reject_rate)
    return f, c, model, xv, xt, yt, pv, pt


@pytest.mark.parametrize("bad", [
    {"features": ["x2", "x1"]}, {"features": ["x1", "x1"]},
    {"distance_method": "none"}, {"covariance_regularization": 0},
    {"confidence_threshold": 0.1}, {"target_false_reject_rate": .9},
    {"chunk_size": 0}, {"min_ood_recall": 1.01},
    {"max_test_false_reject_rate": .001},
])
def test_pydantic_rejects_invalid_configs(bad):
    with pytest.raises(ValidationError):
        Lab4Config.model_validate({**CONFIG.model_dump(), **bad})


def test_full_and_chunk_equivalence(fitted):
    full, chunks, _, _, xt, _, _, pt = fitted
    np.testing.assert_array_equal(full.counts, chunks.counts)
    np.testing.assert_allclose(full.means, chunks.means, atol=CONFIG.stats_atol, rtol=0)
    np.testing.assert_allclose(full.inv_covariances, chunks.inv_covariances, atol=CONFIG.stats_atol, rtol=0)
    np.testing.assert_allclose(full.distance_threshold, chunks.distance_threshold, atol=CONFIG.stats_atol, rtol=0)
    np.testing.assert_array_equal(full.accept(xt, pt), chunks.accept(xt, pt))


def test_calibration_uses_both_gates_and_reject_budget(fitted):
    full, _, _, xv, _, _, pv, _ = fitted
    assert (~full.accept(xv, pv)).mean() <= CONFIG.target_false_reject_rate + 1e-12
    bad = fit_full(ROOT / CONFIG.train_path, .99)
    with pytest.raises(ValueError, match="confidence_threshold alone"):
        calibrate_threshold(bad, xv, pv, CONFIG.target_false_reject_rate)


def test_numeric_ood_and_false_reject_thresholds(fitted):
    full, _, model, _, xt, _, _, pt = fitted
    xood = make_ood_samples(CONFIG.ood_samples, CONFIG.ood_shift, CONFIG.seed)
    po = _positive(model, xood)
    metrics = evaluate_validator(full, xt, pt, xood, po)
    assert metrics["rejection_recall"] >= CONFIG.min_ood_recall
    assert metrics["false_reject_rate"] <= CONFIG.max_test_false_reject_rate
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert (~full.accept(xood[:len(xood)//2], po[:len(xood)//2])).mean() >= CONFIG.min_ood_recall
    assert (~full.accept(xood[len(xood)//2:], po[len(xood)//2:])).mean() >= CONFIG.min_ood_recall


def test_non_numeric_category_and_invalid_shapes_rejected(fitted):
    full = fitted[0]
    assert full.validate_record({"x1": "unknown", "x2": .5})["accepted"] is False
    assert full.validate_record({"x1": float("nan"), "x2": .5})["accepted"] is False
    assert full.validate_record({"x1": 1., "x2": .5, "extra": 0})["accepted"] is False
    assert full.validate_record({"x1": 1., "x2": .5})["accepted"] is None
    with pytest.raises(ValueError, match="shape"):
        full.distances(np.ones((5, 3)))
    with pytest.raises(ValueError, match="finite"):
        full.accept(np.array([[np.inf, 0.]]), np.array([.9]))
    with pytest.raises(ValueError, match="probabilities"):
        full.accept(np.ones((2, 2)), np.array([1.1, .1]))


def test_claimed_label_flip_detected_only_with_claim(fitted):
    full, _, _, _, xt, yt, _, pt = fitted
    # Flipping y cannot change feature-only decision: only explicit claimed labels can.
    bare = full.accept(xt, pt)
    mislabeled = full.accept(xt, pt, 1 - yt)
    assert np.mean(~mislabeled) >= CONFIG.min_ood_recall
    assert np.mean(~mislabeled) > np.mean(~bare)
    example = full.explain(xt, pt, 1 - yt)
    assert any("claimed_label_disagrees" in r["reason"] for r in example)
    with pytest.raises(ValueError, match="claimed_labels"):
        full.accept(xt[:2], pt[:2], np.array([1, 4]))


def test_npz_reload_byte_replay(fitted, tmp_path):
    full, _, _, _, xt, _, _, pt = fitted
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    save_validator(full, a)
    saved = load_validator(a)
    save_validator(saved, b)
    assert a.read_bytes() == b.read_bytes()
    np.testing.assert_array_equal(full.accept(xt, pt), saved.accept(xt, pt))


def test_bad_stream_fails_loudly():
    with pytest.raises(ValueError, match="at least two"):
        _class_stats_online([(np.array([[1., 2.]]), np.array([0]))])
    with pytest.raises(ValueError, match="binary"):
        _class_stats_online([(np.array([[1., 2.]]), np.array([7]))])
    with pytest.raises(ValueError, match="exceeding"):
        fit_chunks(ROOT / CONFIG.chunks_dir, .52, chunk_size=300)


def test_perturbed_chunk_statistics_detected(fitted):
    full = fitted[0]
    x, y = load_xy(ROOT / CONFIG.train_path)
    x[:100, 0] += 40.0
    tmp = _class_stats_online([(x[i:i+400], y[i:i+400]) for i in range(0, len(x), 400)])
    assert max(np.abs(full.means[0] - tmp[0].mean)) > CONFIG.stats_atol


def test_seed_reproducibility():
    a = make_ood_samples(200, 3.5, 42)
    b = make_ood_samples(200, 3.5, 42)
    c = make_ood_samples(200, 3.5, 43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_required_generated_outputs_when_present():
    out = ROOT / "reports/LAB4"
    if not (out / "validator_metrics.csv").exists():
        pytest.skip("Lab4 experiment not executed; run scripts/run_lab4.py first")
    import pandas as pd
    rows = pd.read_csv(out / "validator_metrics.csv")
    assert set(rows["mode"]) == {"full", "chunks"}
    assert all(rows["rejection_recall"] >= CONFIG.min_ood_recall)
    assert all(rows["false_reject_rate"] <= CONFIG.max_test_false_reject_rate)
    assert (out / "validator_params.npz").is_file()
    assert (out / "validator_negative.log").is_file()
    assert (out / "lab4_report.md").is_file()


def test_validator_rejects_tampered_npz(tmp_path, fitted):
    import hashlib
    path = tmp_path / "model.npz"
    save_validator(fitted[0], path)
    good_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_validator(path, expected_sha256=good_hash).means.shape == (2, 2)
    payload = bytearray(path.read_bytes())
    payload[50] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_validator(path, expected_sha256=good_hash)


def test_online_stream_memory_bounded_for_more_batches():
    from moons_lab.utils import measured_call
    x = np.tile(np.array([[0., 1.], [1., 0.]]), (50, 1))
    y = np.tile(np.array([0, 1]), 50)
    def batches(repetitions):
        for _ in range(repetitions):
            yield x, y
    short = measured_call(_class_stats_online, batches(5))
    long = measured_call(_class_stats_online, batches(40))
    assert sum(s.count for s in short.value) == 500
    assert sum(s.count for s in long.value) == 4000
    assert long.peak_memory_mb <= short.peak_memory_mb * 1.5 + .05
