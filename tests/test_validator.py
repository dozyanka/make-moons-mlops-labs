import joblib
import numpy as np

from moons_lab.data import load_xy
from moons_lab.validator import (
    calibrate_threshold,
    evaluate_validator,
    fit_chunks,
    fit_full,
    load_validator,
    make_ood_samples,
    save_validator,
)


def _positive(model, x):
    p = np.asarray(model.predict_proba(x))
    return p[:, 1] if p.ndim == 2 else p


def test_validator_full_chunk_and_serialization(tmp_path):
    full = fit_full("data/processed/train.csv", 0.52)
    chunk = fit_chunks("data/chunks", 0.52)
    np.testing.assert_allclose(full.means, chunk.means, atol=1e-10)
    model = joblib.load("reports/LAB2/ml_model.joblib")
    xv, _ = load_xy("data/processed/validation.csv")
    calibrate_threshold(full, xv, _positive(model, xv), 0.03)
    xt, _ = load_xy("data/processed/test.csv")
    ood = make_ood_samples(500, 3.5, 4)
    metrics = evaluate_validator(full, xt[:500], _positive(model, xt[:500]), ood, _positive(model, ood))
    assert metrics["rejection_recall"] > 0.8
    details = full.explain(ood[:20], _positive(model, ood[:20]))
    assert len(details) == 20
    assert all("reason" in item and "distance_threshold" in item for item in details)
    assert any(not item["accepted"] for item in details)
    path = tmp_path / "v.npz"
    save_validator(full, path)
    loaded = load_validator(path)
    np.testing.assert_allclose(loaded.distances(xt[:20]), full.distances(xt[:20]))
