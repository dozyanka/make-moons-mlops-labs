import numpy as np
from sklearn.preprocessing import StandardScaler

from moons_lab.data import load_xy
from moons_lab.metrics import classification_metrics
from moons_lab.ml_pipeline import (
    load_model,
    save_model,
    scaler_leakage_diagnostic,
    train_linear_baseline,
    train_rbf_chunks,
    train_rbf_full,
    tune_gamma,
)


def test_full_and_chunk_models_are_equivalent(tmp_path):
    full = train_rbf_full(
        "data/processed/train.csv",
        chunk_size=700,
        gamma=1.4,
        n_components=48,
        alpha=1e-4,
        passes=1,
        seed=42,
    )
    chunks = train_rbf_chunks(
        "data/chunks",
        gamma=1.4,
        n_components=48,
        alpha=1e-4,
        passes=1,
        seed=42,
    )
    x, y = load_xy("data/processed/test.csv")
    pf = full.predict_proba(x[:500])
    pc = chunks.predict_proba(x[:500])
    assert np.max(np.abs(pf - pc)) < 1e-12
    assert classification_metrics(y[:500], pf)["accuracy"] > 0.90
    path = tmp_path / "model.joblib"
    save_model(full, path)
    loaded = load_model(path)
    np.testing.assert_allclose(loaded.predict_proba(x[:50]), pf[:50])


def test_tune_and_linear_baseline():
    best, rows = tune_gamma(
        "data/processed/train.csv",
        "data/processed/validation.csv",
        chunk_size=700,
        candidates=[0.7, 1.4],
        n_components=32,
        alpha=1e-4,
        passes=1,
        seed=2,
    )
    assert best in {0.7, 1.4}
    assert len(rows) == 2
    baseline = train_linear_baseline("data/processed/train.csv", 2)
    x, _ = load_xy("data/processed/test.csv")
    assert baseline.predict_proba(x[:2]).shape == (2, 2)


def test_leakage_negative_control_detects_forbidden_fit():
    x_train, _ = load_xy("data/processed/train.csv")
    x_test, _ = load_xy("data/processed/test.csv")
    good = StandardScaler().fit(x_train)
    bad = StandardScaler().fit(np.vstack([x_train, x_test]))
    assert scaler_leakage_diagnostic(good, x_train, x_test)["leak_suspected"] is False
    assert scaler_leakage_diagnostic(bad, x_train, x_test)["leak_suspected"] is True
