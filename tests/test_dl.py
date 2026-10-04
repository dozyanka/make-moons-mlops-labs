import numpy as np
import torch

from moons_lab.data import load_xy
from moons_lab.dl_pipeline import device_parity, load_checkpoint, save_checkpoint, train_full, train_stream


def _kwargs():
    return dict(
        validation_path="data/processed/validation.csv",
        batch_size=512,
        seed=7,
        epochs=2,
        learning_rate=0.01,
        weight_decay=1e-4,
        patience=2,
        device="cpu",
    )


def test_dl_full_save_load(tmp_path):
    result = train_full("data/processed/train.csv", **_kwargs())
    x, _ = load_xy("data/processed/test.csv")
    p = result.predict_proba(x[:100])
    assert p.shape == (100,)
    assert ((p >= 0) & (p <= 1)).all()
    path = tmp_path / "m.pt"
    save_checkpoint(result, path)
    loaded = load_checkpoint(path)
    np.testing.assert_allclose(loaded.predict_proba(x[:100]), p)
    parity = device_parity(loaded, x[:10])
    assert parity["status"] in {"checked", "not_available"}


def test_dl_stream_runs():
    result = train_stream("data/chunks", **_kwargs())
    x, _ = load_xy("data/processed/test.csv")
    assert len(result.predict_proba(x[:20])) == 20
