import numpy as np

from moons_lab.drift import compute_drift, ks_stat, psi


def test_psi_and_ks_detect_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=3000)
    same = rng.normal(size=3000)
    shifted = rng.normal(loc=2.0, size=3000)
    assert psi(ref, same) < psi(ref, shifted)
    assert ks_stat(ref, same) < ks_stat(ref, shifted)


def test_compute_drift():
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=(1000, 2))
    x2 = x1 + 1.0
    p1 = 1 / (1 + np.exp(-x1[:, 0]))
    p2 = 1 / (1 + np.exp(-x2[:, 0]))
    m = compute_drift(x1, x2, p1, p2)
    assert m.max_feature_psi > 0.1
