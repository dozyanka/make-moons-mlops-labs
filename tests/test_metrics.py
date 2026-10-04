import numpy as np

from moons_lab.metrics import bootstrap_intervals, classification_metrics, prediction_max_abs_diff


def test_classification_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
    metrics = classification_metrics(y, p)
    assert metrics["accuracy"] == 1.0
    ci = bootstrap_intervals(y, p, rounds=60, seed=1)
    assert ci["accuracy"][0] <= metrics["accuracy"] <= ci["accuracy"][1]
    assert prediction_max_abs_diff(p, p.copy()) == 0.0
