from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        lo = min(reference.min(), current.min())
        hi = max(reference.max(), current.max())
        edges = np.linspace(lo, hi + 1e-9, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_p = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_p = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_stat(reference: np.ndarray, current: np.ndarray) -> float:
    return float(ks_2samp(reference, current, method="auto").statistic)


@dataclass
class DriftMetrics:
    psi_x1: float
    psi_x2: float
    ks_x1: float
    ks_x2: float
    csi_prediction: float

    @property
    def max_feature_psi(self) -> float:
        return max(self.psi_x1, self.psi_x2)


def compute_drift(reference_x: np.ndarray, current_x: np.ndarray, ref_pred: np.ndarray, cur_pred: np.ndarray) -> DriftMetrics:
    return DriftMetrics(
        psi_x1=psi(reference_x[:, 0], current_x[:, 0]),
        psi_x2=psi(reference_x[:, 1], current_x[:, 1]),
        ks_x1=ks_stat(reference_x[:, 0], current_x[:, 0]),
        ks_x2=ks_stat(reference_x[:, 1], current_x[:, 1]),
        csi_prediction=psi(ref_pred, cur_pred),
    )
