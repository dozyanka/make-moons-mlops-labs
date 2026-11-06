"""Lab 4: interpretable tabular applicability validator (NumPy statistics).

The validator is NOT a second classifier. Class-conditional Mahalanobis
statistics estimate support of the training distribution; a separate
confidence gate tests the reliability of the already-trained ML model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from numbers import Real
from typing import Iterable, Mapping

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .data import iter_chunk_xy, load_xy


@dataclass
class ClassStats:
    count: int
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class ValidatorModel:
    means: np.ndarray
    inv_covariances: np.ndarray
    distance_threshold: float
    confidence_threshold: float
    counts: np.ndarray | None = None
    covariance_regularization: float = 1e-6
    features: tuple[str, str] = ("x1", "x2")

    def __post_init__(self) -> None:
        if self.means.shape != (2, 2) or self.inv_covariances.shape != (2, 2, 2):
            raise ValueError("validator expects exactly two classes and two numeric features")
        if not np.all(np.isfinite(self.means)) or not np.all(np.isfinite(self.inv_covariances)):
            raise ValueError("non-finite validator statistics")
        if not (0.5 <= self.confidence_threshold < 1):
            raise ValueError("confidence_threshold must be in [0.5, 1)")
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")

    @staticmethod
    def _check_features(x: np.ndarray) -> np.ndarray:
        try:
            arr = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("x1 and x2 must be numeric") from exc
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("X must have shape (N, 2), ordered (x1, x2)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x1 and x2 must be finite")
        return arr

    @staticmethod
    def _check_probabilities(proba: np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(proba, dtype=np.float64)
        if arr.ndim != 1 or len(arr) != n or not np.all(np.isfinite(arr)):
            raise ValueError("model probabilities must be a finite 1-D array matching X")
        if np.any((arr < 0) | (arr > 1)):
            raise ValueError("probabilities must be in [0, 1]")
        return arr

    def distances(self, x: np.ndarray) -> np.ndarray:
        rows = self._check_features(x)
        squared = []
        for mean, inv_cov in zip(self.means, self.inv_covariances):
            delta = rows - mean
            squared.append(np.einsum("ni,ij,nj->n", delta, inv_cov, delta))
        return np.sqrt(np.maximum(np.min(np.vstack(squared), axis=0), 0.0))

    def reject_reasons(
        self, x: np.ndarray, model_proba: np.ndarray, claimed_labels: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._check_features(x)
        p = self._check_probabilities(model_proba, len(rows))
        distance = self.distances(rows)
        confidence = np.maximum(p, 1 - p)
        far = distance > self.distance_threshold
        uncertain = confidence < self.confidence_threshold
        labels_inconsistent = np.zeros(len(rows), dtype=bool)
        if claimed_labels is not None:
            y = np.asarray(claimed_labels)
            if y.ndim != 1 or len(y) != len(rows) or not np.all(np.isin(y, [0, 1])):
                raise ValueError("claimed_labels must be a 1-D binary array matching X")
            labels_inconsistent = y != (p >= 0.5).astype(int)
        return far, uncertain, labels_inconsistent

    def accept(
        self, x: np.ndarray, model_proba: np.ndarray, claimed_labels: np.ndarray | None = None
    ) -> np.ndarray:
        far, uncertain, inconsistent = self.reject_reasons(x, model_proba, claimed_labels)
        return ~(far | uncertain | inconsistent)

    def ood_score(
        self, x: np.ndarray, model_proba: np.ndarray, claimed_labels: np.ndarray | None = None
    ) -> np.ndarray:
        far, uncertain, inconsistent = self.reject_reasons(x, model_proba, claimed_labels)
        rows = self._check_features(x)
        p = self._check_probabilities(model_proba, len(rows))
        conf = np.maximum(p, 1 - p)
        # Continuous score for ROC-AUC; gates add a graded confidence penalty.
        score = self.distances(rows) / max(self.distance_threshold, 1e-12)
        score = score + 4.0 * np.maximum(self.confidence_threshold - conf, 0.0)
        if claimed_labels is not None:
            score = score + 2.0 * inconsistent.astype(float)
        return score

    def explain(
        self, x: np.ndarray, model_proba: np.ndarray, claimed_labels: np.ndarray | None = None
    ) -> list[dict[str, float | bool | str]]:
        rows = self._check_features(x)
        p = self._check_probabilities(model_proba, len(rows))
        distance = self.distances(rows)
        confidence = np.maximum(p, 1 - p)
        far, uncertain, inconsistent = self.reject_reasons(rows, p, claimed_labels)
        result: list[dict[str, float | bool | str]] = []
        for i in range(len(rows)):
            causes = []
            if far[i]:
                causes.append("too_far_from_training_distribution")
            if uncertain[i]:
                causes.append("low_model_confidence")
            if inconsistent[i]:
                causes.append("claimed_label_disagrees_with_model")
            result.append({
                "accepted": not bool(causes),
                "reason": "+".join(causes) if causes else "accepted_by_validator_policy",
                "distance": float(distance[i]),
                "distance_threshold": float(self.distance_threshold),
                "confidence": float(confidence[i]),
                "confidence_threshold": float(self.confidence_threshold),
            })
        return result

    def validate_record(self, record: Mapping[str, object]) -> dict[str, object]:
        """Schema-level rejection for non-numeric foreign categories (no silent coercion)."""
        if set(record) != set(self.features):
            return {"accepted": False, "reason": "invalid_feature_schema", "distance": None}
        # External API data must be real numeric values, not silently cast
        # strings such as "0.5" or booleans into floating-point features.
        values = [record[f] for f in self.features]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (Real, np.integer, np.floating)) for v in values):
            return {"accepted": False, "reason": "non_numeric_or_non_finite_feature", "distance": None}
        try:
            x = self._check_features(np.asarray([values], dtype=np.float64))
        except (ValueError, TypeError, OverflowError):
            return {"accepted": False, "reason": "non_numeric_or_non_finite_feature", "distance": None}
        # A valid numeric record requires a working-model probability, so cannot
        # be silently accepted by the schema validator alone.
        return {"accepted": None, "reason": "requires_working_model_probability", "distance": float(self.distances(x)[0])}


def _class_stats_full(x: np.ndarray, y: np.ndarray) -> list[ClassStats]:
    x = ValidatorModel._check_features(x)
    y = np.asarray(y)
    if y.shape != (len(x),) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("train labels must be binary and match feature rows")
    stats = []
    for label in (0, 1):
        part = x[y == label]
        if len(part) < 2:
            raise ValueError(f"not enough training samples of class {label}")
        stats.append(ClassStats(len(part), part.mean(axis=0), np.cov(part, rowvar=False, bias=True)))
    return stats


def _class_stats_online(batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[ClassStats]:
    counts = [0, 0]
    means = [np.zeros(2), np.zeros(2)]
    m2 = [np.zeros((2, 2)), np.zeros((2, 2))]
    seen = False
    for x, y in batches:
        seen = True
        x = ValidatorModel._check_features(x)
        if np.shape(y) != (len(x),) or not np.all(np.isin(y, [0, 1])):
            raise ValueError("batch labels must be binary and match feature rows")
        # Online Welford: no materialized train; memory independent of # of batches.
        for row, raw_label in zip(x, y):
            label = int(raw_label)
            counts[label] += 1
            delta = row - means[label]
            means[label] += delta / counts[label]
            m2[label] += np.outer(delta, row - means[label])
    if not seen or min(counts) < 2:
        raise ValueError("stream must contain at least two rows from each class")
    return [ClassStats(counts[i], means[i], m2[i] / counts[i]) for i in (0, 1)]


def _build(
    stats: list[ClassStats], distance_threshold: float, confidence_threshold: float,
    covariance_regularization: float = 1e-6,
) -> ValidatorModel:
    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be > 0")
    means = np.stack([s.mean for s in stats])
    inv_cov = np.stack([
        np.linalg.inv(s.covariance + np.eye(2) * covariance_regularization) for s in stats
    ])
    return ValidatorModel(
        means, inv_cov, distance_threshold, confidence_threshold,
        counts=np.asarray([s.count for s in stats], dtype=np.int64),
        covariance_regularization=covariance_regularization,
    )


def fit_full(
    train_path: str | Path, confidence_threshold: float, covariance_regularization: float = 1e-6
) -> ValidatorModel:
    x, y = load_xy(train_path)
    return _build(_class_stats_full(x, y), float("inf"), confidence_threshold, covariance_regularization)


def fit_chunks(
    chunks_dir: str | Path, confidence_threshold: float, covariance_regularization: float = 1e-6,
    chunk_size: int | None = None,
) -> ValidatorModel:
    def batches() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for x, y in iter_chunk_xy(chunks_dir):
            if chunk_size is not None and len(x) > chunk_size:
                raise ValueError(f"chunk contains {len(x)} rows, exceeding chunk_size={chunk_size}")
            yield x, y
    return _build(_class_stats_online(batches()), float("inf"), confidence_threshold, covariance_regularization)


def calibrate_threshold(
    validator: ValidatorModel, x_validation: np.ndarray, proba_validation: np.ndarray,
    target_false_reject_rate: float,
) -> float:
    """Choose a distance gate using validation only, after confidence rejections.

    Counting confidence rejections first prevents the common incorrect approach
    of calibrating distance and confidence as though their false rejects were
    disjoint. ``higher`` quantile avoids exceeding the requested empirical rate.
    """
    if not (0 < target_false_reject_rate < 0.5):
        raise ValueError("target_false_reject_rate must be in (0, .5)")
    p = validator._check_probabilities(proba_validation, len(x_validation))
    confidence_ok = np.maximum(p, 1 - p) >= validator.confidence_threshold
    n = len(p)
    if n == 0:
        raise ValueError("validation split is empty")
    conf_rejects = n - int(confidence_ok.sum())
    allowed = int(np.floor(target_false_reject_rate * n + 1e-12))
    if conf_rejects > allowed:
        raise ValueError(
            f"confidence_threshold alone rejects {conf_rejects}/{n}, exceeding target {allowed}/{n}"
        )
    distances = np.sort(validator.distances(x_validation)[confidence_ok])
    allowed_dist_rejects = allowed - conf_rejects
    keep = len(distances) - allowed_dist_rejects
    if keep < 1:
        raise ValueError("cannot calibrate distance gate on the available validation data")
    validator.distance_threshold = float(distances[keep - 1])
    if float((~validator.accept(x_validation, p)).mean()) > target_false_reject_rate + 1e-12:
        raise AssertionError("calibration failed to enforce false-reject budget")
    return validator.distance_threshold


def make_ood_samples(n: int, shift: float, seed: int) -> np.ndarray:
    """Balanced numerical OOD: shifted mean plus independently drawn outliers."""
    if n < 2 or shift <= 0:
        raise ValueError("n >= 2 and shift > 0 required")
    rng = np.random.default_rng(seed)
    half = n // 2
    shifted = rng.normal(
        loc=[shift, shift * .45], scale=[.60, .45], size=(half, 2)
    )
    # Both coordinates well outside the main make_moons training support.
    magnitudes = rng.uniform(4.5, 7.0, size=(n - half, 2))
    signs = rng.choice([-1.0, 1.0], size=(n - half, 2))
    return np.vstack([shifted, magnitudes * signs])


def evaluate_validator(
    validator: ValidatorModel, x_in: np.ndarray, proba_in: np.ndarray,
    x_ood: np.ndarray, proba_ood: np.ndarray,
) -> dict[str, float]:
    ok_in = validator.accept(x_in, proba_in)
    ok_ood = validator.accept(x_ood, proba_ood)
    expected = np.r_[np.zeros(len(ok_in), dtype=int), np.ones(len(ok_ood), dtype=int)]
    rejected = np.r_[~ok_in, ~ok_ood].astype(int)
    scores = np.r_[validator.ood_score(x_in, proba_in), validator.ood_score(x_ood, proba_ood)]
    return {
        "rejection_precision": float(precision_score(expected, rejected, zero_division=0)),
        "rejection_recall": float(recall_score(expected, rejected, zero_division=0)),
        "false_reject_rate": float((~ok_in).mean()),
        "ood_accept_rate": float(ok_ood.mean()),
        "roc_auc": float(roc_auc_score(expected, scores)),
    }


def save_validator(model: ValidatorModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, means=model.means, inv_covariances=model.inv_covariances,
        distance_threshold=model.distance_threshold,
        confidence_threshold=model.confidence_threshold,
        counts=model.counts if model.counts is not None else np.array([-1, -1]),
        covariance_regularization=model.covariance_regularization,
        features=np.asarray(model.features, dtype="U2"),
    )


def load_validator(path: str | Path, expected_sha256: str | None = None) -> ValidatorModel:
    if expected_sha256 is not None:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"validator_params hash mismatch: expected {expected_sha256}, got {actual}")
    with np.load(path, allow_pickle=False) as data:
        return ValidatorModel(
            means=data["means"], inv_covariances=data["inv_covariances"],
            distance_threshold=float(data["distance_threshold"]),
            confidence_threshold=float(data["confidence_threshold"]),
            counts=data["counts"] if "counts" in data.files else None,
            covariance_regularization=(
                float(data["covariance_regularization"])
                if "covariance_regularization" in data.files else 1e-6
            ),
            features=tuple(str(x) for x in data["features"]) if "features" in data.files else ("x1", "x2"),
        )
