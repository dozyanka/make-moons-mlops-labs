from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

    def distances(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        all_dist = []
        for mean, inv_cov in zip(self.means, self.inv_covariances):
            delta = x - mean
            squared = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
            all_dist.append(np.sqrt(np.maximum(squared, 0.0)))
        return np.min(np.vstack(all_dist), axis=0)

    def accept(self, x: np.ndarray, model_proba: np.ndarray) -> np.ndarray:
        confidence = np.maximum(model_proba, 1.0 - model_proba)
        return (self.distances(x) <= self.distance_threshold) & (confidence >= self.confidence_threshold)

    def ood_score(self, x: np.ndarray, model_proba: np.ndarray) -> np.ndarray:
        confidence = np.maximum(model_proba, 1.0 - model_proba)
        d = self.distances(x) / max(self.distance_threshold, 1e-12)
        confidence_penalty = np.maximum(self.confidence_threshold - confidence, 0.0) * 4.0
        return d + confidence_penalty

    def explain(self, x: np.ndarray, model_proba: np.ndarray) -> list[dict[str, float | bool | str]]:
        """Return an interpretable accept/reject decision for each input row."""
        x = np.asarray(x, dtype=np.float64)
        model_proba = np.asarray(model_proba, dtype=np.float64)
        distance = self.distances(x)
        confidence = np.maximum(model_proba, 1.0 - model_proba)
        accepted = (distance <= self.distance_threshold) & (confidence >= self.confidence_threshold)
        rows: list[dict[str, float | bool | str]] = []
        for d, conf, ok in zip(distance, confidence, accepted):
            if ok:
                reason = "input_is_similar_to_training_distribution"
            elif d > self.distance_threshold and conf < self.confidence_threshold:
                reason = "too_far_from_training_distribution_and_low_model_confidence"
            elif d > self.distance_threshold:
                reason = "too_far_from_training_distribution"
            else:
                reason = "low_model_confidence"
            rows.append({
                "accepted": bool(ok),
                "reason": reason,
                "distance": float(d),
                "distance_threshold": float(self.distance_threshold),
                "confidence": float(conf),
                "confidence_threshold": float(self.confidence_threshold),
            })
        return rows


def _class_stats_full(x: np.ndarray, y: np.ndarray) -> list[ClassStats]:
    stats = []
    for label in [0, 1]:
        part = x[y == label]
        stats.append(
            ClassStats(
                count=len(part),
                mean=part.mean(axis=0),
                covariance=np.cov(part, rowvar=False, bias=True),
            )
        )
    return stats


def _class_stats_online(batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[ClassStats]:
    counts = [0, 0]
    means = [np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)]
    m2 = [np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2), dtype=np.float64)]
    for x, y in batches:
        for row, label_raw in zip(x, y):
            label = int(label_raw)
            counts[label] += 1
            delta = row - means[label]
            means[label] += delta / counts[label]
            delta2 = row - means[label]
            m2[label] += np.outer(delta, delta2)
    result = []
    for label in [0, 1]:
        if counts[label] == 0:
            raise ValueError(f"class {label} not observed")
        result.append(
            ClassStats(
                count=counts[label],
                mean=means[label],
                covariance=m2[label] / counts[label],
            )
        )
    return result


def _build(stats: list[ClassStats], distance_threshold: float, confidence_threshold: float) -> ValidatorModel:
    means = np.stack([s.mean for s in stats])
    inv_cov = np.stack([np.linalg.inv(s.covariance + np.eye(2) * 1e-6) for s in stats])
    return ValidatorModel(means, inv_cov, distance_threshold, confidence_threshold)


def fit_full(train_path: str | Path, confidence_threshold: float) -> ValidatorModel:
    x, y = load_xy(train_path)
    return _build(_class_stats_full(x, y), float("inf"), confidence_threshold)


def fit_chunks(chunks_dir: str | Path, confidence_threshold: float) -> ValidatorModel:
    return _build(_class_stats_online(iter_chunk_xy(chunks_dir)), float("inf"), confidence_threshold)


def calibrate_threshold(
    validator: ValidatorModel,
    x_validation: np.ndarray,
    proba_validation: np.ndarray,
    target_false_reject_rate: float,
) -> float:
    confidence = np.maximum(proba_validation, 1.0 - proba_validation)
    conf_reject = float((confidence < validator.confidence_threshold).mean())
    remaining = max(target_false_reject_rate - conf_reject, 0.001)
    quantile = min(0.9999, max(0.90, 1.0 - remaining))
    threshold = float(np.quantile(validator.distances(x_validation), quantile))
    validator.distance_threshold = threshold
    return threshold


def make_ood_samples(n: int, shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2
    shifted = rng.normal(loc=np.array([shift, shift * 0.45]), scale=np.array([0.6, 0.45]), size=(half, 2))
    outliers = rng.uniform(low=-5.5, high=5.5, size=(n - half, 2))
    return np.vstack([shifted, outliers])


def evaluate_validator(
    validator: ValidatorModel,
    x_in: np.ndarray,
    proba_in: np.ndarray,
    x_ood: np.ndarray,
    proba_ood: np.ndarray,
) -> dict[str, float]:
    accept_in = validator.accept(x_in, proba_in)
    accept_ood = validator.accept(x_ood, proba_ood)
    expected = np.concatenate([np.zeros(len(x_in), dtype=int), np.ones(len(x_ood), dtype=int)])
    predicted_reject = np.concatenate([~accept_in, ~accept_ood]).astype(int)
    scores = np.concatenate(
        [validator.ood_score(x_in, proba_in), validator.ood_score(x_ood, proba_ood)]
    )
    return {
        "rejection_precision": float(precision_score(expected, predicted_reject, zero_division=0)),
        "rejection_recall": float(recall_score(expected, predicted_reject, zero_division=0)),
        "false_reject_rate": float((~accept_in).mean()),
        "ood_accept_rate": float(accept_ood.mean()),
        "roc_auc": float(roc_auc_score(expected, scores)),
    }


def save_validator(model: ValidatorModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        means=model.means,
        inv_covariances=model.inv_covariances,
        distance_threshold=model.distance_threshold,
        confidence_threshold=model.confidence_threshold,
    )


def load_validator(path: str | Path) -> ValidatorModel:
    data = np.load(path)
    return ValidatorModel(
        means=data["means"],
        inv_covariances=data["inv_covariances"],
        distance_threshold=float(data["distance_threshold"]),
        confidence_threshold=float(data["confidence_threshold"]),
    )
