from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import joblib
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from .data import iter_chunk_xy, load_xy
from .metrics import classification_metrics
from .utils import seed_everything

BatchFactory = Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]]


@dataclass
class RBFSGDModel:
    scaler: StandardScaler
    rbf: RBFSampler
    classifier: SGDClassifier
    gamma: float
    n_components: int

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.rbf.transform(self.scaler.transform(np.asarray(x, dtype=np.float64)))
        return self.classifier.predict_proba(z)[:, 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def _fit_from_factory(
    batch_factory: BatchFactory,
    *,
    gamma: float,
    n_components: int,
    alpha: float,
    passes: int,
    seed: int,
) -> RBFSGDModel:
    seed_everything(seed)
    scaler = StandardScaler()
    seen = 0
    for x, _ in batch_factory():
        scaler.partial_fit(x)
        seen += len(x)
    if seen == 0:
        raise ValueError("empty training source")

    rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)
    # RBFSampler only needs feature dimensionality; first transformed batch is sufficient.
    first_x, _ = next(iter(batch_factory()))
    rbf.fit(scaler.transform(first_x[: min(8, len(first_x))]))

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        random_state=seed,
        max_iter=1,
        tol=None,
        shuffle=False,
    )
    first = True
    for _ in range(passes):
        for x, y in batch_factory():
            z = rbf.transform(scaler.transform(x))
            if first:
                clf.partial_fit(z, y, classes=np.array([0, 1], dtype=int))
                first = False
            else:
                clf.partial_fit(z, y)
    return RBFSGDModel(scaler=scaler, rbf=rbf, classifier=clf, gamma=gamma, n_components=n_components)


def full_batch_factory(train_path: str | Path, chunk_size: int) -> tuple[BatchFactory, int]:
    x, y = load_xy(train_path)

    def factory() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(x), chunk_size):
            yield x[start : start + chunk_size], y[start : start + chunk_size]

    return factory, len(x)


def chunk_batch_factory(chunks_dir: str | Path) -> BatchFactory:
    def factory() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from iter_chunk_xy(chunks_dir)

    return factory


def train_rbf_full(
    train_path: str | Path,
    *,
    chunk_size: int,
    gamma: float,
    n_components: int,
    alpha: float,
    passes: int,
    seed: int,
) -> RBFSGDModel:
    factory, _ = full_batch_factory(train_path, chunk_size)
    return _fit_from_factory(
        factory,
        gamma=gamma,
        n_components=n_components,
        alpha=alpha,
        passes=passes,
        seed=seed,
    )


def train_rbf_chunks(
    chunks_dir: str | Path,
    *,
    gamma: float,
    n_components: int,
    alpha: float,
    passes: int,
    seed: int,
) -> RBFSGDModel:
    return _fit_from_factory(
        chunk_batch_factory(chunks_dir),
        gamma=gamma,
        n_components=n_components,
        alpha=alpha,
        passes=passes,
        seed=seed,
    )


def train_linear_baseline(train_path: str | Path, seed: int) -> LogisticRegression:
    x, y = load_xy(train_path)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(x, y)
    return model


def tune_gamma(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    chunk_size: int,
    candidates: list[float],
    n_components: int,
    alpha: float,
    passes: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    x_val, y_val = load_xy(validation_path)
    rows: list[dict[str, float]] = []
    for gamma in candidates:
        model = train_rbf_full(
            train_path,
            chunk_size=chunk_size,
            gamma=gamma,
            n_components=n_components,
            alpha=alpha,
            passes=passes,
            seed=seed,
        )
        score = classification_metrics(y_val, model.predict_proba(x_val))["roc_auc"]
        rows.append({"gamma": float(gamma), "validation_roc_auc": score})
    best = max(rows, key=lambda r: (r["validation_roc_auc"], -r["gamma"]))
    return float(best["gamma"]), rows



def scaler_leakage_diagnostic(
    scaler: StandardScaler, train_x: np.ndarray, forbidden_x: np.ndarray, tol: float = 1e-10
) -> dict[str, float | bool]:
    """Detect the common negative-control defect: fitting scaling stats on train+holdout."""
    train_mean = np.asarray(train_x, dtype=np.float64).mean(axis=0)
    combined_mean = np.vstack([train_x, forbidden_x]).mean(axis=0)
    fitted_mean = np.asarray(scaler.mean_, dtype=np.float64)
    error_to_train = float(np.max(np.abs(fitted_mean - train_mean)))
    error_to_combined = float(np.max(np.abs(fitted_mean - combined_mean)))
    return {
        "error_to_train_mean": error_to_train,
        "error_to_train_plus_forbidden_mean": error_to_combined,
        "leak_suspected": bool(error_to_combined + tol < error_to_train),
    }

def save_model(model: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    return joblib.load(path)
