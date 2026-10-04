from __future__ import annotations

import hashlib
import threading
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.datasets import make_moons


class GeneratorSettings(BaseModel):
    mode: Literal["similar", "random"] = "similar"
    n_samples: int = Field(default=1000, ge=10, le=1_000_000)
    seed: int = Field(default=42, ge=0)
    noise: float = Field(default=0.22, ge=0, le=1.0)
    shift_x: float = 0.0
    shift_y: float = 0.0
    rotation_deg: float = Field(default=0.0, ge=-360, le=360)
    scale: float = Field(default=1.0, gt=0, le=10)
    class_1_ratio: float = Field(default=0.5, gt=0.01, lt=0.99)
    label_flip_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    random_range: float = Field(default=4.0, gt=0.1, le=100)


def _rebalance(x: np.ndarray, y: np.ndarray, ratio: float, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    n1 = int(round(n * ratio))
    n0 = n - n1
    idx0 = rng.choice(np.flatnonzero(y == 0), size=n0, replace=True)
    idx1 = rng.choice(np.flatnonzero(y == 1), size=n1, replace=True)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return x[idx], y[idx]


def generate_dataframe(settings: GeneratorSettings) -> pd.DataFrame:
    rng = np.random.default_rng(settings.seed)
    if settings.mode == "similar":
        base_n = max(settings.n_samples * 2, 100)
        x, y = make_moons(n_samples=base_n, noise=settings.noise, random_state=settings.seed)
        x, y = _rebalance(x, y.astype(int), settings.class_1_ratio, rng, settings.n_samples)
    else:
        x = rng.uniform(-settings.random_range, settings.random_range, size=(settings.n_samples, 2))
        # Deliberately weak relation between x and y: this mode is distributionally alien.
        y = rng.binomial(1, settings.class_1_ratio, size=settings.n_samples).astype(int)

    theta = np.deg2rad(settings.rotation_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = (x @ rotation.T) * settings.scale
    x[:, 0] += settings.shift_x
    x[:, 1] += settings.shift_y

    if settings.label_flip_rate > 0:
        flip_n = int(round(settings.label_flip_rate * settings.n_samples))
        if flip_n:
            flip_idx = rng.choice(settings.n_samples, size=flip_n, replace=False)
            y[flip_idx] = 1 - y[flip_idx]

    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y.astype(int)})


def dataframe_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False, float_format="%.10f").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LiveGeneratorController:
    """Thread-safe runtime configuration; updates do not restart the process."""

    def __init__(self, settings: GeneratorSettings | None = None):
        self._lock = threading.RLock()
        self._settings = settings or GeneratorSettings()

    def get_settings(self) -> GeneratorSettings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def update(self, patch: dict) -> GeneratorSettings:
        with self._lock:
            payload = self._settings.model_dump()
            payload.update(patch)
            self._settings = GeneratorSettings.model_validate(payload)
            return self._settings.model_copy(deep=True)

    def generate(self, *, n_samples: int | None = None, seed: int | None = None) -> pd.DataFrame:
        with self._lock:
            settings = self._settings.model_copy(deep=True)
        patch = {}
        if n_samples is not None:
            patch["n_samples"] = n_samples
        if seed is not None:
            patch["seed"] = seed
        if patch:
            settings = settings.model_copy(update=patch)
        return generate_dataframe(settings)
