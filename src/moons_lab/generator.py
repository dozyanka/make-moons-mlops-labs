"""Reproducible, configurable synthetic make_moons data for LAB5 and LAB6–8."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.datasets import make_moons


OUTPUT_COLUMNS = ("x1", "x2", "y")


class GeneratorSettings(BaseModel):
    """P1 / variant 09: two numeric features and a binary target."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mode: Literal["similar", "random"] = "similar"
    n_samples: int = Field(default=1000, ge=10, le=1_000_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    noise: float = Field(default=0.22, ge=0.0, le=1.0)
    shift_x: float = Field(default=0.0, ge=-100.0, le=100.0)
    shift_y: float = Field(default=0.0, ge=-100.0, le=100.0)
    rotation_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    scale: float = Field(default=1.0, gt=0.0, le=10.0)
    class_1_ratio: float = Field(default=0.5, ge=0.01, le=0.99)
    label_flip_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    random_range: float = Field(default=4.0, gt=0.1, le=100.0)
    # Optional controlled periodic perturbation, not a time-series assumption.
    seasonality_amplitude: float = Field(default=0.0, ge=0.0, le=3.0)
    seasonality_period: int = Field(default=100, ge=4, le=100_000)
    output_columns: tuple[str, str, str] = OUTPUT_COLUMNS

    @model_validator(mode="after")
    def validate_output_schema(self) -> "GeneratorSettings":
        if tuple(self.output_columns) != OUTPUT_COLUMNS:
            raise ValueError("output_columns must be exactly ['x1', 'x2', 'y']")
        return self


def _rebalance(
    x: np.ndarray, y: np.ndarray, ratio: float, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    n1 = int(round(n * ratio))
    n0 = n - n1
    idx0 = rng.choice(np.flatnonzero(y == 0), size=n0, replace=True)
    idx1 = rng.choice(np.flatnonzero(y == 1), size=n1, replace=True)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return x[idx].copy(), y[idx].copy()


def generate_dataframe(settings: GeneratorSettings) -> pd.DataFrame:
    """Return an independent, deterministic DataFrame in the declared schema."""
    settings = GeneratorSettings.model_validate(settings.model_dump())
    rng = np.random.default_rng(settings.seed)
    if settings.mode == "similar":
        # make_moons (not Gaussian blobs) preserves the nonlinear geometry.
        base_n = max(settings.n_samples * 2, 100)
        x, y = make_moons(
            n_samples=base_n,
            noise=settings.noise,
            random_state=settings.seed,
        )
        x, y = _rebalance(
            x, y.astype(np.int64), settings.class_1_ratio, rng, settings.n_samples
        )
    else:
        # Uniform noise is deliberately unrelated to the learned moon shapes.
        x = rng.uniform(
            -settings.random_range, settings.random_range, size=(settings.n_samples, 2)
        )
        y = rng.binomial(1, settings.class_1_ratio, size=settings.n_samples).astype(np.int64)

    theta = np.deg2rad(settings.rotation_deg)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    x = (x @ rotation.T) * settings.scale
    x[:, 0] += settings.shift_x
    x[:, 1] += settings.shift_y

    if settings.seasonality_amplitude:
        phase = np.arange(settings.n_samples, dtype=np.float64) / settings.seasonality_period
        x[:, 1] += settings.seasonality_amplitude * np.sin(2.0 * np.pi * phase)

    flip_n = int(round(settings.label_flip_rate * settings.n_samples))
    if flip_n:
        flip_idx = rng.choice(settings.n_samples, size=flip_n, replace=False)
        y[flip_idx] = 1 - y[flip_idx]

    return pd.DataFrame({
        "x1": x[:, 0].astype(np.float64),
        "x2": x[:, 1].astype(np.float64),
        "y": y.astype(np.int64),
    }, columns=list(settings.output_columns))


def dataframe_bytes(df: pd.DataFrame) -> bytes:
    """Stable UTF-8 CSV with LF and 10 decimal places on Windows and Linux."""
    if list(df.columns) != list(OUTPUT_COLUMNS):
        raise ValueError(f"expected columns {OUTPUT_COLUMNS}, got {tuple(df.columns)}")
    return df.to_csv(index=False, float_format="%.10f", lineterminator="\n").encode("utf-8")


def dataframe_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(dataframe_bytes(df)).hexdigest()


def write_dataframe_csv(df: pd.DataFrame, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dataframe_bytes(df)
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class LiveGeneratorController:
    """Thread-safe config updates without restarting the generator service."""

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
            payload = self._settings.model_dump()
        if n_samples is not None:
            payload["n_samples"] = n_samples
        if seed is not None:
            payload["seed"] = seed
        return generate_dataframe(GeneratorSettings.model_validate(payload))
