from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Lab1Config(BaseModel):
    variant: int = Field(default=9, ge=1)
    seed: int = Field(ge=0)
    n_samples: int = Field(ge=1000)
    noise: float = Field(ge=0.0, le=1.0)
    test_fraction: float = Field(gt=0.0, lt=0.5)
    validation_fraction: float = Field(gt=0.0, lt=0.5)
    chunk_size: int = Field(ge=50)

    @model_validator(mode="after")
    def validate_split(self) -> "Lab1Config":
        if self.test_fraction + self.validation_fraction >= 0.6:
            raise ValueError("test_fraction + validation_fraction must be < 0.6")
        train_n = int(self.n_samples * (1 - self.test_fraction - self.validation_fraction))
        if train_n // self.chunk_size <= 10:
            raise ValueError("configuration must produce more than 10 training chunks")
        return self


class Lab2Config(BaseModel):
    seed: int = Field(ge=0)
    chunk_size: int = Field(ge=50)
    rbf_components: int = Field(ge=16, le=4096)
    gamma_candidates: list[float] = Field(min_length=2)
    alpha: float = Field(gt=0)
    passes: int = Field(ge=1, le=30)
    bootstrap_rounds: int = Field(ge=50, le=5000)

    @model_validator(mode="after")
    def validate_gamma(self) -> "Lab2Config":
        if any(g <= 0 for g in self.gamma_candidates):
            raise ValueError("all gamma values must be positive")
        return self


class Lab3Config(BaseModel):
    seed: int = Field(ge=0)
    batch_size: int = Field(ge=8, le=4096)
    epochs: int = Field(ge=1, le=500)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    patience: int = Field(ge=1)
    bootstrap_rounds: int = Field(ge=50)
    device: Literal["cpu", "cuda", "auto"] = "auto"


class Lab4Config(BaseModel):
    seed: int = Field(ge=0)
    chunk_size: int = Field(ge=50)
    target_false_reject_rate: float = Field(gt=0, lt=0.5)
    confidence_threshold: float = Field(ge=0.5, lt=1.0)
    ood_samples: int = Field(ge=100)
    ood_shift: float = Field(gt=0)


class Lab5Config(BaseModel):
    seed: int = Field(ge=0)
    n_samples: int = Field(ge=100)
    similar_noise: float = Field(ge=0, le=1)
    random_range: float = Field(gt=0)
    drift_shift_x: float
    drift_shift_y: float
    drift_rotation_deg: float
    drift_label_flip_rate: float = Field(ge=0, le=0.5)


class Lab6Config(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    registry_path: str
    load_test_requests: int = Field(ge=10)
    metrics_enabled: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Lab7Config(BaseModel):
    seed: int = Field(ge=0)
    benchmark_batch_size: int = Field(ge=100)
    benchmark_repeats: int = Field(ge=2)
    jobs: list[int] = Field(min_length=2)
    window_size: int = Field(ge=100)
    n_windows: int = Field(ge=3)
    drift_start_window: int = Field(ge=1)
    drift_shift_x: float
    drift_shift_y: float
    psi_alert_threshold: float = Field(gt=0)
    csi_alert_threshold: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_windows(self) -> "Lab7Config":
        if self.drift_start_window >= self.n_windows:
            raise ValueError("drift_start_window must be before n_windows")
        if any(j < 1 for j in self.jobs):
            raise ValueError("jobs must be positive")
        return self


class Lab8Config(BaseModel):
    seed: int = Field(ge=0)
    drift_psi_threshold: float = Field(gt=0)
    validator_reject_rate_threshold: float = Field(gt=0, lt=1)
    service_error_rate_threshold: float = Field(ge=0, lt=1)
    alert_cooldown_seconds: int = Field(ge=0)


_CONFIG_TYPES = {
    1: Lab1Config,
    2: Lab2Config,
    3: Lab3Config,
    4: Lab4Config,
    5: Lab5Config,
    6: Lab6Config,
    7: Lab7Config,
    8: Lab8Config,
}


def load_config(lab_number: int, path: str | Path | None = None) -> BaseModel:
    if lab_number not in _CONFIG_TYPES:
        raise ValueError(f"unknown lab number: {lab_number}")
    path = Path(path or f"configs/lab{lab_number}.json")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return _CONFIG_TYPES[lab_number].model_validate(payload)
