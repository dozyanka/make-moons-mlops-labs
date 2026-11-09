"""LAB8-only validated settings. Never mutate frozen LAB4 src/moons_lab/config.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .config import Lab8Config as FrozenLab8Config


class Lab8FullConfig(FrozenLab8Config):
    host: str = "127.0.0.1"
    model_port: int = Field(default=18780, ge=1024, le=65535)
    monitor_port: int = Field(default=18781, ge=1024, le=65535)
    sample_interval_s: float = Field(default=1.0, gt=0, le=2)
    warmup_seconds: int = Field(default=300, ge=300)
    stage_seconds: int = Field(default=600, ge=600)
    min_stage_requests: int = Field(default=10_000, ge=10_000)
    request_timeout_s: float = Field(default=10, gt=0, le=60)
    batch_size: int = Field(default=8, ge=1, le=256)
    drift_shift_x: float = Field(default=4.0, ge=0.2, le=12)
    window_size: int = Field(default=500, ge=100)
    window_step: int = Field(default=250, ge=50)
    detection_deadline_windows: int = Field(default=2, ge=0)
    max_clean_alerts: int = Field(default=1, ge=0)
    max_latency_ratio: float = Field(default=1.5, gt=1, le=4)
    validator_drift_reject_min: float = Field(default=0.90, ge=0.9, le=1)
    baseline_p95_ms: float = Field(default=0, ge=0)
    stage_rate_multiplier: float = Field(default=2.0, ge=1.1, le=5)
    source: Literal["synthetic", "real"] = "synthetic"

    @model_validator(mode="after")
    def validate_complete(self) -> "Lab8FullConfig":
        if self.model_port == self.monitor_port:
            raise ValueError("model_port and monitor_port must differ")
        if self.window_step > self.window_size:
            raise ValueError("window_step cannot exceed window_size")
        return self


def load_lab8_config(path: str | Path = "configs/lab8_full.json") -> Lab8FullConfig:
    with Path(path).open(encoding="utf-8") as f:
        return Lab8FullConfig.model_validate(json.load(f))
