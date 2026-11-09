"""LAB7-only Pydantic configuration; LAB4-published config.py stays byte-frozen.

The original ``moons_lab.config.Lab7Config`` and ``load_config`` are kept
unaltered because LAB4 publication provenance pins that file's SHA-256.
The extended LAB7 model inherits all original constraints and adds only
new LAB7 monitoring/benchmark controls.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, model_validator

from .config import Lab7Config as _FrozenLab7Config


class Lab7Config(_FrozenLab7Config):
    """Extended LAB7 schema without mutating earlier laboratory artifacts."""

    reference_size: int = Field(default=5000, ge=100)
    window_step: int = Field(default=500, ge=10)
    drift_class_ratio: float = Field(default=0.9, ge=0.05, le=0.95)
    detection_deadline_windows: int = Field(default=2, ge=0)
    max_clean_false_alerts: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_step(self) -> "Lab7Config":
        if self.window_step > self.window_size:
            raise ValueError("window_step must not exceed window_size")
        return self


def load_lab7_config(path: str | Path = "configs/lab7.json") -> Lab7Config:
    """Load the LAB7-specific schema, including the streaming extension fields."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return Lab7Config.model_validate(payload)
