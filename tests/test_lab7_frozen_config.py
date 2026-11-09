"""Ensure LAB7 extensions never alter the source archived for LAB4 provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from moons_lab.config import Lab7Config as FrozenLab7Config
from moons_lab.config import load_config as frozen_load_config
from moons_lab.lab7_config import Lab7Config, load_lab7_config

ROOT = Path(__file__).resolve().parents[1]


def test_published_lab4_config_file_is_byte_identical():
    manifest = json.loads((ROOT / "reports/LAB4/source_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["files_sha256"]["src/moons_lab/config.py"]
    current = hashlib.sha256((ROOT / "src/moons_lab/config.py").read_bytes()).hexdigest()
    assert current == expected


def test_lab7_extended_schema_and_legacy_loader_are_isolated():
    source = ROOT / "configs/lab7.json"
    cfg = load_lab7_config(source)
    assert isinstance(cfg, Lab7Config)
    assert cfg.reference_size >= 100
    assert 10 <= cfg.window_step <= cfg.window_size
    assert 0.05 <= cfg.drift_class_ratio <= 0.95
    assert cfg.detection_deadline_windows >= 0
    assert cfg.max_clean_false_alerts >= 0
    legacy = frozen_load_config(7, source)
    assert isinstance(legacy, FrozenLab7Config)
    assert type(legacy) is FrozenLab7Config
    assert "window_step" not in FrozenLab7Config.model_fields
    with pytest.raises(ValidationError, match="window_step"):
        Lab7Config.model_validate({**cfg.model_dump(), "window_step": cfg.window_size + 1})
