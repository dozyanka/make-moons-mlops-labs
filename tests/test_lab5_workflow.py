"""LAB5 integration: compare only the 2 numeric features, guard old artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from moons_lab.config import load_config
from moons_lab.generator import generate_dataframe


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab5_runner", ROOT / "scripts" / "run_lab5.py")
assert spec and spec.loader
lab5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab5)


def test_reference_is_immutable_and_sha_matches_manifest() -> None:
    reference, sha = lab5.read_reference(ROOT / "data/raw/moons.csv")
    baseline = json.loads((ROOT / "reports/LAB1/hash_manifest.json").read_text())
    assert sha == baseline["raw"]["sha256"]
    assert len(reference) == 24000


def test_distributional_criteria_on_real_data() -> None:
    cfg = load_config(5)
    settings = lab5.settings_from_config(cfg)
    reference, _ = lab5.read_reference(ROOT / "data/raw/moons.csv")
    frames = {mode: generate_dataframe(setting) for mode, setting in settings.items()}
    metrics = lab5.compare(reference, frames)
    verdict = lab5.thresholds(metrics)
    assert verdict["similar_pass"]
    assert verdict["random_pass"]
    assert verdict["similar_fraction"] == 1.0
    assert verdict["random_fraction"] == 1.0
    assert len(metrics) == 9
    assert set(metrics["feature"]) == {"x1", "x2", "class_1_share"}
    assert frames["drifted"]["y"].mean() > frames["similar"]["y"].mean() + 0.10


def test_class_psi_and_wrong_schema() -> None:
    p = pd.Series([0, 1] * 100)
    q = pd.Series([0] * 100 + [1] * 100)
    assert lab5.class_psi(p, q) == pytest.approx(0)
    assert lab5.class_psi(p, pd.Series([0] * 170 + [1] * 30)) > 0.1
    with pytest.raises(ValueError):
        lab5.thresholds(pd.DataFrame())


def test_reference_hash_guard(tmp_path: Path) -> None:
    invalid = tmp_path / "fake.csv"
    invalid.write_text("x1,x2,y\n0,0,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        lab5.read_reference(invalid)


def test_all_presets_have_complete_declared_schema() -> None:
    for settings in lab5.settings_from_config(load_config(5)).values():
        assert tuple(settings.output_columns) == ("x1", "x2", "y")
        assert settings.seed == 77
        assert settings.n_samples == 12000
