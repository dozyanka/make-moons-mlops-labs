from fastapi.testclient import TestClient
import pandas as pd
import pytest
from pydantic import ValidationError

from moons_lab.generator import GeneratorSettings, LiveGeneratorController, dataframe_hash, generate_dataframe
from moons_lab.generator_service import create_generator_app


def test_generator_reproducibility_and_modes():
    cfg = GeneratorSettings(n_samples=1000, seed=10, noise=0.22)
    a = generate_dataframe(cfg)
    b = generate_dataframe(cfg)
    assert dataframe_hash(a) == dataframe_hash(b)
    random = generate_dataframe(cfg.model_copy(update={"mode": "random"}))
    assert len(random) == 1000
    assert abs(a["x1"].std() - random["x1"].std()) > 0.5


def test_each_drift_knob_changes_output():
    base = GeneratorSettings(n_samples=1000, seed=11)
    original = generate_dataframe(base)
    knobs = [
        {"shift_x": 1.0},
        {"shift_y": 1.0},
        {"rotation_deg": 20.0},
        {"scale": 1.4},
        {"class_1_ratio": 0.7},
        {"label_flip_rate": 0.1},
        {"noise": 0.4},
    ]
    for patch in knobs:
        changed = generate_dataframe(base.model_copy(update=patch))
        assert dataframe_hash(changed) != dataframe_hash(original)


def test_live_config_updates_without_restart():
    controller = LiveGeneratorController(GeneratorSettings(n_samples=50, seed=5))
    client = TestClient(create_generator_app(controller))
    assert client.get("/health").status_code == 200
    before_id = id(controller)
    response = client.patch("/config", json={"shift_x": 2.0})
    assert response.status_code == 200
    generated = client.post("/generate", json={"n_samples": 20, "seed": 5})
    assert generated.status_code == 200
    assert generated.json()["rows"] == 20
    assert id(controller) == before_id
    assert client.get("/config").json()["shift_x"] == 2.0


def test_bad_generator_config_rejected():
    with pytest.raises(ValidationError):
        GeneratorSettings(n_samples=1)
