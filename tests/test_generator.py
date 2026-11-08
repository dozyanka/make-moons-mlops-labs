"""LAB5 acceptance tests: deterministic generation, schema and every drift knob."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from moons_lab.generator import (
    GeneratorSettings,
    LiveGeneratorController,
    dataframe_bytes,
    dataframe_hash,
    generate_dataframe,
    write_dataframe_csv,
)
from moons_lab.generator_service import create_generator_app


def _settings(**updates: object) -> GeneratorSettings:
    return GeneratorSettings.model_validate({
        "mode": "similar", "n_samples": 2000, "seed": 42, "noise": 0.22,
        **updates,
    })


def test_output_schema_ranges_and_dtypes() -> None:
    for mode in ("similar", "random"):
        frame = generate_dataframe(_settings(mode=mode))
        assert list(frame.columns) == ["x1", "x2", "y"]
        assert frame.shape == (2000, 3)
        assert frame["x1"].dtype == "float64"
        assert frame["x2"].dtype == "float64"
        assert pd.api.types.is_integer_dtype(frame["y"].dtype)
        assert set(frame["y"].unique()) == {0, 1}
        assert np.isfinite(frame[["x1", "x2"]].to_numpy()).all()


def test_reproducibility_is_byte_identical_and_seed_matters(tmp_path) -> None:
    for mode in ("similar", "random"):
        cfg = _settings(mode=mode)
        first = generate_dataframe(cfg)
        second = generate_dataframe(cfg)
        assert dataframe_bytes(first) == dataframe_bytes(second)
        output = tmp_path / f"{mode}.csv"
        assert write_dataframe_csv(first, output) == dataframe_hash(second)
        assert output.read_bytes() == dataframe_bytes(second)
        assert b"\r\n" not in output.read_bytes()
        changed = generate_dataframe(_settings(mode=mode, seed=43))
        assert dataframe_hash(first) != dataframe_hash(changed)


@pytest.mark.parametrize("patch", [
    {"shift_x": 1.0},
    {"shift_y": -0.5},
    {"rotation_deg": 25.0},
    {"scale": 1.5},
    {"class_1_ratio": 0.7},
    {"label_flip_rate": 0.13},
    {"noise": 0.4},
    {"seasonality_amplitude": 0.5},
    {"seasonality_period": 31, "seasonality_amplitude": 0.5},
])
def test_drift_knob_changes_data(patch: dict) -> None:
    original = generate_dataframe(_settings())
    changed = generate_dataframe(_settings(**patch))
    assert dataframe_hash(original) != dataframe_hash(changed)


def test_mean_shifts_are_exact_with_fixed_seed() -> None:
    baseline = generate_dataframe(_settings())
    changed = generate_dataframe(_settings(shift_x=1.25, shift_y=-0.75))
    np.testing.assert_allclose(changed["x1"] - baseline["x1"], 1.25, atol=1e-12)
    np.testing.assert_allclose(changed["x2"] - baseline["x2"], -0.75, atol=1e-12)
    assert baseline["y"].equals(changed["y"])


def test_rotation_preserves_lengths() -> None:
    original = generate_dataframe(_settings())
    rotated = generate_dataframe(_settings(rotation_deg=90))
    np.testing.assert_allclose(
        np.linalg.norm(original[["x1", "x2"]].to_numpy(), axis=1),
        np.linalg.norm(rotated[["x1", "x2"]].to_numpy(), axis=1),
        atol=1e-12,
    )


def test_class_ratio_and_label_flip_count() -> None:
    original = generate_dataframe(_settings())
    rebalanced = generate_dataframe(_settings(class_1_ratio=0.7))
    assert rebalanced["y"].sum() == 1400
    flipped = generate_dataframe(_settings(label_flip_rate=0.13))
    assert (original["y"].to_numpy() != flipped["y"].to_numpy()).sum() == 260
    np.testing.assert_array_equal(original[["x1", "x2"]], flipped[["x1", "x2"]])


def test_seasonality_changes_only_x2() -> None:
    original = generate_dataframe(_settings())
    changed = generate_dataframe(_settings(seasonality_amplitude=0.4, seasonality_period=20))
    np.testing.assert_array_equal(original["x1"], changed["x1"])
    np.testing.assert_array_equal(original["y"], changed["y"])
    assert abs((changed["x2"] - original["x2"]).std()) > 0.1


def test_random_mode_respects_feature_bounds() -> None:
    data = generate_dataframe(_settings(mode="random", random_range=3.0))
    assert data[["x1", "x2"]].to_numpy().min() >= -3
    assert data[["x1", "x2"]].to_numpy().max() <= 3


@pytest.mark.parametrize("bad", [
    {"mode": "unknown"}, {"n_samples": 9}, {"n_samples": -1},
    {"seed": -1}, {"noise": 1.2}, {"rotation_deg": 500},
    {"scale": 0}, {"class_1_ratio": 1},
    {"label_flip_rate": 0.9}, {"random_range": 0},
    {"seasonality_amplitude": -1}, {"seasonality_period": 2},
    {"output_columns": ["x2", "x1", "y"]},
    {"shift_x": float("nan")}, {"shift_y": float("inf")},
    {"unknown_setting": "not allowed"},
])
def test_bad_configs_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        _settings(**bad)


def test_bad_csv_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected columns"):
        dataframe_bytes(pd.DataFrame({"x1": [0], "x2": [1]}))


def test_runtime_controller_updates_without_restart_and_validates() -> None:
    controller = LiveGeneratorController(_settings(n_samples=100))
    first_id = id(controller)
    baseline = controller.generate()
    new_settings = controller.update({"shift_x": 2.0})
    assert new_settings.shift_x == 2.0
    shifted = controller.generate()
    np.testing.assert_allclose(shifted["x1"] - baseline["x1"], 2.0, atol=1e-12)
    assert id(controller) == first_id
    with pytest.raises(ValidationError):
        controller.update({"class_1_ratio": 2.0})
    assert controller.get_settings().shift_x == 2.0
    with pytest.raises(ValidationError):
        controller.generate(n_samples=3)
    with pytest.raises(ValidationError):
        controller.generate(seed=-5)


def test_generator_http_api() -> None:
    controller = LiveGeneratorController(_settings(n_samples=50))
    client = TestClient(create_generator_app(controller))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/config").json()["mode"] == "similar"
    update = client.patch("/config", json={"shift_x": 2.0, "seasonality_amplitude": 0.1})
    assert update.status_code == 200
    assert update.json()["shift_x"] == 2.0
    response = client.post("/generate", json={"n_samples": 20, "seed": 5})
    assert response.status_code == 200
    assert response.json()["rows"] == 20
    assert len(response.json()["records"]) == 20


def test_distinct_modes_are_statistically_different() -> None:
    a = generate_dataframe(_settings())
    b = generate_dataframe(_settings(mode="random"))
    assert abs(a["x1"].std() - b["x1"].std()) > 0.5
