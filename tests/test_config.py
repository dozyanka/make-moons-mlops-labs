from pathlib import Path

import pytest
from pydantic import ValidationError

from moons_lab.config import Lab1Config, Lab2Config, load_config


def test_all_configs_load():
    for lab in range(1, 9):
        assert load_config(lab) is not None


def test_invalid_lab_number():
    with pytest.raises(ValueError):
        load_config(99)


def test_lab1_requires_more_than_ten_chunks():
    with pytest.raises(ValidationError):
        Lab1Config(
            seed=1,
            n_samples=1000,
            noise=0.2,
            test_fraction=0.15,
            validation_fraction=0.15,
            chunk_size=500,
        )


def test_lab2_rejects_negative_gamma():
    with pytest.raises(ValidationError):
        Lab2Config(
            seed=1,
            chunk_size=100,
            rbf_components=32,
            gamma_candidates=[1.0, -1.0],
            alpha=1e-4,
            passes=1,
            bootstrap_rounds=50,
        )
