from pathlib import Path

import numpy as np
import pandas as pd

from moons_lab.config import Lab1Config
from moons_lab.data import DataPaths, iter_csv_batches, iter_chunk_xy, load_xy, make_raw_dataframe, prepare_data


def test_make_raw_dataframe_is_deterministic():
    cfg = Lab1Config(
        seed=7,
        n_samples=2000,
        noise=0.2,
        test_fraction=0.15,
        validation_fraction=0.15,
        chunk_size=100,
    )
    a = make_raw_dataframe(cfg)
    b = make_raw_dataframe(cfg)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.columns) == {"x1", "x2", "y"}


def test_prepare_data_and_iterators(tmp_path):
    cfg = Lab1Config(
        seed=3,
        n_samples=2200,
        noise=0.2,
        test_fraction=0.15,
        validation_fraction=0.15,
        chunk_size=100,
    )
    paths = DataPaths(
        raw=tmp_path / "raw.csv",
        train=tmp_path / "train.csv",
        validation=tmp_path / "val.csv",
        test=tmp_path / "test.csv",
        chunks_dir=tmp_path / "chunks",
    )
    manifest = prepare_data(cfg, paths, manifest_path=tmp_path / "manifest.json")
    assert manifest["splits"]["chunks"] > 10
    x, y = load_xy(paths.train)
    assert x.shape[1] == 2 and len(x) == len(y)
    chunks = list(iter_chunk_xy(paths.chunks_dir))
    assert sum(len(cx) for cx, _ in chunks) == len(x)
    batches = list(iter_csv_batches(paths.train, 111))
    assert sum(len(bx) for bx, _ in batches) == len(x)
