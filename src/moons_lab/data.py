from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from .config import Lab1Config
from .utils import json_dump, sha256_file

FEATURES = ["x1", "x2"]
TARGET = "y"


@dataclass(frozen=True)
class DataPaths:
    raw: Path = Path("data/raw/moons.csv")
    train: Path = Path("data/processed/train.csv")
    validation: Path = Path("data/processed/validation.csv")
    test: Path = Path("data/processed/test.csv")
    chunks_dir: Path = Path("data/chunks")


def make_raw_dataframe(config: Lab1Config) -> pd.DataFrame:
    x, y = make_moons(
        n_samples=config.n_samples,
        noise=config.noise,
        random_state=config.seed,
    )
    return pd.DataFrame(
        {
            "x1": x[:, 0],
            "x2": x[:, 1],
            "y": y.astype(int),
        }
    )


def prepare_data(
    config: Lab1Config,
    paths: DataPaths = DataPaths(),
    manifest_path: str | Path | None = "reports/LAB1/hash_manifest.json",
) -> dict[str, object]:
    for path in [
        paths.raw.parent,
        paths.train.parent,
        paths.chunks_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    df = make_raw_dataframe(config)
    df.to_csv(
        paths.raw,
        index=False,
        float_format="%.10f",
    )

    test_size = int(round(len(df) * config.test_fraction))
    validation_size = int(round(len(df) * config.validation_fraction))

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=config.seed,
        stratify=df[TARGET],
    )

    train, validation = train_test_split(
        train_val,
        test_size=validation_size,
        random_state=config.seed,
        stratify=train_val[TARGET],
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(
        paths.train,
        index=False,
        float_format="%.10f",
    )
    validation.to_csv(
        paths.validation,
        index=False,
        float_format="%.10f",
    )
    test.to_csv(
        paths.test,
        index=False,
        float_format="%.10f",
    )

    for old in paths.chunks_dir.glob("train_chunk_*.csv"):
        old.unlink()

    chunk_paths: list[str] = []

    for idx, start in enumerate(
        range(0, len(train), config.chunk_size)
    ):
        chunk = train.iloc[
            start : start + config.chunk_size
        ]

        path = (
            paths.chunks_dir
            / f"train_chunk_{idx:03d}.csv"
        )

        chunk.to_csv(
            path,
            index=False,
            float_format="%.10f",
        )

        chunk_paths.append(str(path))

    if len(chunk_paths) <= 10:
        raise RuntimeError(
            "expected more than 10 chunks"
        )

    manifest = {
        "source": "sklearn.datasets.make_moons",
        "generator": config.model_dump(),
        "raw": {
            "path": str(paths.raw),
            "sha256": sha256_file(paths.raw),
            "bytes": paths.raw.stat().st_size,
        },
        "splits": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "chunks": len(chunk_paths),
        },
    }

    if manifest_path is not None:
        json_dump(
            manifest_path,
            manifest,
        )

    return manifest


def load_xy(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    return (
        df[FEATURES].to_numpy(dtype=np.float64),
        df[TARGET].to_numpy(dtype=np.int64),
    )


def iter_chunk_xy(
    chunks_dir: str | Path = "data/chunks",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    files = sorted(
        Path(chunks_dir).glob("train_chunk_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"no chunks in {chunks_dir}"
        )

    for path in files:
        yield load_xy(path)


def iter_csv_batches(
    path: str | Path,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for df in pd.read_csv(
        path,
        chunksize=chunk_size,
    ):
        yield (
            df[FEATURES].to_numpy(dtype=np.float64),
            df[TARGET].to_numpy(dtype=np.int64),
        )
