from __future__ import annotations

import hashlib
import json
import os
import random
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class Measurement:
    value: Any
    elapsed_s: float
    peak_memory_mb: float


def measured_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> Measurement:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        value = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        elapsed = time.perf_counter() - start
        tracemalloc.stop()
    return Measurement(value=value, elapsed_s=elapsed, peak_memory_mb=peak / 1024 / 1024)
