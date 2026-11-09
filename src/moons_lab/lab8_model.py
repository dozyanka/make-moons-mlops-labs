"""LAB8-only inference bridge for the immutable, published LAB3 PyTorch model.

The LAB3 implementation moves the module to CPU for every predict_proba call;
that overhead and shared-module mutation are avoidable for a CPU-only service.
The checkpoint and mathematical operations are unchanged.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

# Guards the one-time lazy load if multiple FastAPI requests arrive together.
_LOAD_LOCK = Lock()


class DLJoblibAdapter:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self._model = None
        self._mean = None
        self._std = None

    def _initialize(self) -> None:
        if self._model is not None:
            return
        with _LOAD_LOCK:
            if self._model is not None:
                return
            import torch
            from .dl_pipeline import load_checkpoint

            # CPU inference on small slices: avoid nested OpenMP oversubscription
            # when multiple independent batches run in parallel Python threads.
            torch.set_num_threads(1)
            checkpoint = load_checkpoint(self.checkpoint_path)
            checkpoint.model.eval()
            checkpoint.model.to("cpu")
            self._mean = np.asarray(checkpoint.mean, dtype=np.float32)
            self._std = np.asarray(checkpoint.std, dtype=np.float32)
            # Publish fully initialized state last under the lock.
            self._model = checkpoint.model

    def warmup(self) -> None:
        self.predict_proba(np.zeros((2, 2), dtype=np.float32))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._initialize()
        import torch

        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("expected numeric feature matrix of shape (n, 2)")
        if not np.isfinite(values).all():
            raise ValueError("non-finite feature values")
        normalized = np.ascontiguousarray((values - self._mean) / self._std)
        with torch.inference_mode():
            output = torch.sigmoid(self._model(torch.from_numpy(normalized)))
            return output.numpy().reshape(-1)

    def __getstate__(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "_model": None,
                "_mean": None, "_std": None}
