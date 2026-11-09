"""Bridge unchanged LAB3 checkpoint into the joblib-based LAB6 model registry."""
from __future__ import annotations

from pathlib import Path
import numpy as np


class DLJoblibAdapter:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self._model = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            from .dl_pipeline import load_checkpoint
            import torch
            torch.set_num_threads(1)
            self._model = load_checkpoint(self.checkpoint_path)
        return self._model.predict_proba(np.asarray(x, dtype=np.float64), device="cpu")

    def __getstate__(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "_model": None}
