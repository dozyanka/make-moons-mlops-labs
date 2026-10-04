from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

from .data import FEATURES, TARGET, iter_chunk_xy, load_xy
from .metrics import classification_metrics
from .utils import seed_everything


class MoonMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ChunkIterableDataset(IterableDataset):
    def __init__(self, chunks_dir: str | Path, mean: np.ndarray, std: np.ndarray, batch_size: int):
        super().__init__()
        self.chunks_dir = Path(chunks_dir)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Читаем строго один CSV-чанк за раз и сразу отдаём тензорные мини-батчи.
        # Это сохраняет ограниченную память, но не платит Python-overhead за yield каждой строки.
        for path in sorted(self.chunks_dir.glob("train_chunk_*.csv")):
            df = pd.read_csv(path)
            x = df[FEATURES].to_numpy(dtype=np.float32)
            y = df[TARGET].to_numpy(dtype=np.float32)
            x = (x - self.mean) / self.std
            for start in range(0, len(x), self.batch_size):
                stop = min(start + self.batch_size, len(x))
                yield torch.from_numpy(x[start:stop]), torch.from_numpy(y[start:stop])


@dataclass
class DLResult:
    model: MoonMLP
    mean: np.ndarray
    std: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int

    def predict_proba(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.model.eval()
        target = torch.device(device)
        self.model.to(target)
        standardized = (np.asarray(x, dtype=np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(standardized).to(target))
            probs = torch.sigmoid(logits).cpu().numpy()
        self.model.to("cpu")
        return probs


def _stats_full(train_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    x, _ = load_xy(train_path)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.maximum(std, 1e-8)


def _stats_chunks(chunks_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    for x, _ in iter_chunk_xy(chunks_dir):
        count += len(x)
        total += x.sum(axis=0)
        total_sq += np.square(x).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean, np.sqrt(variance)


def _validation_tensors(path: str | Path, mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_xy(path)
    x = ((x - mean) / std).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def _fit(
    loader_factory,
    validation_path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> DLResult:
    seed_everything(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    target = torch.device(device)
    model = MoonMLP().to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    x_val, y_val = _validation_tensors(validation_path, mean, std)
    x_val, y_val = x_val.to(target), y_val.to(target)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in loader_factory():
            xb = xb.to(target)
            yb = yb.to(target)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(criterion(val_logits, y_val).cpu())
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_metrics = classification_metrics(y_val.cpu().numpy().astype(int), val_prob)
        row = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(n, 1),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.to("cpu")
    return DLResult(model=model, mean=mean, std=std, history=history, best_epoch=best_epoch)


def train_full(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    batch_size: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> DLResult:
    x, y = load_xy(train_path)
    mean, std = _stats_full(train_path)
    xs = ((x - mean) / std).astype(np.float32)
    ys = y.astype(np.float32)
    dataset: Dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))

    def loader_factory() -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return _fit(
        loader_factory,
        validation_path,
        mean,
        std,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
    )


def train_stream(
    chunks_dir: str | Path,
    validation_path: str | Path,
    *,
    batch_size: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> DLResult:
    mean, std = _stats_chunks(chunks_dir)

    def loader_factory() -> DataLoader:
        dataset = ChunkIterableDataset(chunks_dir, mean, std, batch_size=batch_size)
        # Dataset уже выдаёт готовые мини-батчи; batch_size=None запрещает повторную пакетизацию.
        return DataLoader(dataset, batch_size=None, num_workers=0)

    return _fit(
        loader_factory,
        validation_path,
        mean,
        std,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
    )


def save_checkpoint(result: DLResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "mean": result.mean,
            "std": result.std,
            "best_epoch": result.best_epoch,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> DLResult:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MoonMLP()
    model.load_state_dict(payload["state_dict"])
    return DLResult(
        model=model,
        mean=np.asarray(payload["mean"]),
        std=np.asarray(payload["std"]),
        history=[],
        best_epoch=int(payload.get("best_epoch", 0)),
    )


def device_parity(result: DLResult, x: np.ndarray) -> dict[str, float | str]:
    cpu = result.predict_proba(x, device="cpu")
    if not torch.cuda.is_available():
        return {"status": "not_available", "reason": "CUDA unavailable in current environment", "max_abs_diff": float("nan")}
    gpu = result.predict_proba(x, device="cuda")
    return {"status": "checked", "reason": "", "max_abs_diff": float(np.max(np.abs(cpu - gpu)))}
