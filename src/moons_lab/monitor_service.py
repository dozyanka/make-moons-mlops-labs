from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


class Observation(BaseModel):
    x1: float
    x2: float
    probability: float = Field(ge=0, le=1)


@dataclass
class MonitorState:
    maxlen: int = 5000
    data: deque = field(default_factory=lambda: deque(maxlen=5000))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def add(self, obs: Observation) -> None:
        with self.lock:
            self.data.append((obs.x1, obs.x2, obs.probability))

    def metrics(self) -> dict:
        with self.lock:
            if not self.data:
                return {"count": 0, "x1_mean": None, "x2_mean": None, "probability_mean": None}
            arr = np.asarray(self.data, dtype=float)
        return {
            "count": len(arr),
            "x1_mean": float(arr[:, 0].mean()),
            "x2_mean": float(arr[:, 1].mean()),
            "probability_mean": float(arr[:, 2].mean()),
            "metric_descriptions": {
                "x1_mean/x2_mean": "текущие средние входных признаков; резкое изменение указывает на drift",
                "probability_mean": "средняя вероятность класса 1; отслеживает сдвиг предсказаний",
            },
        }

    def save_plot(self, path: str | Path) -> None:
        with self.lock:
            arr = np.asarray(self.data, dtype=float)
        if len(arr) == 0:
            arr = np.zeros((1, 3))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], s=10, alpha=0.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Последние наблюдения сервиса")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def create_monitor_app(state: MonitorState | None = None) -> FastAPI:
    state = state or MonitorState()
    app = FastAPI(title="make_moons monitoring service")
    app.state.monitor_state = state

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(obs: Observation) -> dict[str, int]:
        state.add(obs)
        return {"accepted": 1}

    @app.get("/metrics")
    def metrics() -> dict:
        return state.metrics()

    return app
