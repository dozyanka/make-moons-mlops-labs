"""Independent LAB6 monitoring service with structured log and reproducible plot."""
from __future__ import annotations

import io
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: float = Field(ge=-20, le=20, allow_inf_nan=False)
    x2: float = Field(ge=-20, le=20, allow_inf_nan=False)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


@dataclass
class MonitorState:
    maxlen: int = 5000
    log_path: Path | None = None
    data: deque = field(default_factory=lambda: deque(maxlen=5000))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def add(self, obs: Observation) -> None:
        with self.lock:
            self.data.append((obs.x1, obs.x2, obs.probability))
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                          **obs.model_dump()}, ensure_ascii=False) + "\n")

    def metrics(self) -> dict:
        with self.lock:
            if not self.data:
                return {"count": 0, "x1_mean": None, "x2_mean": None, "probability_mean": None}
            arr = np.asarray(list(self.data), dtype=float)
        return {"count": len(arr), "x1_mean": float(arr[:, 0].mean()),
                "x2_mean": float(arr[:, 1].mean()),
                "probability_mean": float(arr[:, 2].mean()),
                "metric_descriptions": {
                    "x1_mean/x2_mean": "Средние входных признаков, возможный дрейф",
                    "probability_mean": "Средняя вероятность класса 1"}}

    def plot_png(self) -> bytes:
        with self.lock:
            arr = np.asarray(list(self.data), dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(arr):
            ax.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2], s=12, alpha=.6, cmap="viridis")
        ax.set(xlabel="x1", ylabel="x2", title="LAB6: monitored model inputs")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        return buffer.getvalue()

    def save_plot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.plot_png())


def create_monitor_app(state: MonitorState | None = None) -> FastAPI:
    state = state if state is not None else MonitorState()
    app = FastAPI(title="make_moons independent monitoring service")
    app.state.monitor_state = state

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "monitor"}

    @app.post("/ingest")
    def ingest(obs: Observation) -> dict:
        state.add(obs)
        return {"accepted": 1}

    @app.get("/metrics")
    def metrics() -> dict:
        return state.metrics()

    @app.get("/plot.png")
    def plot() -> Response:
        return Response(content=state.plot_png(), media_type="image/png")

    return app
