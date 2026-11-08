"""Independent LAB6 monitor: prediction observations, HTTP health and latency."""
from __future__ import annotations

import io
import json
import threading
import time
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
from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: float = Field(ge=-20, le=20, allow_inf_nan=False)
    x2: float = Field(ge=-20, le=20, allow_inf_nan=False)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class ServiceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    status_code: int = Field(ge=100, le=599)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    model_version: str
    model_uptime_s: float = Field(ge=0, allow_inf_nan=False)


class MonitorHealthResponse(BaseModel):
    status: str
    service: str


class IngestResponse(BaseModel):
    accepted: int


class MonitorMetricsResponse(BaseModel):
    count: int
    x1_mean: float | None
    x2_mean: float | None
    probability_mean: float | None
    metric_descriptions: dict[str, str] = Field(default_factory=dict)
    service_request_count: int = 0
    service_error_count: int = 0
    service_error_rate: float = 0.0
    service_p95_latency_ms: float = 0.0
    service_avg_latency_ms: float = 0.0
    model_last_seen_age_s: float | None = None
    model_uptime_s: float | None = None
    model_version: str | None = None


@dataclass
class MonitorState:
    maxlen: int = 5000
    log_path: Path | None = None
    data: deque = field(default_factory=lambda: deque(maxlen=5000))
    service_data: deque = field(default_factory=lambda: deque(maxlen=5000))
    service_last_seen: float | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def _write_log(self, entry: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add(self, obs: Observation) -> None:
        with self.lock:
            self.data.append((obs.x1, obs.x2, obs.probability))
            self._write_log({"time": datetime.now(timezone.utc).isoformat(),
                             "kind": "prediction", **obs.model_dump()})

    def add_service(self, obs: ServiceObservation) -> None:
        with self.lock:
            self.service_data.append(obs)
            self.service_last_seen = time.time()
            self._write_log({"time": datetime.now(timezone.utc).isoformat(),
                             "kind": "http", **obs.model_dump()})

    def metrics(self) -> dict:
        with self.lock:
            if self.data:
                values = np.asarray(list(self.data), dtype=float)
                summary = {"count": len(values), "x1_mean": float(values[:, 0].mean()),
                           "x2_mean": float(values[:, 1].mean()),
                           "probability_mean": float(values[:, 2].mean())}
            else:
                summary = {"count": 0, "x1_mean": None, "x2_mean": None, "probability_mean": None}
            events = list(self.service_data)
            latencies = [item.latency_ms for item in events]
            errors = sum(item.status_code >= 400 for item in events)
            summary.update({
                "service_request_count": len(events),
                "service_error_count": errors,
                "service_error_rate": errors / max(len(events), 1),
                "service_p95_latency_ms": float(np.quantile(latencies, .95)) if latencies else 0.0,
                "service_avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "model_last_seen_age_s": time.time() - self.service_last_seen if self.service_last_seen else None,
                "model_uptime_s": events[-1].model_uptime_s if events else None,
                "model_version": events[-1].model_version if events else None,
                "metric_descriptions": {
                    "x1_mean/x2_mean": "средние координаты наблюдений",
                    "probability_mean": "средняя вероятность класса 1",
                    "service_error_rate": "доля HTTP-статусов >= 400",
                    "service_p95_latency_ms": "p95 задержки модели по входящим событиям",
                    "model_last_seen_age_s": "секунд с последнего наблюдения сервиса",
                },
            })
            return summary

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
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.plot_png())


def create_monitor_app(state: MonitorState | None = None) -> FastAPI:
    state = state if state is not None else MonitorState()
    app = FastAPI(title="make_moons independent monitoring service")
    app.state.monitor_state = state

    @app.get("/health", response_model=MonitorHealthResponse)
    def health() -> dict:
        return {"status": "ok", "service": "monitor"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(obs: Observation) -> dict:
        state.add(obs)
        return {"accepted": 1}

    @app.post("/ingest/service", response_model=IngestResponse)
    def ingest_service(obs: ServiceObservation) -> dict:
        state.add_service(obs)
        return {"accepted": 1}

    @app.get("/metrics", response_model=MonitorMetricsResponse)
    def metrics() -> dict:
        return state.metrics()

    @app.get("/plot.png", responses={200: {"content": {"image/png": {}}}})
    def plot() -> Response:
        return Response(content=state.plot_png(), media_type="image/png")

    return app
