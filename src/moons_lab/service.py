from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from joblib import Parallel, delayed
from pydantic import BaseModel, Field

from .registry import ModelRegistry

LOGGER = logging.getLogger("moons_lab.service")


class PredictRequest(BaseModel):
    x1: float = Field(ge=-20, le=20)
    x2: float = Field(ge=-20, le=20)
    y_true: int | None = Field(default=None, ge=0, le=1)


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1, max_length=100_000)
    n_jobs: int = Field(default=1, ge=1, le=256)


class VersionRequest(BaseModel):
    version: str
    reason: str = "API request"


class RuntimePatch(BaseModel):
    metrics_enabled: bool | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None


@dataclass
class ServiceState:
    started_at: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    prediction_count: int = 0
    batch_fallbacks: int = 0
    latency_sum_s: float = 0.0
    correct_with_label: int = 0
    labeled_count: int = 0
    metrics_enabled: bool = True
    log_level: str = "INFO"
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record(self, elapsed: float, *, error: bool = False, predictions: int = 0, correct: int = 0, labeled: int = 0) -> None:
        with self.lock:
            self.request_count += 1
            self.latency_sum_s += elapsed
            self.error_count += int(error)
            self.prediction_count += predictions
            self.correct_with_label += correct
            self.labeled_count += labeled

    def metrics(self, active_version: str) -> dict:
        with self.lock:
            avg_ms = 1000 * self.latency_sum_s / max(self.request_count, 1)
            return {
                "uptime_s": time.time() - self.started_at,
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "avg_request_latency_ms": avg_ms,
                "online_accuracy_if_labeled": self.correct_with_label / max(self.labeled_count, 1),
                "labeled_count": self.labeled_count,
                "batch_fallbacks": self.batch_fallbacks,
                "active_version": active_version,
                "metrics_enabled": self.metrics_enabled,
                "metric_descriptions": {
                    "uptime_s": "время работы процесса сервиса",
                    "error_rate": "доля запросов, завершившихся ошибкой",
                    "avg_request_latency_ms": "средняя задержка API-запроса",
                    "online_accuracy_if_labeled": "accuracy только по запросам, где передан y_true",
                },
            }


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    raw = np.asarray(raw)
    if raw.ndim == 2:
        return raw[:, 1]
    return raw.astype(float)


def create_app(registry_path: str | Path, *, state: ServiceState | None = None) -> FastAPI:
    registry = ModelRegistry(registry_path)
    state = state or ServiceState()
    app = FastAPI(title="make_moons model service")
    app.state.registry = registry
    app.state.metrics_state = state

    def one_prediction(item: PredictRequest, version: str | None = None) -> dict:
        selected, model = registry.get_model(version)
        x = np.array([[item.x1, item.x2]], dtype=np.float64)
        p = float(_positive_probability(model, x)[0])
        pred = int(p >= 0.5)
        return {"prediction": pred, "probability": p, "model_version": selected}

    def block_predictions(items: list[PredictRequest], version: str | None = None) -> list[dict]:
        if not items:
            return []
        selected, model = registry.get_model(version)
        x = np.asarray([[item.x1, item.x2] for item in items], dtype=np.float64)
        probabilities = _positive_probability(model, x)
        return [
            {"prediction": int(float(p) >= 0.5), "probability": float(p), "model_version": selected}
            for p in probabilities
        ]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "active_version": registry.active_version}

    @app.post("/predict")
    def predict(item: PredictRequest) -> dict:
        start = time.perf_counter()
        try:
            result = one_prediction(item)
            correct = int(item.y_true is not None and result["prediction"] == item.y_true)
            labeled = int(item.y_true is not None)
            state.record(time.perf_counter() - start, predictions=1, correct=correct, labeled=labeled)
            return result
        except Exception as exc:
            state.record(time.perf_counter() - start, error=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/predict/batch")
    def predict_batch(request: BatchPredictRequest) -> dict:
        start = time.perf_counter()
        try:
            n_jobs = min(request.n_jobs, max(1, os.cpu_count() or 1), max(1, len(request.items)))
            try:
                if n_jobs == 1:
                    results = block_predictions(request.items)
                else:
                    # Параллелим не по одной строке, а крупными блоками: модель вызывается
                    # векторизованно внутри каждого worker, поэтому overhead joblib существенно ниже.
                    block_size = (len(request.items) + n_jobs - 1) // n_jobs
                    blocks = [
                        request.items[start : start + block_size]
                        for start in range(0, len(request.items), block_size)
                    ]
                    nested = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(block_predictions)(block) for block in blocks
                    )
                    results = [item for block in nested for item in block]
            except Exception:
                with state.lock:
                    state.batch_fallbacks += 1
                results = block_predictions(request.items)
            correct = sum(
                int(item.y_true is not None and result["prediction"] == item.y_true)
                for item, result in zip(request.items, results)
            )
            labeled = sum(int(item.y_true is not None) for item in request.items)
            state.record(
                time.perf_counter() - start,
                predictions=len(results),
                correct=correct,
                labeled=labeled,
            )
            return {"count": len(results), "n_jobs": n_jobs, "results": results}
        except Exception as exc:
            state.record(time.perf_counter() - start, error=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/metrics")
    def metrics() -> dict:
        return state.metrics(registry.active_version)

    @app.get("/models")
    def models() -> dict:
        return registry.snapshot()

    @app.post("/promote")
    def promote(request: VersionRequest) -> dict:
        try:
            return registry.promote(request.version, request.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown model version: {request.version}") from exc

    @app.post("/rollback")
    def rollback() -> dict:
        return registry.rollback("API rollback")

    @app.get("/runtime")
    def runtime() -> dict:
        return {"metrics_enabled": state.metrics_enabled, "log_level": state.log_level}

    @app.patch("/runtime")
    def runtime_patch(patch: RuntimePatch) -> dict:
        with state.lock:
            if patch.metrics_enabled is not None:
                state.metrics_enabled = patch.metrics_enabled
            if patch.log_level is not None:
                state.log_level = patch.log_level
                LOGGER.setLevel(getattr(logging, patch.log_level))
        return {"metrics_enabled": state.metrics_enabled, "log_level": state.log_level}

    return app
