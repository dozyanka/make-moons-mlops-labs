"""Versioned FastAPI inference with strict contracts and correct HTTP metrics."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .registry import ModelRegistry, ModelStatus

LOGGER = logging.getLogger("moons_lab.service")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: float = Field(ge=-20, le=20, allow_inf_nan=False)
    x2: float = Field(ge=-20, le=20, allow_inf_nan=False)
    y_true: int | None = Field(default=None, ge=0, le=1)

    @field_validator("x1", "x2", mode="before")
    @classmethod
    def numeric_only(cls, value):
        if isinstance(value, (bool, str)):
            raise ValueError("features must be finite numeric values")
        return value


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[PredictRequest] = Field(min_length=1, max_length=100_000)
    n_jobs: int = Field(default=1, ge=1, le=256)


class VersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1)
    reason: str = "API request"


class VersionStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["preparing", "canary", "retired"]
    reason: str = "API stage"


class RuntimePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics_enabled: bool | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None


class PredictResponse(BaseModel):
    prediction: int
    probability: float = Field(ge=0, le=1)
    model_version: str


class BatchPredictResponse(BaseModel):
    count: int
    n_jobs: int
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    active_version: str


class VersionInfo(BaseModel):
    path: str
    sha256: str
    metrics: dict[str, float]
    status: ModelStatus
    description: str = ""


class RegistryResponse(BaseModel):
    active: str
    versions: dict[str, VersionInfo]
    history: list[dict[str, Any]]


class RuntimeResponse(BaseModel):
    metrics_enabled: bool
    log_level: str


class MetricsResponse(BaseModel):
    uptime_s: float
    request_count: int
    prediction_count: int
    error_count: int
    error_rate: float
    avg_request_latency_ms: float
    p95_request_latency_ms: float
    online_accuracy_if_labeled: float
    labeled_count: int
    batch_fallbacks: int
    active_version: str
    metrics_enabled: bool
    metric_descriptions: dict[str, str]


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
    latency_history_ms: deque = field(default_factory=lambda: deque(maxlen=20000))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record_http(self, elapsed: float, status_code: int, *, enabled_at_start: bool) -> None:
        """Count all HTTP responses, including Pydantic 422 and application errors."""
        with self.lock:
            if not enabled_at_start:
                return
            self.request_count += 1
            self.latency_sum_s += elapsed
            self.latency_history_ms.append(elapsed * 1000)
            self.error_count += int(status_code >= 400)

    def record_predictions(self, predictions: int, correct: int, labeled: int) -> None:
        with self.lock:
            if not self.metrics_enabled:
                return
            self.prediction_count += predictions
            self.correct_with_label += correct
            self.labeled_count += labeled

    def record_fallback(self) -> None:
        with self.lock:
            if self.metrics_enabled:
                self.batch_fallbacks += 1

    def metrics(self, active_version: str) -> dict:
        with self.lock:
            avg_ms = 1000 * self.latency_sum_s / max(self.request_count, 1)
            p95_ms = float(np.quantile(list(self.latency_history_ms), .95)) if self.latency_history_ms else 0.0
            return {
                "uptime_s": time.time() - self.started_at,
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "avg_request_latency_ms": avg_ms,
                "p95_request_latency_ms": p95_ms,
                "online_accuracy_if_labeled": self.correct_with_label / max(self.labeled_count, 1),
                "labeled_count": self.labeled_count,
                "batch_fallbacks": self.batch_fallbacks,
                "active_version": active_version,
                "metrics_enabled": self.metrics_enabled,
                "metric_descriptions": {
                    "uptime_s": "время работы процесса сервиса",
                    "error_rate": "доля HTTP-ответов со статусом >= 400 при включённых метриках",
                    "avg_request_latency_ms": "средняя задержка HTTP-запроса",
                    "online_accuracy_if_labeled": "accuracy только для записей с y_true",
                },
            }


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x))
    return raw[:, 1] if raw.ndim == 2 else raw.astype(float)


def create_app(registry_path: str | Path, *, state: ServiceState | None = None,
               monitor_sink: Callable[[float, float, float], None] | None = None,
               monitor_request_sink: Callable[[str, int, float, str, float], None] | None = None) -> FastAPI:
    registry = ModelRegistry(registry_path)
    state = state if state is not None else ServiceState()
    app = FastAPI(title="make_moons model service")
    app.state.registry = registry
    app.state.metrics_state = state

    @app.middleware("http")
    async def observe_http(request: Request, call_next):
        started = time.perf_counter()
        with state.lock:
            enabled_at_start = state.metrics_enabled
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            state.record_http(elapsed, status_code, enabled_at_start=enabled_at_start)
            if monitor_request_sink is not None:
                # Best effort: monitoring outages may not break prediction traffic.
                from starlette.concurrency import run_in_threadpool
                try:
                    await run_in_threadpool(
                        monitor_request_sink, request.url.path, status_code, elapsed * 1000,
                        registry.active_version, time.time() - state.started_at,
                    )
                except Exception:
                    LOGGER.warning("monitor request sink failed", exc_info=True)

    def one_prediction(item: PredictRequest, version: str | None = None) -> dict:
        selected, model = registry.get_model(version)
        p = float(_positive_probability(model, np.array([[item.x1, item.x2]], dtype=np.float64))[0])
        return {"prediction": int(p >= .5), "probability": p, "model_version": selected}

    def block_predictions(items: list[PredictRequest], version: str | None = None) -> list[dict]:
        if not items:
            return []
        selected, model = registry.get_model(version)
        x = np.asarray([[item.x1, item.x2] for item in items], dtype=np.float64)
        return [{"prediction": int(float(p) >= .5), "probability": float(p), "model_version": selected}
                for p in _positive_probability(model, x)]

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict:
        return {"status": "ok", "active_version": registry.active_version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(item: PredictRequest) -> dict:
        try:
            result = one_prediction(item)
            if monitor_sink is not None:
                try:
                    monitor_sink(item.x1, item.x2, result["probability"])
                except Exception:
                    LOGGER.warning("monitor sink failed; prediction continues", exc_info=True)
            state.record_predictions(1, int(item.y_true is not None and result["prediction"] == item.y_true),
                                     int(item.y_true is not None))
            return result
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail="model unavailable or hash mismatch") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="prediction failed") from exc

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(request: BatchPredictRequest) -> dict:
        try:
            n_jobs = min(request.n_jobs, max(1, os.cpu_count() or 1), len(request.items))
            try:
                if n_jobs == 1:
                    results = block_predictions(request.items)
                else:
                    # Pin a single version for the whole batch across concurrent promotion.
                    pinned = registry.active_version
                    block_size = (len(request.items) + n_jobs - 1) // n_jobs
                    blocks = [request.items[i:i + block_size]
                              for i in range(0, len(request.items), block_size)]
                    nested = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(block_predictions)(block, pinned) for block in blocks
                    )
                    results = [entry for block in nested for entry in block]
            except Exception:
                state.record_fallback()
                results = block_predictions(request.items)
            if monitor_sink is not None:
                for item, result in zip(request.items, results):
                    try:
                        monitor_sink(item.x1, item.x2, result["probability"])
                    except Exception:
                        LOGGER.warning("monitor sink failed; batch continues", exc_info=True)
            correct = sum(int(item.y_true is not None and result["prediction"] == item.y_true)
                          for item, result in zip(request.items, results))
            labeled = sum(int(item.y_true is not None) for item in request.items)
            state.record_predictions(len(results), correct, labeled)
            return {"count": len(results), "n_jobs": n_jobs, "results": results}
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail="model unavailable or hash mismatch") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="prediction failed") from exc

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> dict:
        return state.metrics(registry.active_version)

    @app.get("/models", response_model=RegistryResponse)
    def models() -> dict:
        return registry.snapshot()

    @app.patch("/models/{version}/status", response_model=RegistryResponse)
    def stage(version: str, patch: VersionStageRequest) -> dict:
        try:
            return registry.stage(version, patch.status, patch.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown model version") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/promote", response_model=RegistryResponse)
    def promote(request: VersionRequest) -> dict:
        try:
            return registry.promote(request.version, request.reason)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown model version: {request.version}") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/rollback", response_model=RegistryResponse)
    def rollback() -> dict:
        try:
            return registry.rollback("API rollback")
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail="rollback target unavailable or hash mismatch") from exc

    @app.get("/runtime", response_model=RuntimeResponse)
    def runtime() -> dict:
        return {"metrics_enabled": state.metrics_enabled, "log_level": state.log_level}

    @app.patch("/runtime", response_model=RuntimeResponse)
    def runtime_patch(patch: RuntimePatch) -> dict:
        with state.lock:
            if patch.metrics_enabled is not None:
                state.metrics_enabled = patch.metrics_enabled
            if patch.log_level is not None:
                state.log_level = patch.log_level
                LOGGER.setLevel(getattr(logging, patch.log_level))
            return {"metrics_enabled": state.metrics_enabled, "log_level": state.log_level}

    return app
