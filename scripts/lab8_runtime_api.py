"""Local-only LAB8 live HTTP servers, with real killable process pool for fault C2."""
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import HTTPException
from fastapi.routing import APIRoute

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.monitor_service import MonitorState, create_monitor_app
from moons_lab.service import BatchPredictRequest, BatchPredictResponse, create_app

_WORKER_CACHE = {}


def _process_block(model_path: str, items: list[dict], version: str) -> list[dict]:
    model = _WORKER_CACHE.get(model_path)
    if model is None:
        model = joblib.load(model_path)
        _WORKER_CACHE[model_path] = model
    x = np.asarray([[v["x1"], v["x2"]] for v in items], dtype=float)
    raw = np.asarray(model.predict_proba(x))
    proba = raw[:, 1] if raw.ndim == 2 else raw.ravel()
    return [{"prediction": int(float(p) >= 0.5), "probability": float(p), "model_version": version}
            for p in proba]


def make_app(registry: Path, process_workers: int = 0):
    app = create_app(registry)
    model_registry = app.state.registry

    from functools import lru_cache
    from moons_lab.validator import load_validator

    @lru_cache(maxsize=1)
    def validation_assets():
        return (joblib.load(ROOT / "reports/LAB2/ml_model.joblib"),
                load_validator(ROOT / "reports/LAB4/validator_params.npz"))

    @app.post("/lab8/predict/validated")
    def predict_validated(request: BatchPredictRequest) -> dict:
        """Explicit LAB4 gate: foreign inputs never reach production prediction silently.

        Rejected rows carry reasons, NOT model predictions; guard probabilities
        are retained solely for drift-monitoring statistics.
        """
        ml_model, validator = validation_assets()
        x = np.asarray([[row.x1, row.x2] for row in request.items], dtype=float)
        raw = np.asarray(ml_model.predict_proba(x))
        p = raw[:, 1] if raw.ndim == 2 else raw.ravel()
        allowed = validator.accept(x, p)
        current_version, model = model_registry.get_model()
        accepted_idx = np.flatnonzero(allowed)
        response_rows = [{"rejected": True, "reason": "OOD/confidence validator: LAB4",
                          "model_version": current_version} for _ in request.items]
        if len(accepted_idx):
            accepted_proba = np.asarray(model.predict_proba(x[accepted_idx]))
            if accepted_proba.ndim == 2:
                accepted_proba = accepted_proba[:, 1]
            for idx, prob in zip(accepted_idx, accepted_proba):
                response_rows[int(idx)] = {
                    "rejected": False, "prediction": int(float(prob) >= .5),
                    "probability": float(prob), "model_version": current_version,
                    "validation": "passed",
                }
        return {"count": len(request.items), "accepted_count": int(allowed.sum()),
                "rejected_count": int((~allowed).sum()),
                "silent_accepts": 0,
                "monitor_probabilities": [float(v) for v in p],
                "results": response_rows}

    if not process_workers:
        return app
    if process_workers > (os.cpu_count() or 1):
        raise ValueError("worker count exceeds logical cores")
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=process_workers, mp_context=multiprocessing.get_context("spawn"))
    state = app.state.metrics_state
    reg = app.state.registry
    lock = threading.RLock()
    # Override batch path ONLY in LAB8 fault-mode server; LAB6/LAB7 original source is untouched.
    app.router.routes = [route for route in app.router.routes if not (
        isinstance(route, APIRoute) and route.path == "/predict/batch")]
    app.openapi_schema = None

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def batch(request: BatchPredictRequest):
        nonlocal pool
        version = reg.active_version
        modelpath = reg._model_path(reg.snapshot()["versions"][version])
        reg.get_model(version)  # verify SHA-256 before submitting
        n = min(len(request.items), request.n_jobs, process_workers)
        blocks = [[v.model_dump() for v in request.items[i::n]] for i in range(n)]
        try:
            with lock:
                futures = [pool.submit(_process_block, str(modelpath), part, version) for part in blocks]
                nested = [f.result(timeout=30) for f in futures]
            # Restore original order of strided blocks.
            results = [None] * len(request.items)
            for i, block in enumerate(nested):
                for j, pred in enumerate(block):
                    results[i + j * n] = pred
        except Exception:
            state.record_fallback()
            with lock:
                pool.shutdown(wait=False, cancel_futures=True)
                pool = concurrent.futures.ProcessPoolExecutor(max_workers=process_workers, mp_context=multiprocessing.get_context("spawn"))
            _, model = reg.get_model(version)
            p = np.asarray(model.predict_proba(np.asarray(
                [[v.x1, v.x2] for v in request.items], dtype=float)))
            if p.ndim == 2:
                p = p[:, 1]
            results = [{"prediction": int(float(v) >= .5), "probability": float(v), "model_version": version}
                       for v in p.ravel()]
        state.record_predictions(len(results), 0, 0)
        return {"count": len(results), "n_jobs": n, "results": results}

    @app.get("/lab8/worker-pids")
    def worker_pids() -> dict:
        with lock:
            futures = [pool.submit(os.getpid) for _ in range(process_workers * 3)]
            pids = sorted({future.result(timeout=30) for future in futures})
        return {"pids": pids, "workers_configured": process_workers}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--service", choices=["model", "monitor"], required=True)
    p.add_argument("--registry")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--process-workers", type=int, default=0)
    p.add_argument("--monitor-log")
    args = p.parse_args()
    if args.service == "model":
        if not args.registry:
            p.error("model needs --registry")
        app = make_app(Path(args.registry), args.process_workers)
    else:
        app = create_monitor_app(MonitorState(log_path=Path(args.monitor_log) if args.monitor_log else None))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
