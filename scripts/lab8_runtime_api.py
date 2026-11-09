"""Local-only LAB8 live HTTP servers, with real killable process pool for fault C2."""
from __future__ import annotations

import argparse
import concurrent.futures
import math
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
from moons_lab.lab8_model import DLJoblibAdapter

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
        # Replace only LAB8 serving route. The original LAB6 endpoint is kept as
        # an in-memory callable for ML versions; LAB1-7 source remains untouched.
        original_route = next(route for route in app.router.routes if isinstance(route, APIRoute)
                              and route.path == "/predict/batch")
        original_batch = original_route.endpoint
        parallel_limit = max(1, os.cpu_count() or 1)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=parallel_limit,
                                                     thread_name_prefix="lab8-dl")
        lock = threading.Lock()
        stats = {"requests": 0, "parallel_batches": 0,
                 "max_workers_observed": 0, "last_used_workers": 0,
                 "last_requested_workers": 0, "last_version": None,
                 "backend": "persistent_thread_pool"}
        app.router.routes = [r for r in app.router.routes if r is not original_route]
        app.openapi_schema = None
        # Wrap the application's existing ASGI lifespan. Starlette 1.x removed
        # add_event_handler(); keep both the original lifespan and pool cleanup.
        from contextlib import asynccontextmanager
        previous_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_with_lab8_pool(application):
            try:
                async with previous_lifespan(application) as state:
                    yield state
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

        app.router.lifespan_context = lifespan_with_lab8_pool

        def _work(model: DLJoblibAdapter, array: np.ndarray):
            # The adapter is immutable once warmed up; PyTorch inference_mode
            # uses per-thread state, so concurrent forward passes are safe.
            return threading.get_ident(), model.predict_proba(array)

        @app.post("/predict/batch", response_model=BatchPredictResponse)
        def optimized_batch(request: BatchPredictRequest):
            version, model = model_registry.get_model()  # pinned SHA-verified model
            if not isinstance(model, DLJoblibAdapter):
                return original_batch(request)
            model.warmup()  # completes lazy load outside parallel workers
            x = np.asarray([[v.x1, v.x2] for v in request.items], dtype=np.float32)
            workers = min(request.n_jobs, parallel_limit, len(x))
            requested = request.n_jobs
            observed = set()
            try:
                if workers == 1:
                    probabilities = model.predict_proba(x)
                else:
                    # Stable executor: joblib.Parallel used to rebuild the
                    # scheduler for every tiny HTTP request (~20ms overhead).
                    block = math.ceil(len(x) / workers)
                    futures = [pool.submit(_work, model, x[i:i + block])
                               for i in range(0, len(x), block)]
                    results = [f.result(timeout=30) for f in futures]
                    observed = {tid for tid, _ in results}
                    probabilities = np.concatenate([arr for _, arr in results])
            except Exception:
                # A failed worker does not lose a prediction; it must be counted.
                state = app.state.metrics_state
                state.record_fallback()
                probabilities = model.predict_proba(x)
                workers = 1
                observed.clear()
            rows = [{"prediction": int(float(p) >= .5),
                     "probability": float(p), "model_version": version}
                    for p in probabilities]
            labeled = sum(int(v.y_true is not None) for v in request.items)
            correct = sum(int(v.y_true is not None and v.y_true == row["prediction"])
                          for v, row in zip(request.items, rows))
            app.state.metrics_state.record_predictions(len(rows), correct, labeled)
            with lock:
                stats["requests"] += 1
                stats["parallel_batches"] += int(workers > 1)
                stats["last_requested_workers"] = requested
                stats["last_used_workers"] = len(observed) if workers > 1 else 1
                stats["max_workers_observed"] = max(stats["max_workers_observed"],
                                                     stats["last_used_workers"])
                stats["last_version"] = version
            return {"count": len(rows), "n_jobs": workers, "results": rows}

        @app.get("/lab8/parallel-stats")
        def parallel_stats() -> dict:
            with lock:
                return dict(stats, available_logical_cores=parallel_limit)

        @app.get("/lab8/parallel-proof")
        def parallel_proof(workers: int = 0) -> dict:
            """Demonstrate simultaneous independent executor threads, not CPU saturation.

            No claim that every core was busy for the duration of a stage;
            actual threads used by ordinary inference are recorded separately.
            """
            jobs = parallel_limit if workers == 0 else workers
            if jobs < 1 or jobs > parallel_limit:
                raise HTTPException(status_code=422, detail="workers must be in [1, logical cores]")
            barrier = threading.Barrier(jobs, timeout=15)

            def probe() -> int:
                barrier.wait(timeout=15)
                return threading.get_ident()

            futures = [pool.submit(probe) for _ in range(jobs)]
            seen = {future.result(timeout=20) for future in futures}
            return {"requested_workers": jobs, "simultaneous_threads": len(seen),
                    "all_workers_reachable": len(seen) == jobs}

        @app.post("/lab8/warmup")
        def warmup() -> dict:
            selected, model = model_registry.get_model()
            model.predict_proba(np.zeros((4, 2), dtype=np.float32))
            return {"status": "ready", "version": selected}

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
        nonlocal pool
        # If a worker exits unexpectedly during spawn, report failure promptly.
        from concurrent.futures.process import BrokenProcessPool
        with lock:
            try:
                futures = [pool.submit(os.getpid) for _ in range(process_workers * 3)]
                pids = sorted({future.result(timeout=8) for future in futures})
            except (BrokenProcessPool, concurrent.futures.TimeoutError, OSError) as exc:
                pool.shutdown(wait=False, cancel_futures=True)
                pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=process_workers, mp_context=multiprocessing.get_context("spawn")
                )
                raise HTTPException(status_code=503, detail="process worker discovery failed; pool reset") from exc
        return {"pids": pids, "workers_configured": process_workers}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--service", choices=["model", "monitor"], required=True)
    p.add_argument("--registry")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--process-workers", type=int, default=0)
    p.add_argument("--monitor-log")
    p.add_argument("--unmodified-lab6", action="store_true",
                   help="Measurement reference: serve the original LAB6 batch route without LAB8 optimization")
    args = p.parse_args()
    if args.service == "model":
        if not args.registry:
            p.error("model needs --registry")
        app = create_app(Path(args.registry)) if args.unmodified_lab6 else make_app(Path(args.registry), args.process_workers)
    else:
        app = create_monitor_app(MonitorState(log_path=Path(args.monitor_log) if args.monitor_log else None))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
