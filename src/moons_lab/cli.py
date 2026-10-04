from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from sklearn.model_selection import train_test_split

from .config import load_config
from .data import load_xy
from .dl_pipeline import save_checkpoint, train_full, train_stream
from .generator import GeneratorSettings, generate_dataframe
from .ml_pipeline import save_model, train_rbf_chunks, train_rbf_full
from .service import create_app
from .validator import calibrate_threshold, fit_chunks, fit_full, save_validator


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None)
    return parser


def _ensure_synthetic_splits(seed: int = 4242, chunk_size: int = 700) -> dict[str, str]:
    """Materialize a deterministic LAB5-backed source for CLI end-to-end runs."""
    root = Path("data/runtime_synthetic")
    chunks = root / "chunks"
    root.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(parents=True, exist_ok=True)

    df = generate_dataframe(
        GeneratorSettings(mode="similar", n_samples=24000, seed=seed, noise=0.22)
    )
    train_val, test = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["y"])
    train, validation = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=train_val["y"],
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_path = root / "train.csv"
    val_path = root / "validation.csv"
    test_path = root / "test.csv"
    train.to_csv(train_path, index=False, float_format="%.10f")
    validation.to_csv(val_path, index=False, float_format="%.10f")
    test.to_csv(test_path, index=False, float_format="%.10f")
    for path in chunks.glob("train_chunk_*.csv"):
        path.unlink()
    for idx, start in enumerate(range(0, len(train), chunk_size)):
        train.iloc[start : start + chunk_size].to_csv(
            chunks / f"train_chunk_{idx:03d}.csv", index=False, float_format="%.10f"
        )
    return {
        "train": str(train_path),
        "validation": str(val_path),
        "test": str(test_path),
        "chunks": str(chunks),
    }


def _source_paths(data_source: str, chunk_size: int = 700) -> dict[str, str]:
    if data_source == "real":
        return {
            "train": "data/processed/train.csv",
            "validation": "data/processed/validation.csv",
            "test": "data/processed/test.csv",
            "chunks": "data/chunks",
        }
    if data_source == "synthetic":
        return _ensure_synthetic_splits(chunk_size=chunk_size)
    raise ValueError(f"unknown data source: {data_source}")


def train_ml_entry() -> None:
    parser = _common_parser("Train classical RBF-SGD model")
    parser.add_argument("--source", choices=["full", "chunks"], default="full")
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--output", default="reports/LAB2/ml_model.joblib")
    args = parser.parse_args()
    cfg = load_config(2, args.config)
    paths = _source_paths(args.data_source, cfg.chunk_size)
    gamma = cfg.gamma_candidates[1]
    if args.source == "full":
        model = train_rbf_full(
            paths["train"],
            chunk_size=cfg.chunk_size,
            gamma=gamma,
            n_components=cfg.rbf_components,
            alpha=cfg.alpha,
            passes=cfg.passes,
            seed=cfg.seed,
        )
    else:
        model = train_rbf_chunks(
            paths["chunks"],
            gamma=gamma,
            n_components=cfg.rbf_components,
            alpha=cfg.alpha,
            passes=cfg.passes,
            seed=cfg.seed,
        )
    save_model(model, args.output)


def train_dl_entry() -> None:
    parser = _common_parser("Train PyTorch MLP")
    parser.add_argument("--source", choices=["full", "chunks"], default="full")
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--output", default="reports/LAB3/dl_model.pt")
    args = parser.parse_args()
    cfg = load_config(3, args.config)
    # LAB3 does not carry chunk size, so use the shared LAB1 source contract.
    paths = _source_paths(args.data_source, 700)
    common = dict(
        validation_path=paths["validation"],
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=cfg.device,
    )
    if args.source == "full":
        result = train_full(paths["train"], **common)
    else:
        result = train_stream(paths["chunks"], **common)
    save_checkpoint(result, args.output)


def train_validator_entry() -> None:
    parser = _common_parser("Train applicability validator")
    parser.add_argument("--source", choices=["full", "chunks"], default="full")
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--model", default="reports/LAB2/ml_model.joblib")
    parser.add_argument("--output", default="reports/LAB4/validator_params.npz")
    args = parser.parse_args()
    cfg = load_config(4, args.config)
    paths = _source_paths(args.data_source, cfg.chunk_size)
    validator = (
        fit_full(paths["train"], cfg.confidence_threshold)
        if args.source == "full"
        else fit_chunks(paths["chunks"], cfg.confidence_threshold)
    )
    model = joblib.load(args.model)
    x_val, _ = load_xy(paths["validation"])
    proba = np.asarray(model.predict_proba(x_val))
    if proba.ndim == 2:
        proba = proba[:, 1]
    calibrate_threshold(validator, x_val, proba, cfg.target_false_reject_rate)
    save_validator(validator, args.output)


def generate_entry() -> None:
    parser = _common_parser("Generate controlled synthetic data")
    parser.add_argument("--mode", choices=["similar", "random"], default="similar")
    parser.add_argument("--output", default="data/synthetic/generated.csv")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(5, args.config)
    settings = GeneratorSettings(
        mode=args.mode,
        n_samples=args.n or cfg.n_samples,
        seed=cfg.seed if args.seed is None else args.seed,
        noise=cfg.similar_noise,
        random_range=cfg.random_range,
    )
    df = generate_dataframe(settings)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.10f")


def serve_entry() -> None:
    parser = _common_parser("Serve model through FastAPI")
    args = parser.parse_args()
    cfg = load_config(6, args.config)
    uvicorn.run(create_app(cfg.registry_path), host=cfg.host, port=cfg.port)


def evaluate_entry() -> None:
    parser = _common_parser("Evaluate a joblib model on a data source")
    parser.add_argument("--model", default="reports/LAB2/ml_model.joblib")
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    from .metrics import classification_metrics

    paths = _source_paths(args.data_source, 700)
    data_path = args.data or paths["test"]
    model = joblib.load(args.model)
    x, y = load_xy(data_path)
    p = np.asarray(model.predict_proba(x))
    if p.ndim == 2:
        p = p[:, 1]
    print(classification_metrics(y, p))


def monitor_entry() -> None:
    parser = _common_parser("Run drift summary on real or LAB5 synthetic data")
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--current", default=None)
    parser.add_argument("--model", default="reports/LAB2/ml_model.joblib")
    args = parser.parse_args()
    from .drift import compute_drift

    if args.data_source == "real":
        reference_path = args.reference or "data/processed/train.csv"
        current_path = args.current or "data/processed/test.csv"
    else:
        paths = _source_paths("synthetic", 700)
        reference_path = args.reference or paths["train"]
        if args.current:
            current_path = args.current
        else:
            drifted = generate_dataframe(
                GeneratorSettings(mode="similar", n_samples=3600, seed=989, noise=0.22, shift_x=0.9, shift_y=0.35)
            )
            current_path = "data/runtime_synthetic/drifted_monitor.csv"
            drifted.to_csv(current_path, index=False, float_format="%.10f")

    model = joblib.load(args.model)
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)
    xr = ref[["x1", "x2"]].to_numpy()
    xc = cur[["x1", "x2"]].to_numpy()
    pr = np.asarray(model.predict_proba(xr))
    pc = np.asarray(model.predict_proba(xc))
    if pr.ndim == 2:
        pr = pr[:, 1]
        pc = pc[:, 1]
    print(compute_drift(xr, xc, pr, pc))
