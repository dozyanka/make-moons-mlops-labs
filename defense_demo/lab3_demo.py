from __future__ import annotations

import argparse
import inspect

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import moons_lab.dl_pipeline as dl_pipeline
from common import (
    MODELS,
    ROOT,
    binary_metrics,
    find_csv,
    load_xy,
    print_compare,
    read_metric_row,
    save_result,
)


def checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dictionary")

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "weights",
    ]:
        value = checkpoint.get(key)
        if isinstance(value, dict) and any(
            torch.is_tensor(v) for v in value.values()
        ):
            return value

    tensor_items = {
        key: value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }
    if tensor_items:
        return tensor_items

    raise KeyError("Could not find model state_dict in checkpoint")


def clean_state_keys(state):
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ["module.", "model."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def project_model_from_module(state):
    candidates = []
    for name, obj in inspect.getmembers(dl_pipeline, inspect.isclass):
        if obj.__module__ != dl_pipeline.__name__:
            continue
        try:
            if not issubclass(obj, nn.Module):
                continue
        except TypeError:
            continue

        try:
            model = obj()
        except Exception:
            continue

        try:
            model.load_state_dict(state, strict=True)
            return model, f"{dl_pipeline.__name__}.{name}"
        except Exception:
            candidates.append((name, model))

    return None, ""


def generic_model(activation):
    return nn.Sequential(
        nn.Linear(2, 32),
        activation(),
        nn.Linear(32, 32),
        activation(),
        nn.Linear(32, 1),
    )


def assign_by_shape(model, state):
    weights = [
        value.detach().cpu()
        for value in state.values()
        if torch.is_tensor(value) and value.ndim == 2
    ]
    biases = [
        value.detach().cpu()
        for value in state.values()
        if torch.is_tensor(value) and value.ndim == 1
    ]

    layers = [
        layer for layer in model
        if isinstance(layer, nn.Linear)
    ]

    if len(weights) < 3 or len(biases) < 3:
        raise ValueError("Not enough linear tensors in checkpoint")

    for layer, weight, bias in zip(layers, weights, biases):
        if tuple(layer.weight.shape) != tuple(weight.shape):
            raise ValueError(
                f"Weight shape mismatch: "
                f"{layer.weight.shape} vs {weight.shape}"
            )
        if tuple(layer.bias.shape) != tuple(bias.shape):
            raise ValueError(
                f"Bias shape mismatch: "
                f"{layer.bias.shape} vs {bias.shape}"
            )
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)

    return model


def checkpoint_scaler(checkpoint, x_train):
    key_pairs = [
        ("scaler_mean", "scaler_scale"),
        ("mean", "scale"),
        ("feature_mean", "feature_scale"),
        ("x_mean", "x_scale"),
    ]

    if isinstance(checkpoint, dict):
        for mean_key, scale_key in key_pairs:
            if mean_key in checkpoint and scale_key in checkpoint:
                mean = np.asarray(checkpoint[mean_key], dtype=float)
                scale = np.asarray(checkpoint[scale_key], dtype=float)
                if mean.shape == (2,) and scale.shape == (2,):
                    return mean, scale, f"checkpoint:{mean_key}/{scale_key}"

    scaler = StandardScaler().fit(x_train)
    return scaler.mean_, scaler.scale_, "recomputed from train split"


def infer(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        logits = logits.reshape(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def evaluate_candidate(model, x_test, y_test, expected):
    pred, prob = infer(model, x_test)
    actual = binary_metrics(y_test, pred, prob)
    score = sum(
        abs(float(actual[key]) - float(expected[key]))
        for key in ["accuracy", "f1", "roc_auc"]
    )
    return score, actual, pred, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "streaming"],
        default="full",
    )
    args = parser.parse_args()

    report_mode = "full" if args.mode == "full" else "chunks"
    model_name = (
        "lab3_full.pt"
        if args.mode == "full"
        else "lab3_streaming.pt"
    )

    row = read_metric_row(
        ROOT / "reports" / "LAB3" / "dl_metrics.csv",
        report_mode,
    )
    expected = {
        "accuracy": float(row["accuracy"]),
        "f1": float(row["f1"]),
        "roc_auc": float(row["roc_auc"]),
    }

    train_path = find_csv("train")
    test_path = find_csv("test")
    x_train, _ = load_xy(train_path)
    x_test, y_test = load_xy(test_path)

    checkpoint = torch.load(
        MODELS / model_name,
        map_location="cpu",
        weights_only=False,
    )
    state = clean_state_keys(checkpoint_state(checkpoint))

    mean, scale, scaler_source = checkpoint_scaler(
        checkpoint,
        x_train,
    )
    x_test_scaled = (x_test - mean) / scale

    model, model_source = project_model_from_module(state)

    if model is not None:
        score, actual, _, _ = evaluate_candidate(
            model,
            x_test_scaled,
            y_test,
            expected,
        )
    else:
        options = [
            ("ReLU", nn.ReLU),
            ("Tanh", nn.Tanh),
            ("GELU", nn.GELU),
            ("SiLU", nn.SiLU),
        ]

        best = None
        for name, activation in options:
            candidate = assign_by_shape(
                generic_model(activation),
                state,
            )
            result = evaluate_candidate(
                candidate,
                x_test_scaled,
                y_test,
                expected,
            )
            item = (result[0], name, candidate, result[1])
            if best is None or item[0] < best[0]:
                best = item

        _, activation_name, model, actual = best
        model_source = f"generic MLP 2-32-32-1 / {activation_name}"

    passed = print_compare(
        f"LAB 3 LIVE MODEL — {args.mode}",
        actual,
        expected,
        tolerance=5e-7,
    )

    print("checkpoint:", model_name)
    print("model loader:", model_source)
    print("scaler:", scaler_source)
    print("test data:", test_path.relative_to(ROOT))
    print("objects:", len(y_test))

    save_result(
        f"lab3_{args.mode}_result.json",
        {
            "lab": 3,
            "mode": args.mode,
            "checkpoint": model_name,
            "model_loader": model_source,
            "scaler": scaler_source,
            "actual": actual,
            "report": expected,
            "pass": passed,
        },
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
