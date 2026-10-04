from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab3Config, load_config
from moons_lab.data import load_xy
from moons_lab.dl_pipeline import device_parity, save_checkpoint, train_full, train_stream
from moons_lab.metrics import bootstrap_intervals, classification_metrics, prediction_max_abs_diff
from moons_lab.tracking import tracked_run
from moons_lab.utils import measured_call, sha256_file


def _plot_histories(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, history in histories.items():
        ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation BCE")
    ax.set_title("ЛР3: кривые validation loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/LAB3/figures/loss_curves.png", dpi=150)
    plt.close(fig)


def _config_negative(cfg: Lab3Config) -> None:
    bad = [
        {**cfg.model_dump(), "batch_size": 0},
        {**cfg.model_dump(), "learning_rate": -1.0},
        {**cfg.model_dump(), "device": "tpu"},
    ]
    lines = []
    for i, payload in enumerate(bad, 1):
        try:
            Lab3Config.model_validate(payload)
            lines.append(f"case_{i}: ERROR accepted")
        except ValidationError as exc:
            lines.append(f"case_{i}: REJECTED: {exc.errors()[0]['msg']}")
    (ROOT / "reports/LAB3/config_negative_test.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg: Lab3Config = load_config(3)  # type: ignore[assignment]
    common = dict(
        validation_path=ROOT / "data/processed/validation.csv",
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=cfg.device,
    )
    full_measure = measured_call(train_full, ROOT / "data/processed/train.csv", **common)
    stream_measure = measured_call(train_stream, ROOT / "data/chunks", **common)
    repeat_measure = measured_call(train_full, ROOT / "data/processed/train.csv", **common)
    full = full_measure.value
    stream = stream_measure.value
    repeat = repeat_measure.value
    x_test, y_test = load_xy(ROOT / "data/processed/test.csv")

    outputs = []
    histories = {"full": full.history, "chunks": stream.history}
    pd.DataFrame(full.history).to_csv(ROOT / "reports/LAB3/full_history.csv", index=False)
    pd.DataFrame(stream.history).to_csv(ROOT / "reports/LAB3/chunk_history.csv", index=False)
    _plot_histories(histories)

    save_checkpoint(full, ROOT / "reports/LAB3/dl_model.pt")
    save_checkpoint(stream, ROOT / "reports/LAB3/dl_model_chunk.pt")
    full_p = full.predict_proba(x_test)
    stream_p = stream.predict_proba(x_test)
    repeat_p = repeat.predict_proba(x_test)

    for mode, result, proba, measure, filename in [
        ("full", full, full_p, full_measure, "dl_model.pt"),
        ("chunks", stream, stream_p, stream_measure, "dl_model_chunk.pt"),
    ]:
        metrics = classification_metrics(y_test, proba)
        ci = bootstrap_intervals(y_test, proba, cfg.bootstrap_rounds, cfg.seed)
        with tracked_run(f"lab3_{mode}", experiment="lab3_dl") as run:
            run.log_params({**cfg.model_dump(), "mode": mode, "best_epoch": result.best_epoch})
            run.log_metrics({**metrics, "elapsed_s": measure.elapsed_s, "peak_memory_mb": measure.peak_memory_mb})
            run.log_artifact(ROOT / "reports/LAB3" / filename)
            outputs.append(
                {
                    "mode": mode,
                    **metrics,
                    "accuracy_ci_low": ci["accuracy"][0],
                    "accuracy_ci_high": ci["accuracy"][1],
                    "f1_ci_low": ci["f1"][0],
                    "f1_ci_high": ci["f1"][1],
                    "roc_auc_ci_low": ci["roc_auc"][0],
                    "roc_auc_ci_high": ci["roc_auc"][1],
                    "best_epoch": result.best_epoch,
                    "run_id": run.run_id,
                    "tracking_backend": run.backend,
                }
            )
    pd.DataFrame(outputs).to_csv(ROOT / "reports/LAB3/dl_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"mode": "full", "peak_memory_mb": full_measure.peak_memory_mb, "elapsed_s": full_measure.elapsed_s},
            {"mode": "chunks", "peak_memory_mb": stream_measure.peak_memory_mb, "elapsed_s": stream_measure.elapsed_s},
        ]
    ).to_csv(ROOT / "reports/LAB3/memory_compare.csv", index=False)

    alternate_seed = train_full(
        ROOT / "data/processed/train.csv",
        validation_path=ROOT / "data/processed/validation.csv",
        batch_size=cfg.batch_size,
        seed=cfg.seed + 1,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=cfg.device,
    )
    alternate_seed_p = alternate_seed.predict_proba(x_test)
    seed_negative = {
        "reference_seed": cfg.seed,
        "alternate_seed": cfg.seed + 1,
        "prediction_max_abs_diff": prediction_max_abs_diff(full_p, alternate_seed_p),
        "accuracy_abs_diff": abs(
            classification_metrics(y_test, full_p)["accuracy"]
            - classification_metrics(y_test, alternate_seed_p)["accuracy"]
        ),
    }
    seed_negative["pass"] = bool(seed_negative["prediction_max_abs_diff"] > 0)
    (ROOT / "reports/LAB3/seed_negative_test.json").write_text(
        json.dumps(seed_negative, indent=2), encoding="utf-8"
    )

    reproducibility = {
        "mode": "full_cpu_deterministic",
        "prediction_max_abs_diff": prediction_max_abs_diff(full_p, repeat_p),
        "metric_diff_accuracy": abs(classification_metrics(y_test, full_p)["accuracy"] - classification_metrics(y_test, repeat_p)["accuracy"]),
        "allowed_metric_diff": 0.005,
        "pass": prediction_max_abs_diff(full_p, repeat_p) == 0.0,
    }
    (ROOT / "reports/LAB3/reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    parity = device_parity(full, x_test[:512])
    pd.DataFrame([parity]).to_csv(ROOT / "reports/LAB3/device_parity.csv", index=False)
    ml_metrics = pd.read_csv(ROOT / "reports/LAB2/ml_metrics.csv")
    comparison = {
        "full_vs_chunk_prediction_max_abs_diff": prediction_max_abs_diff(full_p, stream_p),
        "full_model_sha256": sha256_file(ROOT / "reports/LAB3/dl_model.pt"),
        "ml_lab2_full_accuracy": float(ml_metrics.loc[ml_metrics["mode"] == "full", "accuracy"].iloc[0]),
        "dl_full_accuracy": classification_metrics(y_test, full_p)["accuracy"],
        "device_parity": parity,
        "reproducibility": reproducibility,
        "seed_negative_control": seed_negative,
    }
    (ROOT / "reports/LAB3/comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _config_negative(cfg)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
