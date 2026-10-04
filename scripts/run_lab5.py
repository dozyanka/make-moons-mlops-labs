from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab5Config, load_config
from moons_lab.drift import ks_stat, psi
from moons_lab.generator import GeneratorSettings, LiveGeneratorController, dataframe_hash, generate_dataframe
from moons_lab.generator_service import create_generator_app
from moons_lab.tracking import tracked_run
from moons_lab.utils import json_dump


def _metric_rows(reference: pd.DataFrame, current: pd.DataFrame, mode: str, run_id: str, backend: str) -> list[dict]:
    rows = []
    for feature in ["x1", "x2"]:
        rows.append({
            "mode": mode,
            "feature": feature,
            "psi": psi(reference[feature].to_numpy(), current[feature].to_numpy()),
            "ks": ks_stat(reference[feature].to_numpy(), current[feature].to_numpy()),
            "run_id": run_id,
            "tracking_backend": backend,
        })
    rows.append({
        "mode": mode,
        "feature": "class_1_share",
        "psi": abs(float(reference["y"].mean()) - float(current["y"].mean())),
        "ks": abs(float(reference["y"].mean()) - float(current["y"].mean())),
        "run_id": run_id,
        "tracking_backend": backend,
    })
    return rows


def _live_update_demo() -> dict:
    controller = LiveGeneratorController(GeneratorSettings(n_samples=1000, seed=123, shift_x=0.0))
    app = create_generator_app(controller)
    client = TestClient(app)
    process_identity = id(controller)
    before = client.post("/generate", json={"n_samples": 1000, "seed": 555}).json()["records"]
    update = client.patch("/config", json={"shift_x": 1.25, "noise": 0.3}).json()
    after = client.post("/generate", json={"n_samples": 1000, "seed": 555}).json()["records"]
    before_mean = sum(r["x1"] for r in before) / len(before)
    after_mean = sum(r["x1"] for r in after) / len(after)
    return {
        "controller_identity_before_after": process_identity,
        "same_process": id(controller) == process_identity,
        "updated_config": update,
        "x1_mean_before": before_mean,
        "x1_mean_after": after_mean,
        "mean_shift": after_mean - before_mean,
        "verdict": "runtime_update_applied_without_restart" if abs(after_mean - before_mean) > 1.0 else "ERROR",
    }


def main() -> None:
    cfg: Lab5Config = load_config(5)  # type: ignore[assignment]
    reference = pd.read_csv(ROOT / "data/raw/moons.csv")
    settings = {
        "similar": GeneratorSettings(
            mode="similar", n_samples=cfg.n_samples, seed=cfg.seed, noise=cfg.similar_noise
        ),
        "random": GeneratorSettings(
            mode="random", n_samples=cfg.n_samples, seed=cfg.seed, random_range=cfg.random_range
        ),
        "drifted": GeneratorSettings(
            mode="similar",
            n_samples=cfg.n_samples,
            seed=cfg.seed,
            noise=cfg.similar_noise,
            shift_x=cfg.drift_shift_x,
            shift_y=cfg.drift_shift_y,
            rotation_deg=cfg.drift_rotation_deg,
            label_flip_rate=cfg.drift_label_flip_rate,
        ),
    }
    rows: list[dict] = []
    hashes: dict[str, str] = {}
    for mode, setting in settings.items():
        df = generate_dataframe(setting)
        path = ROOT / "data/synthetic" / f"{mode}.csv"
        df.to_csv(path, index=False, float_format="%.10f")
        hashes[mode] = dataframe_hash(df)
        with tracked_run(f"lab5_{mode}", experiment="lab5_generator") as run:
            run.log_params(setting.model_dump())
            metrics = {
                "psi_x1": psi(reference["x1"].to_numpy(), df["x1"].to_numpy()),
                "psi_x2": psi(reference["x2"].to_numpy(), df["x2"].to_numpy()),
                "ks_x1": ks_stat(reference["x1"].to_numpy(), df["x1"].to_numpy()),
                "ks_x2": ks_stat(reference["x2"].to_numpy(), df["x2"].to_numpy()),
            }
            run.log_metrics(metrics)
            run.log_artifact(path)
            rows.extend(_metric_rows(reference, df, mode, run.run_id, run.backend))

    pd.DataFrame(rows).to_csv(ROOT / "reports/LAB5/synthetic_metrics.csv", index=False)
    json_dump(ROOT / "reports/LAB5/generator_config.json", settings["drifted"].model_dump())
    repeat = generate_dataframe(settings["similar"])
    reproducibility = {
        "first_hash": hashes["similar"],
        "repeat_hash": dataframe_hash(repeat),
        "byte_equivalent_csv": hashes["similar"] == dataframe_hash(repeat),
    }
    json_dump(ROOT / "reports/LAB5/reproducibility.json", reproducibility)
    json_dump(ROOT / "reports/LAB5/live_update_demo.log", _live_update_demo())
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(reproducibility, indent=2))


if __name__ == "__main__":
    main()
