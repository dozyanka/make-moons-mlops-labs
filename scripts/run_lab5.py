"""LAB5: deterministic synthetic make_moons generator and verified evidence package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab5Config, load_config
from moons_lab.drift import ks_stat, psi
from moons_lab.generator import (
    GeneratorSettings,
    dataframe_bytes,
    dataframe_hash,
    generate_dataframe,
    write_dataframe_csv,
)

FEATURES = ("x1", "x2")
SIMILAR_LIMIT = 0.10
RANDOM_PSI_FLOOR = 0.20
MIN_SIMILAR_FRACTION = 0.80
MIN_RANDOM_FRACTION = 0.50


def settings_from_config(cfg: Lab5Config) -> dict[str, GeneratorSettings]:
    base = dict(n_samples=cfg.n_samples, seed=cfg.seed)
    return {
        "similar": GeneratorSettings(mode="similar", noise=cfg.similar_noise, **base),
        "random": GeneratorSettings(mode="random", random_range=cfg.random_range, **base),
        "drifted": GeneratorSettings(
            mode="similar", noise=cfg.similar_noise,
            shift_x=cfg.drift_shift_x,
            shift_y=cfg.drift_shift_y,
            rotation_deg=cfg.drift_rotation_deg,
            label_flip_rate=cfg.drift_label_flip_rate,
            class_1_ratio=0.70,
            seasonality_amplitude=0.15,
            **base,
        ),
    }


def read_reference(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"The LAB1 reference dataset is missing: {path}")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = ROOT / "reports" / "LAB1" / "hash_manifest.json"
    expected = json.loads(baseline.read_text(encoding="utf-8"))["raw"]["sha256"]
    if file_hash != expected:
        raise RuntimeError("LAB1 reference CSV changed: refusing to change LAB4 input")
    data = pd.read_csv(path)
    if list(data.columns) != ["x1", "x2", "y"]:
        raise ValueError("LAB1 input must contain x1,x2,y in this order")
    if data.empty or not np.isfinite(data[["x1", "x2"]].to_numpy(dtype=float)).all():
        raise ValueError("LAB1 input has invalid numerical features")
    if not data["y"].isin([0, 1]).all():
        raise ValueError("LAB1 input must have binary classes")
    return data, file_hash


def class_psi(ref: pd.Series, generated: pd.Series) -> float:
    """PSI with two *fixed* categories, unlike continuous-feature quantile bins."""
    a = np.maximum(np.bincount(ref.to_numpy(dtype=int), minlength=2) / len(ref), 1e-9)
    b = np.maximum(
        np.bincount(generated.to_numpy(dtype=int), minlength=2) / len(generated), 1e-9
    )
    return float(np.sum((b - a) * np.log(b / a)))


def compare(reference: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, frame in frames.items():
        for feature in FEATURES:
            r = reference[feature]
            g = frame[feature]
            rows.append({
                "mode": mode,
                "feature": feature,
                "reference_mean": float(r.mean()),
                "generated_mean": float(g.mean()),
                "reference_std": float(r.std()),
                "generated_std": float(g.std()),
                "reference_class_1_share": float(reference["y"].mean()),
                "generated_class_1_share": float(frame["y"].mean()),
                "psi": psi(r.to_numpy(), g.to_numpy()),
                "ks": ks_stat(r.to_numpy(), g.to_numpy()),
            })
        rows.append({
            "mode": mode,
            "feature": "class_1_share",
            "reference_mean": float(reference["y"].mean()),
            "generated_mean": float(frame["y"].mean()),
            "reference_std": float(reference["y"].std()),
            "generated_std": float(frame["y"].std()),
            "reference_class_1_share": float(reference["y"].mean()),
            "generated_class_1_share": float(frame["y"].mean()),
            "psi": class_psi(reference["y"], frame["y"]),
            "ks": ks_stat(reference["y"].to_numpy(), frame["y"].to_numpy()),
        })
    return pd.DataFrame.from_records(rows)


def thresholds(metrics: pd.DataFrame) -> dict[str, object]:
    if not {"mode", "feature", "psi", "ks"}.issubset(metrics.columns):
        raise ValueError("LAB5 metrics lack mandatory mode/feature/psi/ks columns")
    similar = metrics[(metrics["mode"] == "similar") & metrics["feature"].isin(FEATURES)]
    random = metrics[(metrics["mode"] == "random") & metrics["feature"].isin(FEATURES)]
    if len(similar) != len(FEATURES) or len(random) != len(FEATURES):
        raise ValueError("Need both numeric features in both generation modes")
    similar_pass = (similar["psi"] <= SIMILAR_LIMIT) | (similar["ks"] <= SIMILAR_LIMIT)
    random_pass = random["psi"] >= RANDOM_PSI_FLOOR
    score = {
        "similar_fraction": float(similar_pass.mean()),
        "random_fraction": float(random_pass.mean()),
        "similar_pass": bool(similar_pass.mean() >= MIN_SIMILAR_FRACTION),
        "random_pass": bool(random_pass.mean() >= MIN_RANDOM_FRACTION),
    }
    return score


def run_pytest(output: Path) -> int:
    workspace = ROOT / "demo_output" / "pytest_runtime" / f"LAB5_{uuid.uuid4().hex}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_generator.py", "tests/test_lab5_workflow.py",
        "--cov=moons_lab.generator", "--cov-report=term-missing",
        f"--basetemp={workspace}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=ROOT, env=env, check=False, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = "> " + " ".join(command) + "\n\n" + completed.stdout
    (output / "generator_tests.log").write_text(log, encoding="utf-8", newline="\n")
    print(completed.stdout, flush=True)
    if completed.returncode:
        raise RuntimeError("LAB5 pytest failed; see generator_tests.log")
    lines = [line.strip() for line in completed.stdout.splitlines()]
    matches = [line for line in lines if "moons_lab/generator.py" in line.replace("\\", "/")]
    if len(matches) != 1:
        raise RuntimeError("Cannot find generator.py coverage row")
    percentages = re.findall(r"(\d{1,3})%", matches[0])
    if not percentages:
        raise RuntimeError("Cannot parse generator.py coverage")
    coverage = int(percentages[-1])
    if coverage < 80:
        raise RuntimeError(f"LAB5 generator coverage {coverage}% is below 80%")
    return coverage




def execute(report_dir: Path, data_dir: Path, *, skip_tests: bool = False) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(5)
    settings = settings_from_config(cfg)
    reference, source_sha = read_reference(ROOT / "data/raw/moons.csv")
    frames = {name: generate_dataframe(setting) for name, setting in settings.items()}
    hashes: dict[str, str] = {}
    for name, frame in frames.items():
        target = data_dir / f"{name}.csv"
        hashes[name] = write_dataframe_csv(frame, target)
        repeat = generate_dataframe(settings[name])
        if dataframe_bytes(repeat) != target.read_bytes():
            raise RuntimeError(f"Reproducibility failed in {name}: CSV differs by bytes")
        changed_seed = settings[name].model_copy(update={"seed": settings[name].seed + 1})
        if dataframe_hash(generate_dataframe(changed_seed)) == hashes[name]:
            raise RuntimeError(f"Negative control failed: seed does not affect {name}")
    metrics = compare(reference, frames)
    scores = thresholds(metrics)
    if not (scores["similar_pass"] and scores["random_pass"]):
        raise RuntimeError(f"LAB5 distribution criteria failed: {scores}")
    metrics.to_csv(
        report_dir / "synthetic_metrics.csv", index=False,
        float_format="%.8f", lineterminator="\n",
    )
    reproducibility = {
        "byte_identical_on_repeat": True,
        "different_seed_produces_different_sha256": True,
        "sha256": hashes,
        "thresholds": scores,
    }
    (report_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    config_payload = {
        "variant": 9,
        "source": "sklearn.datasets.make_moons",
        "reference_sha256": source_sha,
        "presets": {k: v.model_dump(mode="json") for k, v in settings.items()},
        "outputs_sha256": hashes,
    }
    (report_dir / "generator_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    coverage = run_pytest(report_dir) if not skip_tests else -1
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("LAB5 distribution thresholds:", scores)
    print("LAB5 data SHA-256:", hashes)
    if coverage >= 80:
        print(f"LAB5 PASSED; generator coverage {coverage}%; artifacts: {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/LAB5")
    parser.add_argument("--data-dir", default="data/synthetic")
    # Used by isolated integration tests; regular LAB5 runs always execute pytest.
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    data_dir = Path(args.data_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if report_dir.resolve().is_relative_to((ROOT / "reports/LAB4").resolve()):
        raise ValueError("LAB5 output must never overwrite LAB4 report files")
    if data_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()) or data_dir.resolve().is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("LAB5 output must never overwrite LAB1-4 input CSV files")
    execute(report_dir, data_dir, skip_tests=args.skip_tests)


if __name__ == "__main__":
    main()
