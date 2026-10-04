from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from pydantic import ValidationError
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.alerts import AlertManager, AlertRule
from moons_lab.config import Lab8Config, load_config
from moons_lab.data import load_xy
from moons_lab.dl_pipeline import load_checkpoint
from moons_lab.generator import GeneratorSettings, generate_dataframe
from moons_lab.metrics import classification_metrics
from moons_lab.utils import json_dump, sha256_file
from moons_lab.validator import load_validator


def _positive(model, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x))
    return p[:, 1] if p.ndim == 2 else p


def _evaluate_source(source: str, run_no: int, seed: int) -> dict:
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_times()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    ml_model = joblib.load(ROOT / "reports/LAB2/ml_model.joblib")
    dl = load_checkpoint(ROOT / "reports/LAB3/dl_model.pt")
    validator = load_validator(ROOT / "reports/LAB4/validator_params.npz")
    if source == "real":
        x, y = load_xy(ROOT / "data/processed/test.csv")
    elif source == "synthetic":
        df = generate_dataframe(GeneratorSettings(mode="similar", n_samples=3600, seed=seed, noise=0.22))
        x = df[["x1", "x2"]].to_numpy()
        y = df["y"].to_numpy(dtype=int)
    else:
        raise ValueError(source)
    ml_p = _positive(ml_model, x)
    dl_p = dl.predict_proba(x)
    accepted = validator.accept(x, ml_p)
    elapsed = time.perf_counter() - start
    cpu_after = process.cpu_times()
    rss_after = process.memory_info().rss
    cpu_s = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    cpu_count = max(1, os.cpu_count() or 1)
    return {
        "source": source,
        "repeat": run_no,
        "rows": len(x),
        "ml_accuracy": classification_metrics(y, ml_p)["accuracy"],
        "ml_roc_auc": classification_metrics(y, ml_p)["roc_auc"],
        "dl_accuracy": classification_metrics(y, dl_p)["accuracy"],
        "dl_roc_auc": classification_metrics(y, dl_p)["roc_auc"],
        "validator_accept_rate": float(accepted.mean()),
        "elapsed_s": elapsed,
        "process_cpu_s": cpu_s,
        "cpu_utilization_pct_of_machine": 100.0 * cpu_s / max(elapsed * cpu_count, 1e-12),
        "rss_before_mb": rss_before / (1024 * 1024),
        "rss_after_mb": rss_after / (1024 * 1024),
    }


def _run_end_to_end(seed: int) -> pd.DataFrame:
    rows = []
    for source in ["real", "synthetic"]:
        rows.append(_evaluate_source(source, 1, seed))
        rows.append(_evaluate_source(source, 2, seed))
    frame = pd.DataFrame(rows)
    for source in ["real", "synthetic"]:
        pair = frame[frame["source"] == source]
        numeric = ["ml_accuracy", "ml_roc_auc", "dl_accuracy", "dl_roc_auc", "validator_accept_rate"]
        assert np.max(np.abs(pair.iloc[0][numeric].to_numpy(float) - pair.iloc[1][numeric].to_numpy(float))) == 0.0
    return frame


def _alert_drill(cfg: Lab8Config) -> dict:
    path = ROOT / "reports/LAB8/alerts_sample.jsonl"
    path.write_text("", encoding="utf-8")
    manager = AlertManager(path, cooldown_seconds=cfg.alert_cooldown_seconds)
    rules = {
        "drift": AlertRule("max_feature_psi", cfg.drift_psi_threshold, "above", "warning", "inspect incoming data and generator knobs"),
        "validator": AlertRule(
            "validator_reject_rate",
            cfg.validator_reject_rate_threshold,
            "above",
            "critical",
            "check applicability of current model",
        ),
        "service": AlertRule(
            "service_error_rate",
            cfg.service_error_rate_threshold,
            "above",
            "critical",
            "rollback or inspect service health",
        ),
    }
    drift = pd.read_csv(ROOT / "reports/LAB7/drift_metrics.csv")
    drift_value = float(drift.loc[drift["injected_drift"], ["psi_x1", "psi_x2"]].max().max())

    model = joblib.load(ROOT / "reports/LAB2/ml_model.joblib")
    validator = load_validator(ROOT / "reports/LAB4/validator_params.npz")
    random_df = generate_dataframe(GeneratorSettings(mode="random", n_samples=1500, seed=999, random_range=4.0))
    random_x = random_df[["x1", "x2"]].to_numpy()
    reject_rate = float((~validator.accept(random_x, _positive(model, random_x))).mean())

    emitted = []
    clean = [
        manager.evaluate(rules["drift"], 0.03, now=1000),
        manager.evaluate(rules["validator"], 0.03, now=1000),
        manager.evaluate(rules["service"], 0.0, now=1000),
    ]
    emitted.append(manager.evaluate(rules["drift"], drift_value, now=1100))
    emitted.append(manager.evaluate(rules["validator"], reject_rate, now=1100))
    emitted.append(manager.evaluate(rules["service"], 0.05, now=1100))
    duplicate = manager.evaluate(rules["drift"], drift_value * 1.1, now=1110)
    return {
        "clean_alerts": sum(item is not None for item in clean),
        "drill_alerts": sum(item is not None for item in emitted),
        "duplicate_suppressed": duplicate is None,
        "drift_value": drift_value,
        "validator_reject_rate_on_random": reject_rate,
        "simulated_service_error_rate": 0.05,
        "pass": sum(item is not None for item in clean) == 0 and sum(item is not None for item in emitted) == 3 and duplicate is None,
    }


def _install_probe() -> str:
    lines = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wheel_dir = tmp / "wheel"
        wheel_dir.mkdir()
        cmd = [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(wheel_dir)]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        lines.append("$ " + " ".join(cmd))
        lines.append(proc.stdout)
        lines.append(proc.stderr)
        lines.append(f"wheel_exit={proc.returncode}")
        wheels = list(wheel_dir.glob("*.whl"))
        if proc.returncode == 0 and wheels:
            env_dir = tmp / "venv"
            venv.EnvBuilder(with_pip=True).create(env_dir)
            py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)
            install = subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheels[0])], text=True, capture_output=True, env=clean_env)
            lines.append("$ clean-venv pip install --no-deps wheel")
            lines.append(install.stdout)
            lines.append(install.stderr)
            lines.append(f"install_exit={install.returncode}")
            probe = subprocess.run([str(py), "-c", "import moons_lab; print(moons_lab.__version__)"], text=True, capture_output=True, env=clean_env, cwd=tmp)
            lines.append("$ clean-venv python -c import moons_lab")
            lines.append(probe.stdout)
            lines.append(probe.stderr)
            lines.append(f"import_exit={probe.returncode}")
        lines.append("NOTE: dependency resolution from PyPI was not tested in the build container because outbound package access is disabled.")
        lines.append("RUN_ME_WINDOWS.ps1 performs the full online install with [all] extras on the student's machine.")
    return "\n".join(lines)


def _invalid_config_check(cfg: Lab8Config) -> str:
    try:
        Lab8Config.model_validate({**cfg.model_dump(), "drift_psi_threshold": -1})
        return "ERROR invalid config accepted"
    except ValidationError as exc:
        return f"PASS invalid config rejected: {exc.errors()[0]['msg']}"


def build_artifact_manifest() -> dict:
    roots = ["configs", "src", "scripts", "tests", "reports", "models", "data", ".github", "diagrams", "docs"]
    files = []
    for root_name in roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "artifacts_manifest.json" or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    for extra in ["README.md", "SUBMISSION_MAP.md", "LECTURE_REQUIREMENTS_MAP.md", "pyproject.toml", "requirements-lock.txt", "RUN_ME_WINDOWS.ps1", "RUN_LAB_WINDOWS.ps1", "GITHUB_SETUP.md", "defense_checklist.md", ".gitignore", ".gitattributes", ".pre-commit-config.yaml"]:
        path = ROOT / extra
        if path.exists():
            files.append({"path": extra, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"manifest_version": 1, "self_excluded": True, "file_count": len(files), "files": sorted(files, key=lambda x: x["path"])}


def main() -> None:
    cfg: Lab8Config = load_config(8)  # type: ignore[assignment]
    e2e = _run_end_to_end(cfg.seed)
    e2e.to_csv(ROOT / "reports/LAB8/end_to_end.csv", index=False)
    resources = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_logical": os.cpu_count() or 1,
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_gib": psutil.virtual_memory().total / (1024 ** 3),
        "e2e_elapsed_s_sum": float(e2e["elapsed_s"].sum()),
        "e2e_process_cpu_s_sum": float(e2e["process_cpu_s"].sum()),
        "e2e_peak_rss_mb": float(e2e[["rss_before_mb", "rss_after_mb"]].to_numpy().max()),
        "mean_cpu_utilization_pct_of_machine": float(e2e["cpu_utilization_pct_of_machine"].mean()),
    }
    json_dump(ROOT / "reports/LAB8/system_resources.json", resources)
    alert_result = _alert_drill(cfg)
    json_dump(ROOT / "reports/LAB8/alert_drill.json", alert_result)
    (ROOT / "reports/LAB8/install.log").write_text(_install_probe(), encoding="utf-8")
    (ROOT / "reports/LAB8/config_negative_test.log").write_text(_invalid_config_check(cfg) + "\n", encoding="utf-8")
    json_dump(ROOT / "reports/LAB8/artifacts_manifest.json", build_artifact_manifest())
    print(e2e.to_string(index=False))
    print(json.dumps(alert_result, indent=2))


if __name__ == "__main__":
    main()
