"""Reproducible LAB6 runtime registry from immutable, published LAB2 models."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .utils import sha256_file


def build_registry(root: str | Path, registry_dir: str | Path) -> Path:
    root = Path(root).resolve()
    dst = Path(registry_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "v1": (root / "reports/LAB2/linear_baseline.joblib", "model_v1.joblib"),
        "v2": (root / "reports/LAB2/ml_model.joblib", "model_v2.joblib"),
    }
    originals = {version: sha256_file(source) for version, (source, _) in artifacts.items()}
    for version, (source, name) in artifacts.items():
        if not source.is_file():
            raise FileNotFoundError(f"published LAB2 model missing: {source}")
        target = dst / name
        if not target.is_file() or sha256_file(target) != originals[version]:
            shutil.copyfile(source, target)
        if sha256_file(target) != originals[version]:
            raise AssertionError(f"copied LAB2 model changed: {version}")
    comparison = json.loads((root / "reports/LAB2/comparison.json").read_text(encoding="utf-8"))
    versions = {}
    for version, (_, name) in artifacts.items():
        source_name = 'linear_baseline_test' if version == 'v1' else 'full_test'
        versions[version] = {
            "path": name,
            "sha256": originals[version],
            "metrics": {"test_accuracy": float(comparison[source_name]["accuracy"])},
            "status": "production" if version == "v2" else "retired",
            "description": ("Linear LogisticRegression baseline" if version == "v1"
                            else "StandardScaler + RBFSampler + SGDClassifier"),
        }
    registry = {"active": "v2", "versions": versions, "history": [
        {"action": "initial_registration", "from": None, "to": "v1", "reason": "baseline"},
        {"action": "promote", "from": "v1", "to": "v2", "reason": "better accuracy on LAB2 test"},
    ]}
    path = dst / "registry.json"
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path
