"""Create an isolated LAB6 registry once; subsequent launches retain its state."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .utils import sha256_file


def build_registry(root: str | Path, registry_dir: str | Path, *, reset: bool = False) -> Path:
    """Keep production state across restarts; reset only for an explicit test run.

    Published LAB2 models are never modified.  When a registry exists, a missing
    or corrupted runtime artifact is an error rather than silently repaired.
    """
    root = Path(root).resolve()
    dst = Path(registry_dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "registry.json"
    artifacts = {
        "v1": (root / "reports/LAB2/linear_baseline.joblib", "model_v1.joblib"),
        "v2": (root / "reports/LAB2/ml_model.joblib", "model_v2.joblib"),
    }
    originals = {}
    for version, (source, _) in artifacts.items():
        if not source.is_file():
            raise FileNotFoundError(f"published LAB2 model missing: {source}")
        originals[version] = sha256_file(source)

    if path.exists() and not reset:
        from .registry import ModelRegistry

        saved = ModelRegistry(path)  # validates every registered artifact
        snapshot = saved.snapshot()
        if set(snapshot.get("versions", {})) != set(artifacts):
            raise ValueError("existing LAB6 registry has unexpected versions")
        for version, (_, filename) in artifacts.items():
            entry = snapshot["versions"][version]
            candidate = (path.parent / filename).resolve()
            registered = (path.parent / entry["path"]).resolve()
            if registered != candidate or entry["sha256"] != originals[version]:
                raise ValueError(f"existing LAB6 registry differs from published LAB2: {version}")
        return path

    for version, (source, name) in artifacts.items():
        target = dst / name
        if not target.is_file() or sha256_file(target) != originals[version]:
            shutil.copyfile(source, target)
        if sha256_file(target) != originals[version]:
            raise AssertionError(f"copied LAB2 model changed: {version}")

    comparison = json.loads((root / "reports/LAB2/comparison.json").read_text(encoding="utf-8"))
    versions = {}
    for version, (_, name) in artifacts.items():
        source_name = "linear_baseline_test" if version == "v1" else "full_test"
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
    temporary = path.with_suffix(".json.new")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path
