"""Byte-preservation and report-manifest tests for LAB4 Git publication."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_lab4_data", ROOT / "scripts/verify_lab4_data.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_baseline_includes_every_original_chunk() -> None:
    baseline = json.loads((ROOT / "LAB4/data_integrity_baseline.json").read_text(encoding="utf-8"))
    assert len([p for p in baseline["files"] if p.startswith("data/chunks/")]) == 24
    assert len(baseline["files"]) == 28
    assert baseline["files"]["data/raw/moons.csv"]["sha256"] == (
        "dd38bd033f2300777fb0420e4c7bf895946435fa3b76c4d495a69e120b54be91"
    )


def test_baseline_matches_original_files() -> None:
    mod.verify_data()


def test_invalid_hash_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / "sample.csv"
    file.write_bytes(b"x1,x2,y\n1,2,0\n")
    bad = {"format_version": 1, "files": {"sample.csv": {"sha256": "f" * 64, "bytes": 11}},
           "working_model": {"path": "sample.csv", "sha256": "f" * 64, "bytes": 11}}
    (tmp_path / "baseline.json").write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "baseline.json")
    with pytest.raises(AssertionError, match="BYTE HASH MISMATCH"):
        mod.verify_data()
