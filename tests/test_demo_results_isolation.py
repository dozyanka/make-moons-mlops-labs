"""DEMO never overwrites saved reference JSON/CSV and LAB1 hash manifest."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from DEMO import menu


def test_menu_redirects_generated_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(menu, "DEMO_OUTPUT", tmp_path / "demo_output")
    child_env = menu._demo_subprocess_env()
    assert child_env["MOONS_DEMO_RESULTS_DIR"] == str(tmp_path / "demo_output" / "defense_results")
    assert child_env["MOONS_DEMO_MANIFEST_PATH"] == str(tmp_path / "demo_output" / "data" / "hash_manifest.json")
    assert not (tmp_path / "demo_output").exists()


def test_result_writer_respects_isolation(tmp_path: Path) -> None:
    # Read common.py with a new interpreter, so the env is read during import.
    destination = tmp_path / "runtime" / "results"
    env = os.environ.copy()
    env["MOONS_DEMO_RESULTS_DIR"] = str(destination)
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(menu.ROOT / 'defense_demo')!r}); "
        "from common import save_result; "
        "save_result('test_result.json', {'pass': True})"
    )
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True)
    assert json.loads((destination / "test_result.json").read_text(encoding="utf-8")) == {"pass": True}
    assert not (tmp_path / "defense_demo").exists()


def test_result_writer_defaults_to_demo_output(tmp_path: Path) -> None:
    # A plain python defense_demo/run_all.py without menu should also be safe.
    env = os.environ.copy()
    env.pop("MOONS_DEMO_RESULTS_DIR", None)
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(menu.ROOT / 'defense_demo')!r}); "
        "from common import RESULTS; print(RESULTS)"
    )
    output = subprocess.check_output([sys.executable, "-c", script],
                                     cwd=tmp_path, env=env, text=True).strip()
    assert Path(output) == menu.ROOT / "demo_output" / "defense_results"


def test_prepare_data_manifest_can_be_redirected(tmp_path: Path) -> None:
    # Run with a reduced (still valid) sample count to avoid full trainings.
    configs = tmp_path / "configs"
    configs.mkdir()
    original = json.loads((menu.ROOT / "configs" / "lab1.json").read_text(encoding="utf-8"))
    original["n_samples"] = 2000
    original["chunk_size"] = 100
    (configs / "lab1.json").write_text(json.dumps(original), encoding="utf-8")
    path = tmp_path / "runtime" / "hash_manifest.json"
    env = os.environ.copy()
    env["MOONS_DEMO_MANIFEST_PATH"] = str(path)
    subprocess.run([sys.executable, str(menu.ROOT / "scripts" / "prepare_data.py")],
                   cwd=tmp_path, env=env, stdout=subprocess.PIPE, text=True, check=True)
    assert path.is_file()
    assert not (tmp_path / "reports" / "LAB1" / "hash_manifest.json").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["raw"]["sha256"]


def test_menu_runner_propagates_output_locations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", tmp_path / "demo_output")
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(menu.subprocess, "run", fake_run)
    menu.run(["defense_demo/run_all.py"])
    assert len(recorded) == 1
    assert recorded[0]["cwd"] == tmp_path
    assert recorded[0]["env"]["MOONS_DEMO_RESULTS_DIR"] == str(tmp_path / "demo_output" / "defense_results")


def test_training_data_restored_and_regenerated_copy_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    (root / "data/raw").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "data/chunks").mkdir(parents=True)
    raw = root / "data/raw/moons.csv"
    train = root / "data/processed/train.csv"
    old_chunk = root / "data/chunks/train_chunk_000.csv"
    raw.write_bytes(b"original\r\n")
    train.write_bytes(b"old train\r\n")
    old_chunk.write_bytes(b"old chunk\r\n")
    monkeypatch.setattr(menu, "ROOT", root)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", root / "demo_output")

    with menu.preserved_training_data("LAB1"):
        raw.write_bytes(b"regenerated\n")
        train.write_bytes(b"new train\n")
        old_chunk.unlink()
        (root / "data/chunks/train_chunk_001.csv").write_bytes(b"new chunk\n")

    assert raw.read_bytes() == b"original\r\n"
    assert train.read_bytes() == b"old train\r\n"
    assert old_chunk.read_bytes() == b"old chunk\r\n"
    assert not (root / "data/chunks/train_chunk_001.csv").exists()
    assert (root / "demo_output/LAB1/data/raw/moons.csv").read_bytes() == b"regenerated\n"
    assert (root / "demo_output/LAB1/data/chunks/train_chunk_001.csv").read_bytes() == b"new chunk\n"


def test_training_data_restored_even_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    original = root / "data/raw/moons.csv"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original")
    monkeypatch.setattr(menu, "ROOT", root)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", root / "demo_output")
    with pytest.raises(RuntimeError, match="injected"):
        with menu.preserved_training_data("LAB2"):
            original.write_bytes(b"damaged")
            raise RuntimeError("injected")
    assert original.read_bytes() == b"original"


def test_lab7_report_menu_never_promises_generated_markdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", tmp_path / "demo_output")
    menu.open_lab7_report()
    output = capsys.readouterr().out
    assert "lab7_report.md" in output
    assert "summary.json" not in output
    assert "для расчёта выберите пункт 20" in output
    assert "новый отчёт" not in output
