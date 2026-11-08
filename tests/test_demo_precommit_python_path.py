"""Regression checks for Windows pre-commit hooks launched by DEMO."""
from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import pytest

from DEMO import menu


def test_child_env_puts_active_python_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "environment" / "Scripts" / "python.exe"
    monkeypatch.setattr(menu, "PYTHON", executable)
    original = f"old-folder{os.pathsep}another-folder"
    monkeypatch.setenv("PATH", original)

    child_env = menu._demo_subprocess_env()

    assert child_env["PATH"] == f"{executable.parent}{os.pathsep}{original}"
    assert os.environ["PATH"] == original


def test_run_exe_forwards_path_to_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    executable = tmp_path / ".venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(menu, "PYTHON", executable)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

    monkeypatch.setattr(menu.subprocess, "run", fake_run)
    menu.run_exe(["pre-commit.exe", "run", "--all-files"])

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == ["pre-commit.exe", "run", "--all-files"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is True
    assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(executable.parent)


def test_logged_precommit_also_inherits_venv_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", tmp_path / "demo_output")
    executable = tmp_path / ".venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(menu, "PYTHON", executable)
    observed = []

    class DummyProcess:
        def __init__(self):
            self.stdout = io.StringIO("checks passed\n")

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        observed.append((cmd, kwargs))
        return DummyProcess()

    monkeypatch.setattr(menu.subprocess, "Popen", fake_popen)
    menu.run_exe(["pre-commit.exe", "run", "--all-files"], log_name="LAB1_PRECOMMIT")

    assert len(observed) == 1
    assert observed[0][1]["env"]["PATH"].split(os.pathsep)[0] == str(executable.parent)
    assert list((tmp_path / "demo_output" / "logs").glob("LAB1_PRECOMMIT_*.log"))


def test_normal_python_runner_also_has_active_python_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    monkeypatch.setattr(menu, "PYTHON", tmp_path / ".venv" / "Scripts" / "python.exe")
    observed = []

    def fake_run(cmd, **kwargs):
        observed.append((cmd, kwargs))

    monkeypatch.setattr(menu.subprocess, "run", fake_run)
    menu.run(["scripts/prepare_data.py"])

    assert len(observed) == 1
    cmd, kwargs = observed[0]
    assert cmd[0] == str(menu.PYTHON)
    assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(menu.PYTHON.parent)


def test_precommit_hook_python_entry_is_resolvable() -> None:
    # Pre-commit's nested hooks inherit the augmented PATH. On Windows,
    # subprocess.run(["python"], env=...) can resolve its initial executable
    # via the *parent* process PATH, so it is not a valid test of child PATH.
    child_env = menu._demo_subprocess_env()
    resolved = shutil.which("python", path=child_env["PATH"])
    assert resolved is not None
    assert Path(resolved).resolve() == menu.PYTHON.resolve()
