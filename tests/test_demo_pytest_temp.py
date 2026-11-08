"""Regression tests: instructor pytest runs do not depend on Windows system Temp."""
from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from DEMO import menu


def test_pytest_arguments_get_private_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "demo_output"
    monkeypatch.setattr(menu, "DEMO_OUTPUT", output)
    original = ["-m", "pytest", "tests/test_data.py", "-q"]
    actual = menu._with_project_pytest_temp(original, log_name="LAB1_TESTS")
    assert original == ["-m", "pytest", "tests/test_data.py", "-q"]
    assert actual[:len(original)] == original
    assert len(actual) == len(original) + 1
    assert actual[-1].startswith("--basetemp=")
    assert Path(actual[-1].split("=", 1)[1]).parent == output / "pytest_runtime"
    assert (output / "pytest_runtime").is_dir()
    assert menu._with_project_pytest_temp(original, log_name="LAB1_TESTS")[-1] != actual[-1]


def test_pytest_explicit_workspace_and_non_pytest_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(menu, "DEMO_OUTPUT", tmp_path / "demo_output")
    existing = ["-m", "pytest", "-q", "--basetemp=custom/path"]
    assert menu._with_project_pytest_temp(existing, log_name="TESTS") == existing
    existing_split = ["-m", "pytest", "--basetemp", "custom/path"]
    assert menu._with_project_pytest_temp(existing_split, log_name="TESTS") == existing_split
    non_pytest = ["scripts/run_lab1.py"]
    assert menu._with_project_pytest_temp(non_pytest, log_name="LAB1") == non_pytest
    assert not (tmp_path / "demo_output").exists()


def test_real_menu_runner_passes_basetemp_to_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", tmp_path / "demo_output")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(menu.subprocess, "run", fake_run)
    menu.run(["-m", "pytest", "tests/test_eda.py", "-q"])
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "--basetemp=" in " ".join(command)
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is True


@pytest.mark.parametrize("menu_action", ["run_lab1", "run_lab2", "run_lab3", "all_tests"])
def test_all_lab1_to_lab3_pytest_menu_routes_use_common_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, menu_action: str,
) -> None:
    commands = []
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    monkeypatch.setattr(menu, "prepare_data", lambda: None)
    monkeypatch.setattr(menu, "run", lambda args, **kwargs: commands.append(args))

    @contextlib.contextmanager
    def no_op_backup(_lab: str):
        yield

    monkeypatch.setattr(menu, "reproduced_report", no_op_backup)
    getattr(menu, menu_action)()
    pytest_calls = [x for x in commands if x[:2] == ["-m", "pytest"]]
    assert len(pytest_calls) == 1
