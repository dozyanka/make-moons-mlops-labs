"""Regression tests for LAB4 instructor launcher and report navigation."""
from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from DEMO import menu


ROOT = Path(__file__).resolve().parents[1]


def test_lab4_smoke_checks_real_files() -> None:
    menu.lab4_smoke_test()


def test_lab4_smoke_reports_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="LAB4 instructor DEMO missing"):
        menu.lab4_smoke_test()


def test_lab4_quick_demo_uses_saved_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr(menu, "run", lambda args, **kw: commands.append((args, kw)))
    menu.lab4_quick_demo()
    assert commands == [
        (["defense_demo/lab4_demo.py"], {"log_name": "LAB4_QUICK_DEMO"})
    ]


def test_lab4_quick_demo_fails_without_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(menu, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Сначала выполните ЛР4"):
        menu.lab4_quick_demo()


def test_full_lab4_uses_backup_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def backup(_lab: str):
        assert _lab == "LAB4"
        events.append("backup")
        try:
            yield
        finally:
            events.append("restore")

    def run_python(args, **kwargs):
        events.append("run-python")
        assert args == ["scripts/run_lab4.py"] or "pytest" in args

    def run_exe(args, **kwargs):
        events.append("run-powershell")
        assert "-SkipInstall" in args

    monkeypatch.setattr(menu, "reproduced_report", backup)
    monkeypatch.setattr(menu, "run", run_python)
    monkeypatch.setattr(menu, "run_exe", run_exe)
    menu.run_lab4()
    assert events[0] == "backup"
    assert events[-1] == "restore"
    assert any(x.startswith("run-") for x in events)


def test_full_lab4_restores_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def backup(_lab: str):
        events.append("backup")
        try:
            yield
        finally:
            events.append("restore")

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(menu, "reproduced_report", backup)
    monkeypatch.setattr(menu, "run", fail)
    monkeypatch.setattr(menu, "run_exe", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        menu.run_lab4()
    assert events == ["backup", "restore"]


def test_mlflow_file_backend_is_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure existing SQLite UI remains the default; LAB4 gets opt-in file mode.
    import inspect

    signature = inspect.signature(menu.start_mlflow)
    assert signature.parameters["file_backend"].default is False
    assert "MLFLOW_ALLOW_FILE_STORE" in inspect.getsource(menu.start_mlflow)
    assert "sqlite:///mlflow.db" in inspect.getsource(menu.start_mlflow)


def test_menu_exposes_lab4_choices(capsys: pytest.CaptureFixture[str]) -> None:
    menu.show_menu()
    text = capsys.readouterr().out
    for entry in ("10.", "11.", "12.", "13."):
        assert entry in text


def test_readme_has_report_navigation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = ROOT / "reports/LAB4/lab4_report.md"
    detail = (ROOT / "reports/LAB4/lab4_report.md").read_text(encoding="utf-8")
    assert "DEMO/RUN.cmd" in readme
    assert "reports/LAB4/" in readme
    assert report.is_file()
    assert "validator_params.npz" in detail


def test_published_lab4_report_is_not_overwritten_by_smoke() -> None:
    # Smoke must only read; the committed report is the source of the benchmark.
    path = ROOT / "reports/LAB4/lab4_report.md"
    before = path.read_bytes()
    menu.lab4_smoke_test()
    assert path.read_bytes() == before


def test_reproduced_report_restores_original_and_separates_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    report = root / "reports/LAB4/lab4_report.md"
    report.parent.mkdir(parents=True)
    report.write_text("ORIGINAL", encoding="utf-8")
    monkeypatch.setattr(menu, "ROOT", root)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", root / "demo_output")

    with menu.reproduced_report("LAB4"):
        report.write_text("NEW RUN", encoding="utf-8")
    assert report.read_text(encoding="utf-8") == "ORIGINAL"
    new_report = root / "demo_output/LAB4/lab4_report.md"
    assert new_report.read_text(encoding="utf-8") == "NEW RUN"

    with pytest.raises(RuntimeError, match="deliberate failure"):
        with menu.reproduced_report("LAB4"):
            report.write_text("FAILED RUN", encoding="utf-8")
            raise RuntimeError("deliberate failure")
    assert report.read_text(encoding="utf-8") == "ORIGINAL"
    assert new_report.read_text(encoding="utf-8") == "NEW RUN"


def test_lab4_mlflow_backend_command_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    exe = root / ".venv/Scripts/mlflow.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"stub")
    (root / "mlruns").mkdir()
    monkeypatch.setattr(menu, "ROOT", root)
    monkeypatch.setattr(menu, "DEMO_OUTPUT", root / "demo_output")
    port_results = iter((False, True))
    monkeypatch.setattr(menu, "port_is_open", lambda _port: next(port_results))
    captures = []

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captures.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(menu.subprocess, "Popen", fake_popen)
    result = menu.start_mlflow(port=5010, open_browser=False, file_backend=True)
    assert isinstance(result, FakeProcess)
    assert len(captures) == 1
    command, kwargs = captures[0]
    assert command[command.index("--backend-store-uri") + 1] == str(root / "mlruns")
    assert kwargs["env"]["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert "127.0.0.1" in command


def test_quick_demo_does_not_regenerate_checked_in_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(menu, "prepare_data", lambda: calls.append("regenerated"))
    monkeypatch.setattr(menu, "run", lambda args, **_kw: calls.append(args))
    menu.quick_demo()
    assert calls == [["defense_demo/run_all.py"]]
