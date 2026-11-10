"""The all-labs CLI must not erase historical reports or accidentally run 55min LAB8."""
from __future__ import annotations

import pytest

from scripts import run_all


def test_default_reproduces_labs_1_to_7_and_skips_lab8(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    for lab in range(1, 8):
        monkeypatch.setattr(run_all.menu, f"run_lab{lab}", lambda n=lab: calls.append(n))
    monkeypatch.setattr(run_all.menu, "run_lab8", lambda quick: calls.append((8, quick)))
    run_all.main([])
    assert calls == list(range(1, 8))


def test_lab8_is_opt_in_and_quick_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    for lab in range(1, 8):
        monkeypatch.setattr(run_all.menu, f"run_lab{lab}", lambda: None)
    monkeypatch.setattr(run_all.menu, "run_lab8", lambda quick: calls.append(quick))
    run_all.main(["--lab8", "quick"])
    run_all.main(["--lab8", "full"])
    assert calls == [True, False]


def test_historical_audit_skips_all_lab_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(run_all, "verify_historical_lab8", lambda: called.append("audit"))
    for lab in range(1, 8):
        monkeypatch.setattr(run_all.menu, f"run_lab{lab}", lambda: called.append("lab"))
    run_all.main(["--audit-original-lab8"])
    assert called == ["audit"]
