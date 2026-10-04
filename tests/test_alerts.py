import json

import pytest

from moons_lab.alerts import AlertManager, AlertRule


def test_alerts_fire_and_suppress(tmp_path):
    path = tmp_path / "alerts.jsonl"
    manager = AlertManager(path, cooldown_seconds=60)
    rule = AlertRule("psi", 0.2, "above", "warning", "inspect drift")
    assert manager.evaluate(rule, 0.1, now=1000) is None
    assert manager.evaluate(rule, 0.5, now=1001) is not None
    assert manager.evaluate(rule, 0.6, now=1010) is None
    assert manager.evaluate(rule, 0.7, now=1062) is not None
    assert len(path.read_text().strip().splitlines()) == 2


def test_below_rule_and_bad_direction(tmp_path):
    manager = AlertManager(tmp_path / "a.jsonl", cooldown_seconds=0)
    rule = AlertRule("quality", 0.8, "below")
    assert manager.evaluate(rule, 0.7, now=10) is not None
    bad = AlertRule("x", 1.0, "sideways")
    with pytest.raises(ValueError):
        bad.fires(0.0)
