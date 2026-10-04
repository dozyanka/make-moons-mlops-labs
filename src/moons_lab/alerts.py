from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AlertRule:
    metric: str
    threshold: float
    direction: str = "above"
    level: str = "warning"
    action: str = "inspect"

    def fires(self, value: float) -> bool:
        if self.direction == "above":
            return value > self.threshold
        if self.direction == "below":
            return value < self.threshold
        raise ValueError(f"unknown direction: {self.direction}")


class AlertManager:
    def __init__(self, path: str | Path, cooldown_seconds: int = 60):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self._last: dict[str, float] = {}

    def evaluate(self, rule: AlertRule, value: float, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        if not rule.fires(value):
            return None
        key = f"{rule.metric}:{rule.level}"
        if now - self._last.get(key, float("-inf")) < self.cooldown_seconds:
            return None
        self._last[key] = now
        payload = {
            "time": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "metric": rule.metric,
            "value": float(value),
            "threshold": float(rule.threshold),
            "level": rule.level,
            "action": rule.action,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload
