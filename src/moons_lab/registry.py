from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .utils import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, registry_path: str | Path):
        self.path = Path(registry_path)
        self._lock = threading.RLock()
        self._payload = self._read()
        self._cache: dict[str, Any] = {}
        self._verify_all()

    def _read(self) -> dict:
        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)

    def _verify_all(self) -> None:
        for version, item in self._payload["versions"].items():
            model_path = Path(item["path"])
            if not model_path.exists():
                raise FileNotFoundError(f"model {version} missing: {model_path}")
            actual = sha256_file(model_path)
            if actual != item["sha256"]:
                raise ValueError(f"hash mismatch for model {version}: expected {item['sha256']}, got {actual}")

    @property
    def active_version(self) -> str:
        with self._lock:
            return str(self._payload["active"])

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._payload)

    def get_model(self, version: str | None = None) -> tuple[str, Any]:
        with self._lock:
            version = version or self._payload["active"]
            if version not in self._payload["versions"]:
                raise KeyError(version)
            if version not in self._cache:
                item = self._payload["versions"][version]
                model_path = Path(item["path"])
                actual = sha256_file(model_path)
                if actual != item["sha256"]:
                    raise ValueError(f"hash mismatch for model {version}")
                self._cache[version] = joblib.load(model_path)
            return version, self._cache[version]

    def promote(self, version: str, reason: str = "manual promote") -> dict:
        with self._lock:
            if version not in self._payload["versions"]:
                raise KeyError(version)
            previous = self._payload["active"]
            if previous == version:
                return self.snapshot()
            self._payload["versions"][previous]["status"] = "retired"
            self._payload["versions"][version]["status"] = "production"
            self._payload["active"] = version
            self._payload.setdefault("history", []).append(
                {"time": utc_now(), "action": "promote", "from": previous, "to": version, "reason": reason}
            )
            self._write()
            return self.snapshot()

    def rollback(self, reason: str = "manual rollback") -> dict:
        with self._lock:
            history = self._payload.get("history", [])
            current = self._payload["active"]
            candidates = [h for h in reversed(history) if h.get("action") == "promote" and h.get("to") == current]
            if candidates:
                target = candidates[0]["from"]
            else:
                other = [v for v in self._payload["versions"] if v != current]
                if not other:
                    raise RuntimeError("no rollback target")
                target = other[0]
            previous = current
            self._payload["versions"][previous]["status"] = "retired"
            self._payload["versions"][target]["status"] = "production"
            self._payload["active"] = target
            self._payload.setdefault("history", []).append(
                {"time": utc_now(), "action": "rollback", "from": previous, "to": target, "reason": reason}
            )
            self._write()
            return self.snapshot()
