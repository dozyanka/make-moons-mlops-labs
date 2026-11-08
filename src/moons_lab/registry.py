"""SHA-256 checked, atomically persisted model registry and lifecycle states."""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib

from .utils import sha256_file

ModelStatus = Literal["preparing", "canary", "production", "retired"]
_STATUSES = {"preparing", "canary", "production", "retired"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, registry_path: str | Path):
        self.path = Path(registry_path)
        self._lock = threading.RLock()
        with self.path.open("r", encoding="utf-8") as handle:
            self._payload = json.load(handle)
        self._cache: dict[str, Any] = {}
        self._verify_all()

    def _model_path(self, item: dict) -> Path:
        path = Path(item["path"])
        return path if path.is_absolute() else self.path.parent / path

    def _verify_all(self) -> None:
        versions = self._payload["versions"]
        active = self._payload["active"]
        if active not in versions or versions[active]["status"] != "production":
            raise ValueError("registry must have an active production version")
        if sum(item["status"] == "production" for item in versions.values()) != 1:
            raise ValueError("registry must contain exactly one production version")
        for version, item in versions.items():
            if item["status"] not in _STATUSES:
                raise ValueError(f"invalid model status: {version}")
            model_path = self._model_path(item)
            if not model_path.is_file():
                raise FileNotFoundError(f"model {version} missing: {model_path}")
            actual = sha256_file(model_path)
            if actual != item["sha256"]:
                raise ValueError(f"hash mismatch for model {version}: expected {item['sha256']}, got {actual}")

    def _commit(self, updated: dict) -> None:
        """Persist before changing the in-memory snapshot (all under RLock)."""
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
        self._payload = updated

    @property
    def active_version(self) -> str:
        with self._lock:
            return str(self._payload["active"])

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._payload)

    def get_model(self, version: str | None = None) -> tuple[str, Any]:
        with self._lock:
            selected = version or self._payload["active"]
            if selected not in self._payload["versions"]:
                raise KeyError(selected)
            item = self._payload["versions"][selected]
            model_path = self._model_path(item)
            actual = sha256_file(model_path)
            if actual != item["sha256"]:
                self._cache.pop(selected, None)
                raise ValueError(f"hash mismatch for model {selected}")
            if selected not in self._cache:
                self._cache[selected] = joblib.load(model_path)
            return selected, self._cache[selected]

    def stage(self, version: str, status: Literal["preparing", "canary", "retired"],
              reason: str = "manual stage") -> dict:
        """Explicit state transitions. Active production cannot be demoted."""
        with self._lock:
            if version not in self._payload["versions"]:
                raise KeyError(version)
            if version == self._payload["active"]:
                raise ValueError("cannot stage active production version")
            old = self._payload["versions"][version]["status"]
            valid = {
                "retired": {"preparing", "canary"},
                "preparing": {"canary", "retired"},
                "canary": {"preparing", "retired"},
            }
            if status != old and status not in valid.get(old, set()):
                raise ValueError(f"invalid transition {old} -> {status}")
            if status == "canary":
                self.get_model(version)  # preload validated candidate
            if status != old:
                updated = self.snapshot()
                updated["versions"][version]["status"] = status
                updated.setdefault("history", []).append({
                    "time": utc_now(), "action": "stage", "from": old, "to": status,
                    "version": version, "reason": reason,
                })
                self._commit(updated)
            return self.snapshot()

    def promote(self, version: str, reason: str = "manual promote") -> dict:
        with self._lock:
            if version not in self._payload["versions"]:
                raise KeyError(version)
            previous = self._payload["active"]
            if previous == version:
                return self.snapshot()
            if self._payload["versions"][version]["status"] == "preparing":
                raise ValueError("model must be canary-tested before promotion")
            # Previously-retired models may be promoted for backward-compatible rollback drills.
            self.get_model(version)
            updated = self.snapshot()
            updated["versions"][previous]["status"] = "retired"
            updated["versions"][version]["status"] = "production"
            updated["active"] = version
            updated.setdefault("history", []).append({
                "time": utc_now(), "action": "promote", "from": previous, "to": version, "reason": reason,
            })
            self._commit(updated)
            return self.snapshot()

    def rollback(self, reason: str = "manual rollback") -> dict:
        with self._lock:
            current = self._payload["active"]
            history = self._payload.get("history", [])
            switches = [h for h in reversed(history) if h.get("action") in {"promote", "rollback"}
                        and h.get("to") == current and h.get("from") in self._payload["versions"]]
            if switches:
                target = switches[0]["from"]
            else:
                others = [v for v in self._payload["versions"] if v != current]
                if not others:
                    raise RuntimeError("no rollback target")
                target = others[0]
            self.get_model(target)
            updated = self.snapshot()
            updated["versions"][current]["status"] = "retired"
            updated["versions"][target]["status"] = "production"
            updated["active"] = target
            updated.setdefault("history", []).append({
                "time": utc_now(), "action": "rollback", "from": current, "to": target, "reason": reason,
            })
            self._commit(updated)
            return self.snapshot()
