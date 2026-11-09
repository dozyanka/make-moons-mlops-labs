"""Replayable LAB8 alert/stream subsystem, without touching prior laboratories."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .alerts import AlertManager, AlertRule
from .drift import psi, ks_stat


class StreamGuard:
    """Bounded, stepped PSI/CSI detector with pause/resume checkpointing."""
    def __init__(self, reference: np.ndarray, ref_proba: np.ndarray, *,
                 window: int, step: int, threshold: float, alert_path: Path,
                 checkpoint: Path, cooldown: int = 60):
        if not 0 < step <= window:
            raise ValueError("invalid window/step")
        if np.asarray(reference).shape[1] != 2:
            raise ValueError("reference must have two numeric features")
        self.reference = np.asarray(reference, dtype=float)
        self.ref_p = np.asarray(ref_proba, dtype=float)
        self.window = window
        self.step = step
        self.threshold = threshold
        self.queue: deque[tuple[float, float, float]] = deque(maxlen=window)
        self.processed = 0
        self.next_boundary = window
        self.window_index = -1
        self.checkpoint = checkpoint
        self.alerts = AlertManager(alert_path, cooldown_seconds=cooldown)
        self.rule = AlertRule("max_psi_csi", threshold, "above", "warning", "investigate drift; rollback model")
        self.rows: list[dict] = []
        self.degraded = False
        self.phase = "initial"
        self.gaps: list[dict] = []

    def add(self, x: np.ndarray, probabilities: np.ndarray) -> list[dict]:
        x = np.asarray(x, dtype=float).reshape(-1, 2)
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        if len(x) != len(probabilities):
            raise ValueError("features/probabilities length mismatch")
        events = []
        for (a, b), p in zip(x, probabilities):
            if not all(map(math.isfinite, (a, b, p))):
                raise ValueError("non-finite observation")
            self.queue.append((float(a), float(b), float(p)))
            self.processed += 1
            if self.processed >= self.next_boundary:
                values = np.asarray(self.queue)
                vals = [psi(self.reference[:, j], values[:, j]) for j in (0, 1)]
                csi = psi(self.ref_p, values[:, 2])
                ks1 = ks_stat(self.reference[:, 0], values[:, 0])
                ks2 = ks_stat(self.reference[:, 1], values[:, 1])
                self.window_index += 1
                value = max(*vals, csi)
                alert = self.alerts.evaluate(self.rule, value)
                row = {"window": self.window_index, "phase": self.phase, "processed": self.processed,
                       "psi_x1": vals[0], "psi_x2": vals[1], "ks_x1": ks1, "ks_x2": ks2, "csi": csi,
                       "metric": value, "alert": alert is not None,
                       "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                self.rows.append(row)
                events.append(row)
                self.next_boundary += self.step
        return events

    def rebase(self, probabilities: np.ndarray, phase: str) -> None:
        """Model switches change prediction distribution; reset CSI reference to avoid false alarms."""
        p = np.asarray(probabilities, dtype=float).reshape(-1)
        if len(p) != len(self.reference):
            raise ValueError("reference prediction count mismatch")
        self.ref_p = p
        self.phase = phase
        self.queue.clear()
        self.next_boundary = self.processed + self.window

    def pause(self) -> None:
        self.degraded = True
        self.gaps.append({"kind": "stream_paused", "time": time.time(), "at_processed": self.processed})
        self.save()

    def resume(self) -> None:
        self.restore()
        self.degraded = False
        self.gaps.append({"kind": "stream_resumed", "time": time.time(), "at_processed": self.processed})
        self.save()

    def save(self) -> None:
        obj = {"processed": self.processed, "next_boundary": self.next_boundary,
               "window_index": self.window_index, "phase": self.phase, "buffer": list(self.queue),
               "gaps": self.gaps, "degraded": self.degraded}
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint.with_suffix(".new")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        tmp.replace(self.checkpoint)

    def restore(self) -> None:
        obj = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        self.processed = int(obj["processed"])
        self.next_boundary = int(obj["next_boundary"])
        self.window_index = int(obj["window_index"])
        self.phase = str(obj.get("phase", "initial"))
        self.queue = deque((tuple(v) for v in obj["buffer"]), maxlen=self.window)
        self.gaps = list(obj["gaps"])
        self.degraded = bool(obj["degraded"])

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            cols = ["window", "phase", "processed", "psi_x1", "psi_x2", "ks_x1", "ks_x2", "csi", "metric", "alert", "timestamp_utc"]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self.rows)


def hash_manifest(root: Path, report: Path) -> dict:
    """Immutable snapshot; volatile caches and the manifest itself are omitted."""
    excluded = {".git", ".venv", "demo_output", "mlruns", "mlartifacts", "__pycache__", ".pytest_cache",
                ".gitignore", "data/runtime_synthetic"}
    entries = []
    for p in root.rglob("*"):
        if not p.is_file() or p == report or any(x in excluded for x in p.relative_to(root).parts):
            continue
        if p.name in {"mlflow.db", ".coverage"} or p.suffix in {".pyc", ".tmp"}:
            continue
        if p.relative_to(root).parts[0] not in {"reports", "data", "configs", "src", "scripts", "tests", "LAB4", "LAB5", "LAB6", "LAB7", "LAB8", "diagrams", "DEMO", "defense_demo", ".github"} and p.name not in {"pyproject.toml", "README.md", "RUN_DEMO.cmd"}:
            continue
        rel = p.relative_to(root).as_posix()
        entries.append({"path": rel, "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    # The runtime evidence is intentionally ignored by Git, but belongs in the
    # deliverable manifest and must be auditable independently.
    runtime_evidence = []
    if report.parent.is_dir():
        for p in sorted(report.parent.iterdir()):
            if p.is_file() and p != report:
                runtime_evidence.append({"path": p.relative_to(root).as_posix() if p.is_relative_to(root)
                                         else str(p.resolve()),
                                         "bytes": p.stat().st_size,
                                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return {"schema": 2, "items": sorted(entries, key=lambda e: e["path"]),
            "runtime_evidence": runtime_evidence, "count": len(entries) + len(runtime_evidence)}
