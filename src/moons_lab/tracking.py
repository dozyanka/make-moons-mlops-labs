from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .utils import json_dump


class RunHandle:
    def __init__(self, run_id: str, backend: str, base: Path | None = None, mlflow_mod: Any = None):
        self.run_id = run_id
        self.backend = backend
        self.base = base
        self.mlflow = mlflow_mod

    def log_params(self, params: dict[str, Any]) -> None:
        if self.backend == "mlflow":
            self.mlflow.log_params({k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()})
        else:
            json_dump(self.base / "params.json", params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        metrics = {k: float(v) for k, v in metrics.items()}
        if self.backend == "mlflow":
            self.mlflow.log_metrics(metrics)
        else:
            json_dump(self.base / "metrics.json", metrics)

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if self.backend == "mlflow":
            self.mlflow.log_artifact(str(path))
        else:
            dst = self.base / "artifacts" / path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


@contextmanager
def tracked_run(run_name: str, experiment: str = "make_moons") -> Iterator[RunHandle]:
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as active:
            yield RunHandle(active.info.run_id, "mlflow", mlflow_mod=mlflow)
    except ImportError:
        run_id = uuid.uuid4().hex
        base = Path("runs_fallback") / experiment / run_id
        base.mkdir(parents=True, exist_ok=True)
        (base / "meta.json").write_text(
            json.dumps({"run_id": run_id, "run_name": run_name, "backend": "fallback"}, indent=2),
            encoding="utf-8",
        )
        yield RunHandle(run_id, "fallback", base=base)

