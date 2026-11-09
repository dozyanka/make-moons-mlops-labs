"""Evidently 0.7+ snapshot generation; failure is fatal for LAB7 evidence."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def evidently_available() -> bool:
    try:
        import evidently  # noqa: F401
        return True
    except ImportError:
        return False


def save_evidently_report(reference: pd.DataFrame, current: pd.DataFrame,
                          output_path: str | Path) -> str:
    """Produce a real Evidently HTML report, using current then reference.

    The contract is specifically Evidently >=0.7; do not quietly pretend
    successful completion if the optional package is missing or its API fails.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError("LAB7 requires Evidently: pip install -e '.[monitoring]'") from exc
    required = ["x1", "x2", "probability"]
    if not set(required).issubset(reference.columns) or not set(required).issubset(current.columns):
        raise ValueError("Evidently requires x1, x2, probability for both datasets")
    schema = DataDefinition(numerical_columns=required)
    ref_ds = Dataset.from_pandas(reference[required].reset_index(drop=True), data_definition=schema)
    cur_ds = Dataset.from_pandas(current[required].reset_index(drop=True), data_definition=schema)
    snapshot = Report([DataDriftPreset()]).run(cur_ds, ref_ds)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output))
    if not output.exists() or output.stat().st_size < 100:
        raise RuntimeError("Evidently returned empty HTML")
    return "saved"


def evaluate_evidently_window(reference: pd.DataFrame, current: pd.DataFrame,
                              html_path: str | Path | None = None) -> dict:
    """Run Evidently on a *single* bounded window, optionally save HTML.

    The numeric PSI/KS/CSI used for alert decisions are independently computed
    by moons_lab.drift; Evidently is run on every window as a cross-check.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError("Missing Evidently >=0.7. Install project monitoring extra") from exc
    columns = ["x1", "x2", "probability"]
    schema = DataDefinition(numerical_columns=columns)
    ref = Dataset.from_pandas(reference[columns], data_definition=schema)
    cur = Dataset.from_pandas(current[columns], data_definition=schema)
    snapshot = Report([DataDriftPreset()]).run(cur, ref)
    metrics = snapshot.dict().get("metrics", [])
    if html_path is not None:
        target = Path(html_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(target))
        if not target.is_file() or target.stat().st_size < 100:
            raise RuntimeError(f"Empty Evidently report: {target}")
    return {"evidently_evaluated": True, "evidently_metrics_count": len(metrics)}
