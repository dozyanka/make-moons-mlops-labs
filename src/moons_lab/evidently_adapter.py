from __future__ import annotations

from pathlib import Path

import pandas as pd


def evidently_available() -> bool:
    try:
        import evidently  # noqa: F401

        return True
    except ImportError:
        return False


def save_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: str | Path) -> str:
    """Best-effort integration isolated from core monitoring because Evidently API changes between releases."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return "unavailable"
    try:
        report = Report([DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(result, "save_html"):
            result.save_html(str(output_path))
        elif hasattr(report, "save_html"):
            report.save_html(str(output_path))
        else:
            return "api_not_supported"
        return "saved"
    except Exception as exc:  # pragma: no cover - depends on external version
        return f"error:{type(exc).__name__}:{exc}"
