"""Evidently data-drift report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Compare current feature data with a reference feature distribution."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = list(reference_data.columns)
    report = Report([DataDriftPreset(columns=feature_columns)])
    snapshot = report.run(
        current_data=current_data[feature_columns],
        reference_data=reference_data[feature_columns],
    )
    snapshot.save_html(str(output))
    return output
