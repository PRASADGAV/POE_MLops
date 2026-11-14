"""Pytest checks for the California Housing MLOps pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carshare_mlops.data import (
    FEATURE_COLUMNS,
    load_housing_dataset,
    make_synthetic_shift,
    split_and_label_data,
)
from carshare_mlops.model import evaluate_model, train_model

# California Housing has 20,640 rows and 9 columns (8 features + 1 target).
EXPECTED_ROWS = 20_640
EXPECTED_COLS = 9


def test_california_housing_dataset_loads_with_expected_schema():
    data = load_housing_dataset()
    assert data.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    assert set(FEATURE_COLUMNS + ["MedHouseVal"]).issubset(data.columns)
    assert not data[FEATURE_COLUMNS + ["MedHouseVal"]].isna().any().any()


def test_feature_values_are_in_valid_ranges():
    data = load_housing_dataset()
    # Latitude in California: roughly 32–42 N
    assert data["Latitude"].between(32, 42).all()
    # Longitude in California: roughly -125 to -114
    assert data["Longitude"].between(-125, -114).all()
    # HouseAge must be non-negative
    assert (data["HouseAge"] >= 0).all()
    # MedInc must be positive
    assert (data["MedInc"] > 0).all()


def test_label_threshold_is_calculated_from_training_partition_only():
    data = load_housing_dataset()
    split = split_and_label_data(data)
    assert split.price_threshold == pytest.approx(split.train_frame["MedHouseVal"].median())
    assert set(split.y_train.unique()).issubset({0, 1})
    assert set(split.y_test.unique()).issubset({0, 1})
    assert len(split.X_train) + len(split.X_test) == len(data)


def test_split_is_reproducible():
    data = load_housing_dataset()
    first = split_and_label_data(data)
    second = split_and_label_data(data)
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_random_forest_trains_and_returns_valid_classification_metrics():
    data = load_housing_dataset()
    split = split_and_label_data(data)
    model = train_model(split.X_train, split.y_train)
    metrics, predictions, probabilities = evaluate_model(model, split.X_test, split.y_test)

    assert len(predictions) == len(split.y_test)
    assert probabilities.shape == (len(split.y_test),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert set(np.unique(predictions)).issubset({0, 1})
    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_synthetic_drift_batch_is_reproducible_and_does_not_mutate_reference():
    data = load_housing_dataset()
    reference = split_and_label_data(data).X_test[FEATURE_COLUMNS].reset_index(drop=True)
    reference_before = reference.copy(deep=True)

    shifted = make_synthetic_shift(reference)
    shifted_again = make_synthetic_shift(reference)

    pd.testing.assert_frame_equal(shifted, shifted_again)
    pd.testing.assert_frame_equal(reference, reference_before)

    assert np.allclose(shifted["MedInc"],     reference["MedInc"]     + 0.5)
    assert np.allclose(shifted["HouseAge"],   reference["HouseAge"]   + 5)
    assert np.allclose(shifted["Latitude"],   reference["Latitude"]   + 0.5)
    assert np.allclose(shifted["Longitude"],  reference["Longitude"]  - 0.5)


def test_evidently_report_is_created(tmp_path):
    pytest.importorskip("evidently")
    from carshare_mlops.monitoring import generate_drift_report

    data = load_housing_dataset()
    reference = split_and_label_data(data).X_test[FEATURE_COLUMNS].reset_index(drop=True)
    current = make_synthetic_shift(reference)
    output = tmp_path / "drift_report.html"

    returned_path = generate_drift_report(reference, current, output)

    assert returned_path == output
    assert output.exists()
    assert output.stat().st_size > 1_000
