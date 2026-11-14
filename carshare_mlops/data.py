"""Data loading and preparation for the California Housing dataset."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_SOURCE_COLUMN = "MedHouseVal"
TARGET_COLUMN = "high_price"
TEST_SIZE = 0.20
RANDOM_STATE = 42


class DataSplit(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    price_threshold: float
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def load_housing_dataset() -> pd.DataFrame:
    """Load the California Housing dataset from scikit-learn; no download required."""
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame.copy()
    required = set(FEATURE_COLUMNS + [TARGET_SOURCE_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"California Housing dataset is missing expected columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("The California Housing dataset is empty.")
    if data[list(required)].isna().any().any():
        raise ValueError("The California Housing dataset contains missing values in required columns.")
    return data


def split_and_label_data(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Split raw records first, then derive the label threshold from training only.

    A district is labelled high price (1) when its MedHouseVal is at least the
    median MedHouseVal in the training partition. This keeps the test partition
    out of the threshold calculation.
    """
    required = set(FEATURE_COLUMNS + [TARGET_SOURCE_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    train_frame, test_frame = train_test_split(
        data.copy(), test_size=test_size, random_state=random_state
    )
    threshold = float(train_frame[TARGET_SOURCE_COLUMN].median())

    X_train = train_frame[FEATURE_COLUMNS].copy()
    X_test = test_frame[FEATURE_COLUMNS].copy()
    y_train = (train_frame[TARGET_SOURCE_COLUMN] >= threshold).astype("int64")
    y_test = (test_frame[TARGET_SOURCE_COLUMN] >= threshold).astype("int64")
    y_train.name = TARGET_COLUMN
    y_test.name = TARGET_COLUMN

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        price_threshold=threshold,
        train_frame=train_frame,
        test_frame=test_frame,
    )


def make_synthetic_shift(reference_features: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic, artificial current batch to demo data drift.

    Shifts are intentionally simulated:
      - MedInc        +0.5  (income shift)
      - HouseAge      +5    (older housing stock)
      - AveRooms      +0.3  (slightly larger homes)
      - AveBedrms     +0.1
      - Population    +50
      - AveOccup      +0.2
      - Latitude      +0.5  (geo shift northward)
      - Longitude     -0.5  (geo shift westward)
    This is not real production data — it is a controlled classroom demonstration.
    """
    current = reference_features[FEATURE_COLUMNS].copy()
    current["MedInc"]     = current["MedInc"]     + 0.5
    current["HouseAge"]   = current["HouseAge"]   + 5
    current["AveRooms"]   = current["AveRooms"]   + 0.3
    current["AveBedrms"]  = current["AveBedrms"]  + 0.1
    current["Population"] = current["Population"] + 50
    current["AveOccup"]   = current["AveOccup"]   + 0.2
    current["Latitude"]   = current["Latitude"]   + 0.5
    current["Longitude"]  = current["Longitude"]  - 0.5
    return current
