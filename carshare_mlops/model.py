"""Model creation, training, and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def build_model() -> RandomForestClassifier:
    """Build a reproducible Random Forest classifier."""
    return RandomForestClassifier(**MODEL_PARAMETERS)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    model = build_model()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Return scalar metrics, hard predictions, and class-1 probabilities."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }
    return metrics, predictions, probabilities


def confusion_matrix_as_list(
    y_true: pd.Series, predictions: np.ndarray
) -> list[list[int]]:
    return confusion_matrix(y_true, predictions, labels=[0, 1]).astype(int).tolist()
