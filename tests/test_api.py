"""HTTPX-based Pytest tests for the FastAPI prediction service (api.py).

Uses FastAPI's TestClient (which wraps HTTPX) to exercise every endpoint
without starting a real server process.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app

# ---------------------------------------------------------------------------
# Shared client fixture — spins up the app (runs lifespan/startup) once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# Sample district payload (Los Angeles area, mid-range income)
SAMPLE_DISTRICT = {
    "MedInc": 3.5,
    "HouseAge": 28.0,
    "AveRooms": 5.2,
    "AveBedrms": 1.05,
    "Population": 1200.0,
    "AveOccup": 2.9,
    "Latitude": 34.2,
    "Longitude": -118.5,
}

# High-income district (San Francisco area) — expected to be high_price
HIGH_INCOME_DISTRICT = {
    "MedInc": 10.0,
    "HouseAge": 20.0,
    "AveRooms": 7.0,
    "AveBedrms": 1.0,
    "Population": 800.0,
    "AveOccup": 2.5,
    "Latitude": 37.75,
    "Longitude": -122.4,
}

# Low-income district — expected to be low_price
LOW_INCOME_DISTRICT = {
    "MedInc": 0.8,
    "HouseAge": 45.0,
    "AveRooms": 3.5,
    "AveBedrms": 1.1,
    "Population": 2000.0,
    "AveOccup": 4.5,
    "Latitude": 34.0,
    "Longitude": -117.5,
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def test_root_returns_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "California Housing" in body["service"]


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------
def test_info_returns_model_metadata(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "RandomForestClassifier"
    assert "price_threshold" in body
    assert isinstance(body["price_threshold"], float)
    assert len(body["features"]) == 8


# ---------------------------------------------------------------------------
# /predict — single district
# ---------------------------------------------------------------------------
def test_predict_returns_valid_response_for_sample_district(client):
    r = client.post("/predict", json=SAMPLE_DISTRICT)
    assert r.status_code == 200
    body = r.json()
    assert body["high_price"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold_used"] > 0
    assert isinstance(body["interpretation"], str)


def test_predict_high_income_district_is_high_price(client):
    r = client.post("/predict", json=HIGH_INCOME_DISTRICT)
    assert r.status_code == 200
    assert r.json()["high_price"] == 1


def test_predict_low_income_district_is_low_price(client):
    r = client.post("/predict", json=LOW_INCOME_DISTRICT)
    assert r.status_code == 200
    assert r.json()["high_price"] == 0


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in SAMPLE_DISTRICT.items() if k != "MedInc"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_predict_wrong_type_returns_422(client):
    bad = {**SAMPLE_DISTRICT, "MedInc": "not-a-number"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_probability_is_consistent_with_label(client):
    r = client.post("/predict", json=HIGH_INCOME_DISTRICT)
    body = r.json()
    if body["high_price"] == 1:
        assert body["probability"] >= 0.5
    else:
        assert body["probability"] < 0.5


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------
def test_batch_predict_returns_correct_count(client):
    payload = [SAMPLE_DISTRICT, HIGH_INCOME_DISTRICT, LOW_INCOME_DISTRICT]
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_batch_predict_all_labels_are_binary(client):
    payload = [SAMPLE_DISTRICT, HIGH_INCOME_DISTRICT, LOW_INCOME_DISTRICT]
    r = client.post("/predict/batch", json=payload)
    for pred in r.json()["predictions"]:
        assert pred["high_price"] in (0, 1)
        assert 0.0 <= pred["probability"] <= 1.0


def test_batch_predict_empty_list_returns_422(client):
    r = client.post("/predict/batch", json=[])
    assert r.status_code == 422


def test_batch_predict_single_item_works(client):
    r = client.post("/predict/batch", json=[SAMPLE_DISTRICT])
    assert r.status_code == 200
    assert r.json()["count"] == 1
