"""FastAPI prediction service for the California Housing classifier.

Endpoints
---------
GET  /              — health check
GET  /info          — dataset and model metadata
POST /predict       — predict high_price for one district
POST /predict/batch — predict high_price for multiple districts
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared state — model is trained once at startup
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train the model on startup so every request can use it instantly."""
    from carshare_mlops.data import load_housing_dataset, split_and_label_data
    from carshare_mlops.model import train_model

    data = load_housing_dataset()
    split = data, split_and_label_data(data)
    _, sp = split
    model = train_model(sp.X_train, sp.y_train)

    _state["model"] = model
    _state["threshold"] = sp.price_threshold
    _state["feature_columns"] = list(sp.X_train.columns)
    print(f"[startup] model trained | threshold={sp.price_threshold:.4f}")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="California Housing Price Classifier",
    description=(
        "Binary classifier that flags a California census district as "
        "higher-price (1) or lower-price (0) based on housing and demographic features. "
        "Powered by a RandomForestClassifier trained on sklearn's built-in dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DistrictFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income (tens of thousands USD)", json_schema_extra={"example": 3.5})
    HouseAge: float = Field(..., description="Median house age in years", json_schema_extra={"example": 28.0})
    AveRooms: float = Field(..., description="Average rooms per household", json_schema_extra={"example": 5.2})
    AveBedrms: float = Field(..., description="Average bedrooms per household", json_schema_extra={"example": 1.05})
    Population: float = Field(..., description="Block group population", json_schema_extra={"example": 1200.0})
    AveOccup: float = Field(..., description="Average household occupancy", json_schema_extra={"example": 2.9})
    Latitude: float = Field(..., description="Block group latitude", json_schema_extra={"example": 34.2})
    Longitude: float = Field(..., description="Block group longitude", json_schema_extra={"example": -118.5})


class PredictionResponse(BaseModel):
    high_price: int = Field(..., description="1 = higher-price district, 0 = lower-price district")
    probability: float = Field(..., description="Probability of being a higher-price district")
    threshold_used: float = Field(..., description="MedHouseVal threshold (training median)")
    interpretation: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "California Housing Price Classifier", "version": "1.0.0"}


@app.get("/info", summary="Model and dataset metadata")
def info():
    if not _state:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "dataset": "sklearn.datasets.fetch_california_housing",
        "rows": 20640,
        "features": _state["feature_columns"],
        "target": "high_price",
        "target_rule": "MedHouseVal >= training median",
        "price_threshold": _state["threshold"],
        "model": "RandomForestClassifier",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict for a single district")
def predict(features: DistrictFeatures):
    if not _state:
        raise HTTPException(status_code=503, detail="Model not ready")

    import pandas as pd

    model = _state["model"]
    threshold = _state["threshold"]
    cols = _state["feature_columns"]

    row = pd.DataFrame([features.model_dump()])[cols]
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])

    return PredictionResponse(
        high_price=pred,
        probability=round(prob, 4),
        threshold_used=round(threshold, 4),
        interpretation=(
            "Higher-price district (MedHouseVal likely >= training median)"
            if pred == 1
            else "Lower-price district (MedHouseVal likely < training median)"
        ),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Predict for multiple districts")
def predict_batch(districts: list[DistrictFeatures]):
    if not _state:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not districts:
        raise HTTPException(status_code=422, detail="Provide at least one district")
    if len(districts) > 500:
        raise HTTPException(status_code=422, detail="Batch size cannot exceed 500")

    import pandas as pd

    model = _state["model"]
    threshold = _state["threshold"]
    cols = _state["feature_columns"]

    df = pd.DataFrame([d.model_dump() for d in districts])[cols]
    preds = model.predict(df).tolist()
    probs = model.predict_proba(df)[:, 1].tolist()

    results = [
        PredictionResponse(
            high_price=int(p),
            probability=round(float(pr), 4),
            threshold_used=round(threshold, 4),
            interpretation=(
                "Higher-price district" if p == 1 else "Lower-price district"
            ),
        )
        for p, pr in zip(preds, probs)
    ]
    return BatchPredictionResponse(predictions=results, count=len(results))
