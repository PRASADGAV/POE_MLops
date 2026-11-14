# Viva Quick Notes — California Housing MLOps POE

## Dataset
- **Source:** `sklearn.datasets.fetch_california_housing()` — bundled with scikit-learn, no CSV download.
- **Size:** 20,640 rows × 9 columns (8 features + 1 target).
- **Origin:** 1990 California census block groups.
- **Target column:** `MedHouseVal` — median house value in $100,000 units.

## Features
| Feature | Meaning |
|---|---|
| MedInc | Median income (tens of thousands USD) |
| HouseAge | Median house age in block group |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

## Binary target definition
- Split the data first (80/20, random_state=42).
- Calculate median `MedHouseVal` on the **training partition only**.
- `high_price = 1` if `MedHouseVal >= training_median`, else `0`.
- This prevents test-data leakage into the threshold calculation.

## Three tools

### Pytest
- Validates dataset shape, feature ranges, split reproducibility, model outputs, drift batch.
- Run: `python -m pytest -v`
- 7 test functions in `tests/test_pipeline.py`.

### MLflow
- Experiment: `CaliforniaHousing_HighPrice_Classification`
- Logs: params, metrics, tags, model (with signature), HTML report, metrics JSON, predictions CSV.
- Tracking store: `sqlite:///mlflow.db` (local file, no server needed).
- UI: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

### Evidently
- Compares reference features (test split) vs a synthetic shifted batch.
- Synthetic shifts: MedInc +0.5, HouseAge +5, Lat +0.5, Lon -0.5, etc.
- Output: `outputs/evidently_housing_data_drift.html`
- Uses `DataDriftPreset` — detects statistical distribution changes per feature.
- **Synthetic batch = classroom demonstration, not real production data.**

## Model
- `RandomForestClassifier` — 300 trees, max_depth=10, min_samples_leaf=2, random_state=42.
- Metrics: accuracy, precision, recall, F1.

## Key answers for viva questions

**Why binary classification on a regression target?**  
The dataset gives a continuous house value. Creating a high/low label is a standard classroom technique to demonstrate classification workflows on regression data.

**Why calculate the threshold from training data only?**  
Using test data to set the threshold would leak information from the held-out set, inflating apparent performance.

**What does Evidently's DataDriftPreset do?**  
It runs statistical tests (e.g. Wasserstein distance, PSI) on each feature column to detect whether the distribution of incoming data has shifted from the reference.

**Why is the current batch synthetic?**  
We have no real production data stream. The synthetic batch demonstrates how drift monitoring would work in practice — deliberately shifting features to trigger drift alerts.

**What does the MLflow run capture?**  
Model hyperparameters, evaluation metrics, dataset tags, the trained model artifact (with schema signature), the Evidently report, metrics JSON, and prediction CSV.
