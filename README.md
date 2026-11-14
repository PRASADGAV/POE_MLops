# ML DevOps POE: California Housing Price Classification

**Tools demonstrated:** Pytest + MLflow + Evidently  
**Domain:** Real estate / housing  
**Use case:** Classify a California census district as lower-price or higher-price using housing and demographic features.

## 1. Dataset (no CSV download)

The project uses the California Housing dataset built into scikit-learn:

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
```

Scikit-learn describes this as 20,640 observations of California census block groups from the 1990 census. It has eight feature columns and one target column:

| Column | Description |
|---|---|
| `MedInc` | Median income in block group (tens of thousands USD) |
| `HouseAge` | Median house age in block group (years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` | Median house value (target, in units of $100,000) |

The dataset is bundled with scikit-learn. The project does **not** download a separate CSV.

## 2. Problem statement

A real estate analyst wants a simple baseline that flags districts likely to have above-median house values. We use all eight features as inputs.

The dataset provides `MedHouseVal` as a continuous value. The project creates a binary target:

- `high_price = 1` when `MedHouseVal` is greater than or equal to the **training partition's median**.
- `high_price = 0` when `MedHouseVal` is below that training median.

The raw rows are split before calculating the threshold. This avoids using the held-out test partition to define the label threshold.

## 3. The three tools

### Pytest — automated testing

`tests/test_pipeline.py` checks:

1. The dataset loads and has the expected schema (20,640 rows, 9 columns).
2. Feature values are in valid real-world ranges (latitude, longitude, age, income).
3. The train/test split is reproducible and the price threshold comes from training data only.
4. The Random Forest model produces valid predictions and metrics.
5. The synthetic current batch is deterministic and does not mutate the reference data.
6. Evidently creates an HTML report (the test runs when Evidently is installed).

Run:

```bash
python -m pytest -v
```

### MLflow — experiment tracking

The training run is saved under experiment `CaliforniaHousing_HighPrice_Classification` in a local SQLite tracking store (`mlflow.db`). The run records:

- Model parameters (tree count, depth, leaf size, random seed)
- Evaluation metrics (accuracy, precision, recall, F1)
- Dataset, domain, target-rule, and monitoring tags
- The trained scikit-learn model and its inferred signature
- The Evidently HTML report, metrics JSON, and prediction CSV as artifacts

Open the user interface after running the pipeline:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then visit `http://127.0.0.1:5000`.

### Evidently — data drift monitoring

The project compares:

- **Reference data:** held-out test features from the California Housing dataset.
- **Current data:** a deterministic synthetic copy with deliberate shifts.

The controlled demonstration applies: MedInc +0.5, HouseAge +5, AveRooms +0.3, AveBedrms +0.1, Population +50, AveOccup +0.2, Latitude +0.5, Longitude -0.5. Evidently's `DataDriftPreset` evaluates feature-distribution differences and writes:

`outputs/evidently_housing_data_drift.html`

The current batch is artificial. Explain it as a simulated data-drift example, not actual production drift.

## 4. Model

The solution uses `RandomForestClassifier` with 300 trees, maximum depth 10, minimum leaf size 2, and a fixed random seed. The code evaluates accuracy, precision, recall, F1 score, and a confusion matrix on the held-out partition.

## 5. Run on Windows

1. Extract the ZIP.
2. Ensure Python 3.11 is installed.
3. Double-click `setup_windows.bat` once. It creates `.venv` and installs `requirements.txt`.
4. Double-click `run_windows.bat`. It runs Pytest first and only starts the pipeline if tests pass.
5. Open `outputs/evidently_housing_data_drift.html` in a browser.
6. Double-click `start_mlflow_ui.bat` to inspect the MLflow experiment.

## 6. Run from the IDE terminal

```powershell
python -m pip install -r requirements.txt
python -m pytest -v
python main.py
```

## 7. Files produced

| File | Purpose |
|---|---|
| `outputs/eda_summary.json` | Dataset shape, null counts, split sizes, target rule and class counts |
| `outputs/classification_metrics.json` | Accuracy, precision, recall, F1 and confusion matrix |
| `outputs/test_predictions.csv` | Actual and predicted test labels and high-price probabilities |
| `outputs/synthetic_current_features.csv` | Artificial current feature batch used for the drift demo |
| `outputs/evidently_housing_data_drift.html` | Interactive Evidently HTML drift report |
| `mlflow.db` and `mlartifacts/` | Local MLflow experiment metadata and artifacts |

## 8. Viva explanation in 30 seconds

"I selected the real estate domain and used scikit-learn's built-in California Housing dataset, which loads in Python without downloading a separate CSV. I used all eight census block-group features to classify each district as lower or higher house price, where high price means a median house value at or above the median calculated from the training split. Pytest validates the data schema, feature ranges, split reproducibility, model outputs, and simulated drift batch. MLflow tracks experiment parameters, metrics, model and artifacts. Evidently compares reference features with a deliberately shifted synthetic batch and generates an HTML data-drift report."

## 9. Notes

- Internet is needed once to install the packages in `requirements.txt`, but not to fetch the dataset after scikit-learn is installed.
- This is an academic demonstration. The synthetic current batch is not real production data.

## Official documentation

- Scikit-learn California Housing: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- MLflow Tracking Quickstart: https://mlflow.org/docs/latest/ml/tracking/quickstart
- Evidently Data Drift preset: https://evidentlyai.github.io/evidently/api-reference/main/evidently/presets.html
- Pytest getting started: https://docs.pytest.org/en/stable/getting-started.html
