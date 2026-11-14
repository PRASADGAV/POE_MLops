# 🚀 Start Guide — California Housing MLOps POE

---

## What This Project Does

This project builds a **binary classification pipeline** on the California Housing dataset using three MLOps tools:

| Tool | Role |
|---|---|
| **Pytest** | Automated testing — validates data, model, and drift batch |
| **MLflow** | Experiment tracking — logs parameters, metrics, model, and artifacts |
| **Evidently** | Data drift monitoring — compares reference vs synthetic current data |

---

## Quick Start (3 steps)

### Step 1 — Install dependencies (first time only)
```powershell
python -m pip install -r requirements.txt
```

### Step 2 — Run tests + pipeline
```powershell
python -m pytest -v
python main.py
```

### Step 3 — View results
- **Drift report:** open `outputs/evidently_housing_data_drift.html` in your browser
- **MLflow UI:** run the command below, then visit http://127.0.0.1:5000
```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

---

## Output Files

| File | Description |
|---|---|
| `outputs/eda_summary.json` | Dataset shape, column list, class counts, threshold |
| `outputs/classification_metrics.json` | Accuracy, precision, recall, F1, confusion matrix |
| `outputs/test_predictions.csv` | Actual vs predicted labels with probabilities |
| `outputs/synthetic_current_features.csv` | Synthetic drift batch used by Evidently |
| `outputs/evidently_housing_data_drift.html` | Interactive HTML drift report |
| `mlflow.db` | Local SQLite MLflow experiment store |
| `mlartifacts/` | MLflow model and artifact files |

---

## Project File Structure

```
ML_DevOps_CarShare_POE/
├── main.py                        ← Pipeline entry point
├── requirements.txt               ← Python dependencies
├── carshare_mlops/
│   ├── data.py                    ← Dataset loading, splitting, synthetic shift
│   ├── model.py                   ← RandomForest train + evaluate
│   ├── monitoring.py              ← Evidently drift report
│   └── tracking.py                ← MLflow logging
├── tests/
│   └── test_pipeline.py           ← 7 Pytest test functions
└── outputs/                       ← Generated artifacts (auto-created)
```

---

## Dataset Brief — California Housing

### Origin
- **Source:** `sklearn.datasets.fetch_california_housing()` — bundled with scikit-learn, no download needed.
- **Origin:** 1990 US Census, California block groups (small geographic units ~1,000 people).
- **Size:** 20,640 rows × 9 columns — zero missing values.

---

### Input Features (8 columns)

| # | Feature | Full Name | Unit | Mean | Min | Max | What It Represents |
|---|---|---|---|---|---|---|---|
| 1 | `MedInc` | Median Income | $10,000s | 3.87 | 0.50 | 15.00 | Median income of households in the block group |
| 2 | `HouseAge` | Median House Age | years | 28.6 | 1 | 52 | Median age of houses in the block group |
| 3 | `AveRooms` | Average Rooms | rooms/household | 5.43 | 0.85 | 141.9 | Average number of rooms per household |
| 4 | `AveBedrms` | Average Bedrooms | bedrooms/household | 1.10 | 0.33 | 34.1 | Average number of bedrooms per household |
| 5 | `Population` | Population | people | 1,425 | 3 | 35,682 | Total population of the block group |
| 6 | `AveOccup` | Average Occupancy | people/household | 3.07 | 0.69 | 1,243 | Average number of people per household |
| 7 | `Latitude` | Latitude | degrees N | 35.63 | 32.54 | 41.95 | Geographic latitude of block group centroid |
| 8 | `Longitude` | Longitude | degrees W | -119.57 | -124.35 | -114.31 | Geographic longitude of block group centroid |

> **Key insight:** `MedInc` is the strongest predictor of house price — higher income neighbourhoods consistently show higher property values.

---

### Target Variable

| Column | `MedHouseVal` |
|---|---|
| Type | Continuous → converted to binary |
| Unit | $100,000 USD |
| Range | $14,999 – $500,001 |
| Mean | $206,856 |
| Median | $179,700 ← **classification threshold** |

#### How the binary label is created

```
high_price = 1   if MedHouseVal >= 1.7985  (≥ $179,850)
high_price = 0   if MedHouseVal <  1.7985  (< $179,850)
```

The threshold (median = **1.7985**) is calculated from the **training partition only** to avoid data leakage.

| Class | Label | Count | % |
|---|---|---|---|
| Lower price | 0 | 10,315 | 49.98% |
| Higher price | 1 | 10,325 | 50.02% |

The classes are almost perfectly balanced — no need for resampling.

---

### Train / Test Split

| Partition | Rows | % |
|---|---|---|
| Training | 16,512 | 80% |
| Test | 4,128 | 20% |

Split uses `random_state=42` for full reproducibility.

---

## Model

| Setting | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Trees (`n_estimators`) | 300 |
| Max depth (`max_depth`) | 10 |
| Min leaf size (`min_samples_leaf`) | 2 |
| Random seed | 42 |

### Results on test set

| Metric | Score |
|---|---|
| **Accuracy** | **88.25%** |
| **Precision** | **88.72%** |
| **Recall** | **87.46%** |
| **F1 Score** | **88.09%** |

**Confusion matrix:**

```
                 Predicted 0   Predicted 1
  Actual 0          1850          228
  Actual 1           257         1793
```

---

## Synthetic Drift Demo (Evidently)

Since there is no live production stream, a deterministic synthetic batch is created by shifting the test features:

| Feature | Shift Applied |
|---|---|
| `MedInc` | +0.5 (income increase) |
| `HouseAge` | +5 years (older housing stock) |
| `AveRooms` | +0.3 |
| `AveBedrms` | +0.1 |
| `Population` | +50 |
| `AveOccup` | +0.2 |
| `Latitude` | +0.5° northward |
| `Longitude` | -0.5° westward |

Evidently's `DataDriftPreset` runs statistical tests on each feature and flags which columns have drifted. Open `outputs/evidently_housing_data_drift.html` to see the interactive results.

---

## Common Questions

**Why classify instead of regress?**  
The pipeline demonstrates binary classification tools (precision, recall, F1, confusion matrix). Converting the continuous target to high/low labels is standard practice in MLOps coursework.

**Why is the threshold from training data only?**  
Using the test set to compute the threshold would leak information from the held-out partition, artificially inflating metrics.

**What does the MLflow run store?**  
All hyperparameters, evaluation metrics, dataset tags, the trained model (with schema signature), the Evidently HTML report, metrics JSON, and predictions CSV — everything needed to reproduce or audit the experiment.

**Is this production-ready?**  
No. It is an academic demonstration. The drift batch is synthetic, the model is a baseline, and no deployment infrastructure is included.
