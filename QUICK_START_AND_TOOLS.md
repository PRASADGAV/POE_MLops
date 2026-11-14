# Quick Start Guide & Tools Reference
## California Housing MLOps POE

---

## Quick Start (3 Commands)

```powershell
# 1. Install dependencies (first time only)
python -m pip install -r requirements.txt

# 2. Run all tests
python -m pytest tests/ -v

# 3. Run the full pipeline
python main.py
```

That's it. The pipeline trains the model, generates the Evidently drift report, and logs everything to MLflow.

---

## View Results

| What | How |
|---|---|
| MLflow experiment UI | Run `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` then open http://127.0.0.1:5000 |
| FastAPI prediction API | Run `python -m uvicorn api:app --port 8000 --reload` then open http://127.0.0.1:8000/docs |
| Evidently drift report | Open `outputs/evidently_housing_data_drift.html` in your browser |
| Jenkins CI dashboard | Open http://localhost:8081 |

---

## Project Structure

```
ML_DevOps_CarShare_POE/
|
|-- main.py                        Pipeline entry point
|-- api.py                         FastAPI prediction service
|-- Jenkinsfile                    Jenkins CI pipeline definition
|-- requirements.txt               Python dependencies
|-- workflow_diagram.png           Visual pipeline diagram (project root)
|-- QUICK_START_AND_TOOLS.md       This file
|-- README.md                      Full project documentation
|-- VIVA_QUICK_NOTES.md            Revision notes
|-- START_GUIDE.md                 Dataset brief and start guide
|
|-- carshare_mlops/
|   |-- data.py                    Dataset loading, splitting, synthetic shift
|   |-- model.py                   RandomForest train + evaluate
|   |-- monitoring.py              Evidently drift report generation
|   |-- tracking.py                MLflow experiment logging
|   `-- __init__.py
|
|-- tests/
|   |-- test_pipeline.py           7 pytest tests (data, model, evidently)
|   `-- test_api.py                12 HTTPX tests (FastAPI endpoints)
|
`-- outputs/                       Generated files (auto-created on run)
    |-- eda_summary.json
    |-- classification_metrics.json
    |-- test_predictions.csv
    |-- synthetic_current_features.csv
    `-- evidently_housing_data_drift.html
```

---

## Output Files

| File | Description |
|---|---|
| `outputs/eda_summary.json` | Dataset shape, column list, class counts, threshold value |
| `outputs/classification_metrics.json` | Accuracy, precision, recall, F1, confusion matrix |
| `outputs/test_predictions.csv` | Actual vs predicted labels with probabilities for all test rows |
| `outputs/synthetic_current_features.csv` | Synthetically shifted batch used by Evidently |
| `outputs/evidently_housing_data_drift.html` | Interactive HTML drift report |
| `mlflow.db` | Local SQLite MLflow tracking store |
| `mlartifacts/` | MLflow model artifacts and logged files |
| `workflow_diagram.png` | Full pipeline workflow diagram image |

---

---

# Tools Reference — What Each Tool Is and How It Is Used Here

---

## 1. Python 3

**What it is:**
Python is the programming language the entire project is written in. Version 3.11.9 is used here.

**How it is used here:**
Every component — data loading, model training, API, testing, monitoring, and tracking — is written in Python. It is the runtime for all scripts and the venv that Jenkins builds on each run.

**Key commands:**
```powershell
python --version          # verify version
python main.py            # run the pipeline
python -m pytest tests/   # run all tests
python -m uvicorn api:app --port 8000  # start API server
```

---

## 2. Visual Studio Code

**What it is:**
VS Code is a lightweight but powerful source code editor made by Microsoft. It supports Python development through extensions.

**How it is used here:**
Used as the primary IDE to write, edit, and run all project files. The Kiro AI assistant runs inside VS Code to help build the project.

**Extensions to install (Ctrl+Shift+X):**
- Python Extension Pack
- Pylance
- GitLens

---

## 3. Git for Windows

**What it is:**
Git is a distributed version control system that tracks changes in source code. Git for Windows (version 2.53) adds Git support natively on Windows.

**How it is used here:**
The project folder is initialised as a Git repository. All source files are committed and pushed to GitHub. Jenkins uses Git to pull the latest code on every build.

**Commands used in this project:**
```powershell
git init                                      # initialise repo
git add .                                     # stage all files
git commit -m "message"                       # commit changes
git remote add origin <url>                   # link to GitHub
git push -u origin main                       # push to GitHub
git branch -M main                            # rename branch
```

**Commits made:**
```
a76f553  fix: add skops_trusted_types for RandomForest
980b123  fix: use absolute Python path in Jenkinsfile
0b59400  Initial commit: California Housing MLOps pipeline
```

---

## 4. GitHub Account

**What it is:**
GitHub is a cloud-based hosting platform for Git repositories. It stores your code online and enables collaboration and CI/CD integration.

**How it is used here:**
The project is pushed to `https://github.com/PRASADGAV/POE_MLops`. Jenkins is configured to pull from this repository on every build, so a push to GitHub automatically triggers the CI pipeline.

**Repository:** `https://github.com/PRASADGAV/POE_MLops`
**Branch:** `main`

---

## 5. Java JDK

**What it is:**
Java Development Kit is the runtime environment required to run Jenkins. Jenkins is a Java application distributed as a `.war` file.

**How it is used here:**
Java JDK 21 LTS is installed on the machine. Jenkins uses it silently — you do not write Java code for this project. Without Java, Jenkins cannot start.

**Installed version:**
```
java version "21.0.11" 2026-04-21 LTS
Java HotSpot(TM) 64-Bit Server VM (build 21.0.11+9-LTS-211)
```

---

## 6. Jenkins

**What it is:**
Jenkins is an open-source automation server used for Continuous Integration and Continuous Delivery (CI/CD). It runs pipelines automatically when code changes.

**How it is used here:**
A `Jenkinsfile` in the project root defines a 6-stage declarative pipeline. When you click Build Now (or push to GitHub with polling enabled), Jenkins:

1. Pulls the latest code from GitHub
2. Creates a Python virtual environment and installs all dependencies
3. Runs syntax checks on all Python files
4. Runs the 7 pipeline unit tests
5. Runs the 12 API unit tests
6. Runs the full ML pipeline (trains model, generates report, logs to MLflow)
7. Archives all output files as build artifacts

**Access:** http://localhost:8081

**Jenkinsfile pipeline stages:**
```
Checkout  -->  Setup Environment  -->  Lint  -->  Unit Tests (Pipeline)  -->  Unit Tests (API)  -->  Run ML Pipeline  -->  Archive Artifacts
```

**All 6 stages passed on the last build.**

---

## 7. FastAPI

**What it is:**
FastAPI is a modern, high-performance Python web framework for building APIs. It is based on Python type hints and automatically generates interactive documentation.

**How it is used here:**
`api.py` defines a prediction service that loads the trained RandomForest model at startup and exposes 4 endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns `{"status": "ok"}` |
| `/info` | GET | Returns model metadata, 8 feature names, price threshold |
| `/predict` | POST | Accepts one district's features, returns `high_price` (0/1) and probability |
| `/predict/batch` | POST | Accepts up to 500 districts, returns all predictions |

**Example request:**
```json
POST /predict
{
  "MedInc": 10.0,
  "HouseAge": 20.0,
  "AveRooms": 7.0,
  "AveBedrms": 1.0,
  "Population": 800.0,
  "AveOccup": 2.5,
  "Latitude": 37.75,
  "Longitude": -122.4
}
```

**Example response:**
```json
{
  "high_price": 1,
  "probability": 0.9433,
  "threshold_used": 1.7985,
  "interpretation": "Higher-price district (MedHouseVal likely >= training median)"
}
```

**Interactive docs:** http://127.0.0.1:8000/docs

---

## 8. Uvicorn

**What it is:**
Uvicorn is a lightning-fast ASGI (Asynchronous Server Gateway Interface) web server for Python. It is the recommended server for running FastAPI applications.

**How it is used here:**
Uvicorn serves the FastAPI app defined in `api.py`. The `--reload` flag is used in development so the server restarts automatically when code changes.

**Command to start the API server:**
```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

**In Jenkins**, the API is tested using FastAPI's `TestClient` (which wraps HTTPX internally) so no live server process is needed during testing.

---

## 9. Pytest

**What it is:**
Pytest is the most widely used Python testing framework. It discovers and runs test functions automatically and produces detailed pass/fail reports.

**How it is used here:**
19 tests are split across two files:

**`tests/test_pipeline.py` — 7 tests (data and model validation)**

| Test | What it checks |
|---|---|
| `test_california_housing_dataset_loads_with_expected_schema` | Dataset has 20,640 rows and 9 columns |
| `test_feature_values_are_in_valid_ranges` | Latitude 32-42 N, longitude -125 to -114, age >= 0, income > 0 |
| `test_label_threshold_is_calculated_from_training_partition_only` | Threshold equals training median exactly |
| `test_split_is_reproducible` | Same rows every run with random_state=42 |
| `test_random_forest_trains_and_returns_valid_classification_metrics` | All 4 metrics between 0.0 and 1.0 |
| `test_synthetic_drift_batch_is_reproducible_and_does_not_mutate_reference` | Shifts applied correctly, reference unchanged |
| `test_evidently_report_is_created` | HTML report file exists and is larger than 1 KB |

**`tests/test_api.py` — 12 tests (FastAPI endpoint validation)**

| Test | What it checks |
|---|---|
| `test_root_returns_ok` | GET / returns 200 and status ok |
| `test_info_returns_model_metadata` | GET /info returns 8 features and threshold |
| `test_predict_returns_valid_response_for_sample_district` | POST /predict returns valid label and probability |
| `test_predict_high_income_district_is_high_price` | High MedInc (10.0) correctly classified as 1 |
| `test_predict_low_income_district_is_low_price` | Low MedInc (0.8) correctly classified as 0 |
| `test_predict_missing_field_returns_422` | Missing field triggers validation error |
| `test_predict_wrong_type_returns_422` | String in numeric field triggers validation error |
| `test_predict_probability_is_consistent_with_label` | Probability >= 0.5 when label = 1 |
| `test_batch_predict_returns_correct_count` | 3 inputs returns count = 3 |
| `test_batch_predict_all_labels_are_binary` | All batch labels are 0 or 1 |
| `test_batch_predict_empty_list_returns_422` | Empty list triggers validation error |
| `test_batch_predict_single_item_works` | Single item batch works correctly |

**Run all tests:**
```powershell
python -m pytest tests/ -v
```

**Result: 19/19 passed**

---

## 10. HTTPX

**What it is:**
HTTPX is a fully featured HTTP client for Python that supports both synchronous and asynchronous requests. It is used by FastAPI's `TestClient` to make HTTP requests to the API during testing.

**How it is used here:**
All 12 API tests in `tests/test_api.py` use FastAPI's `TestClient`, which is built on top of HTTPX. The TestClient starts the FastAPI app in-process (no live server needed) and makes real HTTP requests to every endpoint.

```python
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)   # HTTPX under the hood

def test_predict_high_income_district_is_high_price(client):
    r = client.post("/predict", json=HIGH_INCOME_DISTRICT)
    assert r.status_code == 200
    assert r.json()["high_price"] == 1
```

---

## 11. Evidently

**What it is:**
Evidently is an open-source Python library for monitoring machine learning models in production. It evaluates data quality, data drift, and model performance.

**How it is used here:**
`carshare_mlops/monitoring.py` uses Evidently's `DataDriftPreset` to compare two feature datasets:

- **Reference data:** the held-out test features from the real California Housing dataset
- **Current data:** a synthetically shifted version of those features (simulating production drift)

The 8 deliberate shifts applied to create the current batch:

| Feature | Shift |
|---|---|
| MedInc | +0.5 |
| HouseAge | +5 years |
| AveRooms | +0.3 |
| AveBedrms | +0.1 |
| Population | +50 |
| AveOccup | +0.2 |
| Latitude | +0.5 degrees |
| Longitude | -0.5 degrees |

Evidently runs statistical tests (Wasserstein distance, PSI) on each feature and writes an interactive HTML report showing which features have drifted.

**Output:** `outputs/evidently_housing_data_drift.html`

**Key code:**
```python
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report([DataDriftPreset(columns=feature_columns)])
snapshot = report.run(current_data=current, reference_data=reference)
snapshot.save_html("outputs/evidently_housing_data_drift.html")
```

> **Note:** The current batch is synthetic and created for classroom demonstration only. It is not real production data.

---

## 12. Pandas

**What it is:**
Pandas is the standard Python library for data manipulation and analysis. It provides DataFrame and Series structures for working with tabular data.

**How it is used here:**
Pandas is used throughout the pipeline:

| Location | Usage |
|---|---|
| `data.py` | Loads the dataset as a DataFrame, performs train/test split, creates binary labels |
| `model.py` | X_train, X_test passed as DataFrames to RandomForest |
| `main.py` | Builds prediction DataFrame, saves CSV outputs |
| `monitoring.py` | Passes DataFrames to Evidently for drift comparison |
| `tracking.py` | Passes X_train DataFrame to MLflow for model signature inference |
| `api.py` | Converts incoming JSON to DataFrame for model.predict() |

---

## 13. MLflow

**What it is:**
MLflow is an open-source platform for the full machine learning lifecycle. It handles experiment tracking, model packaging, and model registry.

**How it is used here:**
`carshare_mlops/tracking.py` logs a complete experiment run to a local SQLite database (`mlflow.db`) under the experiment name `CaliforniaHousing_HighPrice_Classification`.

**What gets logged per run:**

| Category | Items logged |
|---|---|
| Parameters | n_estimators, max_depth, min_samples_leaf, random_state, dataset name, target rule, threshold, feature count |
| Metrics | accuracy, precision, recall, f1_score |
| Tags | domain, problem type, model type, dataset source, drift demo type |
| Model | Trained RandomForestClassifier with input signature and example |
| Artifacts | Evidently HTML report, classification_metrics.json, test_predictions.csv |

**Start the UI:**
```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

**Access:** http://127.0.0.1:5000

---

## 14. Python Extension Pack (VS Code)

**What it is:**
A VS Code extension bundle that includes IntelliSense, linting, debugging, Jupyter notebook support, and Python test integration for VS Code.

**How it is used here:**
Provides syntax highlighting, auto-complete, inline error detection, and the ability to run and debug Python files directly inside VS Code.

**Install:** Ctrl+Shift+X → search "Python Extension Pack" → Install

---

## 15. Pylance (VS Code)

**What it is:**
Pylance is Microsoft's high-performance language server for Python in VS Code. It provides fast type checking, auto-imports, and code navigation powered by Pyright.

**How it is used here:**
Gives real-time type error feedback as you write code. For example, it flags when a function expects a `pd.DataFrame` but receives a `list`, or when a return type does not match the annotation.

**Install:** Ctrl+Shift+X → search "Pylance" → Install

---

## 16. GitLens (VS Code)

**What it is:**
GitLens is a VS Code extension that supercharges Git integration. It shows inline blame annotations, commit history, and file-level comparisons.

**How it is used here:**
Lets you see who changed which line and when, view the full commit history of any file, and compare branches — all without leaving VS Code.

**Install:** Ctrl+Shift+X → search "GitLens" → Install

---

## Full Tool Summary Table

| # | Tool | Category | Version | Status |
|---|---|---|---|---|
| 1 | Python 3 | Language | 3.11.9 | Used throughout |
| 2 | Visual Studio Code | IDE | Latest | Development environment |
| 3 | Git for Windows | Version Control | 2.53.0 | 3 commits, pushed to GitHub |
| 4 | GitHub Account | Code Hosting | - | github.com/PRASADGAV/POE_MLops |
| 5 | Java JDK | Runtime | 21.0.11 LTS | Required for Jenkins |
| 6 | Jenkins | CI/CD | Latest | 6-stage pipeline, all green |
| 7 | FastAPI | Web Framework | 0.115+ | api.py — 4 endpoints |
| 8 | Uvicorn | ASGI Server | 0.31+ | Serves FastAPI on port 8000 |
| 9 | Pytest | Testing | 8.3.3 | 19 tests, 19 passed |
| 10 | HTTPX | HTTP Client | 0.28.1 | 12 API tests via TestClient |
| 11 | Evidently | ML Monitoring | 0.7.19 | HTML drift report generated |
| 12 | Pandas | Data Library | 2.2.2 | Used in all pipeline stages |
| 13 | MLflow | Experiment Tracking | 3.x | Experiment logged to SQLite |
| 14 | Python Extension Pack | VS Code Extension | Latest | Install from marketplace |
| 15 | Pylance | VS Code Extension | Latest | Install from marketplace |
| 16 | GitLens | VS Code Extension | Latest | Install from marketplace |

---

## CI/CD Flow Summary

```
You write code in VS Code
        |
        v
git commit + git push  -->  github.com/PRASADGAV/POE_MLops
                                        |
                                        v
                              Jenkins polls GitHub
                                        |
                                        v
                            [Stage 1]  Checkout code
                                        |
                            [Stage 2]  python -m venv + pip install -r requirements.txt
                                        |
                            [Stage 3]  py_compile (lint check all .py files)
                                        |
                            [Stage 4]  pytest tests/test_pipeline.py  (7/7 pass)
                                        |
                            [Stage 5]  pytest tests/test_api.py       (12/12 pass)
                                        |
                            [Stage 6]  python main.py
                                       - trains RandomForest
                                       - generates Evidently HTML report
                                       - logs run to MLflow
                                        |
                            [Post]     Archive outputs/ as Jenkins artifacts
```

---

*Generated for: California Housing MLOps POE — PRASADGAV/POE_MLops*
