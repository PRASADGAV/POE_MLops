"""Generate a clean workflow diagram for the California Housing MLOps pipeline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(18, 24))
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

C = {
    "data_b":      "#2d6fa8",
    "pytest_b":    "#2ea84a",
    "model_b":     "#7c3aed",
    "mlflow_b":    "#e07c2a",
    "evidently_b": "#2a7ae0",
    "output_b":    "#e02a2a",
    "white":       "#f0f0f0",
    "gray":        "#888888",
}

def box(ax, x, y, w, h, face, edge, radius=0.35, lw=2.5, alpha=0.92):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge,
                       linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)

def label(ax, x, y, text, size=11, color="#f0f0f0", weight="bold",
          ha="center", va="center", zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#00000066")])

def arrow(ax, x1, y1, x2, y2, color="#6677aa", lw=2.0):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18),
        zorder=2)

def section_box(ax, x, y, w, h, face, edge, title, subtitle):
    box(ax, x, y, w, h, face, edge, radius=0.5, lw=3)
    label(ax, x, y + h*0.18, title, size=13, color="#ffffff", weight="bold")
    label(ax, x, y - h*0.22, subtitle, size=9.5, color="#aaaacc", weight="normal")

# ═══════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════
box(ax, 9, 23.1, 17.5, 1.5, "#1a1d2e", "#4455aa", radius=0.5, lw=3)
label(ax, 9, 23.35, "California Housing  -  MLOps Pipeline Workflow", size=17, color="#ffffff")
label(ax, 9, 22.88, "Pytest  |  MLflow  |  Evidently  |  RandomForestClassifier  |  sklearn built-in dataset",
      size=10, color="#8899cc", weight="normal")

# ═══════════════════════════════════════════════
# STEP 1 - DATA SOURCE
# ═══════════════════════════════════════════════
section_box(ax, 9, 21.4, 16, 1.4, "#1e3a5f", C["data_b"],
            "STEP 1  -  Data Source",
            "sklearn.datasets.fetch_california_housing()   |   20,640 rows   |   0 missing values")

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
for i, f in enumerate(features):
    px = 1.4 + i * 2.17
    box(ax, px, 20.3, 1.9, 0.52, "#0d2340", C["data_b"], radius=0.22, lw=1.5)
    label(ax, px, 20.3, f, size=8.5, color="#7ec8f0")

box(ax, 15.6, 20.3, 2.5, 0.52, "#2a0d0d", "#e05555", radius=0.22, lw=1.5)
label(ax, 15.6, 20.3, "TARGET: MedHouseVal", size=8, color="#f08080")

label(ax, 0.75, 21.5, "Features:", size=8.5, color=C["gray"], ha="left")
label(ax, 0.75, 20.3, "-->", size=9, color=C["gray"], ha="left")

arrow(ax, 9, 20.65, 9, 20.1)

# ═══════════════════════════════════════════════
# STEP 2 - PYTEST
# ═══════════════════════════════════════════════
section_box(ax, 9, 19.3, 16, 1.4, "#1a3a2a", C["pytest_b"],
            "STEP 2  -  Pytest  (tests/test_pipeline.py)",
            "7 automated tests validate the entire pipeline before training begins")

tests = [
    "Dataset schema\n& row count",
    "Feature value\nrange checks",
    "Threshold from\ntrain only",
    "Split\nreproducibility",
    "RF metrics\nvalidity",
    "Drift batch\nnon-mutating",
    "Evidently report\ncreated",
]
for i, t in enumerate(tests):
    px = 1.3 + i * 2.25
    box(ax, px, 18.25, 2.0, 0.78, "#0d2a16", C["pytest_b"], radius=0.25, lw=1.5)
    label(ax, px, 18.25, t, size=7.8, color="#7ef0a0")

arrow(ax, 9, 18.6, 9, 18.0)

# ═══════════════════════════════════════════════
# STEP 3 - PREPROCESSING
# ═══════════════════════════════════════════════
section_box(ax, 9, 17.3, 16, 1.4, "#2a1a3a", C["model_b"],
            "STEP 3  -  Preprocessing  (carshare_mlops/data.py)",
            "Split data first  -->  compute threshold from training partition only  -->  create binary labels")

pre_steps = [
    "train_test_split\n80% / 20%\nrandom_state=42",
    "Compute median\nMedHouseVal\non train only",
    "Label: high_price=1\nif MedHouseVal\n>= 1.7985",
    "X_train  16,512 rows\nX_test    4,128 rows\ny_train / y_test",
    "make_synthetic_shift\nfor drift demo\n(8 feature offsets)",
]
for i, s in enumerate(pre_steps):
    px = 1.9 + i * 3.55
    box(ax, px, 16.3, 3.1, 0.9, "#1a0d2a", C["model_b"], radius=0.25, lw=1.5)
    label(ax, px, 16.3, s, size=8, color="#c4a0f0")

arrow(ax, 9, 16.75, 9, 15.6)

# ═══════════════════════════════════════════════
# STEP 4 - TRAINING  (left)
# ═══════════════════════════════════════════════
section_box(ax, 5, 15.0, 7.5, 1.4, "#2a1a3a", C["model_b"],
            "STEP 4  -  Model Training",
            "carshare_mlops/model.py  -  RandomForestClassifier")

params = ["n_estimators=300", "max_depth=10", "min_samples_leaf=2", "random_state=42", "n_jobs=-1"]
for i, p in enumerate(params):
    px = 1.8 + i * 1.65
    box(ax, px, 14.0, 1.5, 0.45, "#12082a", C["model_b"], radius=0.18, lw=1.2)
    label(ax, px, 14.0, p, size=7.5, color="#d0b0ff")

# STEP 4b - EVALUATION (right)
section_box(ax, 13.5, 15.0, 7.5, 1.4, "#2a1a3a", "#a060f0",
            "STEP 4b  -  Evaluation",
            "accuracy  |  precision  |  recall  |  F1  |  confusion matrix")

metrics_boxes = ["Accuracy\n88.25%", "Precision\n88.72%", "Recall\n87.46%", "F1 Score\n88.09%"]
for i, m in enumerate(metrics_boxes):
    px = 10.4 + i * 1.75
    box(ax, px, 14.0, 1.55, 0.72, "#1a0a3a", "#a060f0", radius=0.2, lw=1.2)
    label(ax, px, 14.0, m, size=8, color="#e0c8ff")

arrow(ax, 8.75, 15.0, 10.25, 15.0, color="#8855cc")
arrow(ax, 9, 13.65, 9, 12.9)

# ═══════════════════════════════════════════════
# SPLIT ARROWS
# ═══════════════════════════════════════════════
arrow(ax, 9, 12.9, 4.5, 12.4, color=C["mlflow_b"])
arrow(ax, 9, 12.9, 13.5, 12.4, color=C["evidently_b"])

# ═══════════════════════════════════════════════
# STEP 5a - MLFLOW
# ═══════════════════════════════════════════════
section_box(ax, 4.5, 11.75, 8, 1.3, "#3a2a1a", C["mlflow_b"],
            "STEP 5a  -  MLflow Tracking",
            "carshare_mlops/tracking.py  |  sqlite:///mlflow.db")

mlflow_items = [
    "Experiment\nCaliforniaHousing\nHighPrice_Class",
    "log_params\nhyperparameters\n+ dataset info",
    "log_metrics\nacc, precision\nrecall, F1",
    "log_model\nRF + signature\n+ input example",
    "log_artifact\nHTML + JSON\n+ CSV",
]
for i, it in enumerate(mlflow_items):
    px = 1.0 + i * 1.75
    box(ax, px, 10.75, 1.55, 0.88, "#2a1800", C["mlflow_b"], radius=0.2, lw=1.2)
    label(ax, px, 10.75, it, size=7.4, color="#f0c080")

# ═══════════════════════════════════════════════
# STEP 5b - EVIDENTLY
# ═══════════════════════════════════════════════
section_box(ax, 13.5, 11.75, 8, 1.3, "#1a2a3a", C["evidently_b"],
            "STEP 5b  -  Evidently Monitoring",
            "carshare_mlops/monitoring.py  |  DataDriftPreset")

ev_items = [
    "Reference data\nTest split\n(real features)",
    "Current data\nSynthetic shifted\nbatch (demo)",
    "DataDriftPreset\nstatistical tests\nper feature",
    "Drift detected\non 8 features\n(all shifted)",
    "Save HTML\nevidently_housing\n_data_drift.html",
]
for i, it in enumerate(ev_items):
    px = 9.9 + i * 1.75
    box(ax, px, 10.75, 1.55, 0.88, "#0a1a2a", C["evidently_b"], radius=0.2, lw=1.2)
    label(ax, px, 10.75, it, size=7.4, color="#80c8f0")

arrow(ax, 4.5, 10.3, 4.5, 9.65, color=C["mlflow_b"])
arrow(ax, 13.5, 10.3, 13.5, 9.65, color=C["evidently_b"])

# ═══════════════════════════════════════════════
# STEP 6 - OUTPUTS
# ═══════════════════════════════════════════════
section_box(ax, 9, 9.1, 16, 1.3, "#2a1a1a", C["output_b"],
            "STEP 6  -  Output Artifacts",
            "All files written to outputs/   |   MLflow artifacts in mlartifacts/")

outputs_list = [
    "eda_summary\n.json",
    "classification\n_metrics.json",
    "test_predictions\n.csv",
    "synthetic_current\n_features.csv",
    "evidently_housing\n_data_drift.html",
    "mlflow.db\n(SQLite store)",
    "mlartifacts/\n(model files)",
]
for i, o in enumerate(outputs_list):
    px = 1.3 + i * 2.3
    box(ax, px, 8.15, 2.0, 0.72, "#2a0a0a", C["output_b"], radius=0.2, lw=1.5)
    label(ax, px, 8.15, o, size=8, color="#f0a0a0")

arrow(ax, 9, 7.75, 9, 7.1)

# ═══════════════════════════════════════════════
# STEP 7 - VIEW RESULTS
# ═══════════════════════════════════════════════
section_box(ax, 9, 6.6, 16, 1.4, "#1a2a1a", "#44aa66",
            "STEP 7  -  View Results",
            "Inspect experiment in MLflow UI  |  Open Evidently drift report in browser")

box(ax, 4.5, 5.65, 7.5, 1.1, "#0d1f0d", "#44aa66", radius=0.3, lw=2)
label(ax, 4.5, 5.78, "MLflow UI", size=11, color="#aaffaa")
label(ax, 4.5, 5.52, "http://127.0.0.1:5000  -  runs, metrics, model artifacts", size=8.5, color="#88cc88", weight="normal")

box(ax, 13.5, 5.65, 7.5, 1.1, "#0d1a2a", "#4499cc", radius=0.3, lw=2)
label(ax, 13.5, 5.78, "Evidently HTML Report", size=11, color="#aaccff")
label(ax, 13.5, 5.52, "outputs/evidently_housing_data_drift.html  -  per-feature drift", size=8.5, color="#88aadd", weight="normal")

# ═══════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════
box(ax, 9, 4.3, 16.5, 1.5, "#13151f", "#333355", radius=0.4, lw=1.5)
label(ax, 9, 4.9, "Legend", size=10, color="#8899cc")

legend_items = [
    (C["data_b"],      "Data / Features"),
    (C["pytest_b"],    "Pytest Tests"),
    (C["model_b"],     "Model Pipeline"),
    (C["mlflow_b"],    "MLflow Tracking"),
    (C["evidently_b"], "Evidently Monitoring"),
    (C["output_b"],    "Output Artifacts"),
]
for i, (color, name) in enumerate(legend_items):
    lx = 1.6 + i * 2.65
    box(ax, lx, 4.3, 0.42, 0.42, color, color, radius=0.1, lw=1)
    label(ax, lx + 0.6, 4.3, name, size=8.5, color="#ccccdd", weight="normal", ha="left")

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
label(ax, 9, 3.4,
      "Python 3.11  |  scikit-learn  |  RandomForestClassifier  |  MLflow 3.x  |  Evidently 0.7.19  |  Pytest 8.x",
      size=8.5, color="#555577", weight="normal")
label(ax, 9, 3.0,
      "Dataset: sklearn.datasets.fetch_california_housing()  -  20,640 rows  |  Accuracy: 88.25%  |  F1: 88.09%",
      size=8.5, color="#555577", weight="normal")

# confusion matrix mini table
box(ax, 9, 2.1, 7, 1.4, "#1a1a2a", "#6666aa", radius=0.3, lw=1.5)
label(ax, 9, 2.6, "Confusion Matrix (Test Set - 4,128 rows)", size=9, color="#aaaadd")
label(ax, 7.5, 2.25, "Pred 0", size=8.5, color="#aaaacc")
label(ax, 9.0, 2.25, "Pred 1", size=8.5, color="#aaaacc")
label(ax, 6.3, 1.9,  "Actual 0", size=8.5, color="#aaaacc")
label(ax, 6.3, 1.65, "Actual 1", size=8.5, color="#aaaacc")
label(ax, 7.5, 1.9,  "1850", size=9, color="#aaffaa", weight="bold")
label(ax, 9.0, 1.9,  "228",  size=9, color="#ffaaaa", weight="bold")
label(ax, 7.5, 1.65, "257",  size=9, color="#ffaaaa", weight="bold")
label(ax, 9.0, 1.65, "1793", size=9, color="#aaffaa", weight="bold")

plt.tight_layout(pad=0)
out = "outputs/workflow_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Saved: {out}")
plt.close()
