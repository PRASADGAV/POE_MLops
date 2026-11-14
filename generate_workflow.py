"""Generate a full MLOps workflow diagram — saved to project root as workflow_diagram.png"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import matplotlib.lines as mlines

fig = plt.figure(figsize=(22, 30))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis("off")
fig.patch.set_facecolor("#0d0f1a")
ax.set_facecolor("#0d0f1a")

# ── Colour palette ─────────────────────────────────────────────────────────
BG       = "#0d0f1a"
PANEL    = "#13162a"
ACCENT   = "#4466ff"

COL = {
    "tools":    ("#1a2a4a", "#3a7bd5"),   # blue
    "data":     ("#1a3a2a", "#2ea86a"),   # green
    "pre":      ("#2a1a4a", "#8855ee"),   # purple
    "model":    ("#3a1a2a", "#dd4488"),   # pink
    "mlflow":   ("#3a2a0a", "#dd9922"),   # amber
    "evidently":("#0a2a3a", "#22aadd"),   # cyan
    "api":      ("#1a3a1a", "#44cc44"),   # lime
    "jenkins":  ("#3a1a1a", "#ee4444"),   # red
    "output":   ("#2a2a1a", "#cccc22"),   # yellow
    "git":      ("#2a1a3a", "#aa55ff"),   # violet
}

# ── Helpers ────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, face, edge, radius=0.3, lw=2.0, alpha=0.95, zorder=3):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge,
                       linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)

def txt(ax, x, y, s, size=10, color="#f0f0ff", weight="bold",
        ha="center", va="center", zorder=6):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#00000088")])

def arr(ax, x1, y1, x2, y2, color="#6688cc", lw=1.8, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=16), zorder=2)

def section(ax, x, y, w, h, key, title, subtitle=""):
    face, edge = COL[key]
    rbox(ax, x, y, w, h, face, edge, radius=0.45, lw=2.5)
    txt(ax, x, y + h*0.2, title, size=12, color="#ffffff")
    if subtitle:
        txt(ax, x, y - h*0.22, subtitle, size=8.5, color="#aabbcc", weight="normal")

def pill(ax, x, y, w, h, key, label, size=8):
    face, edge = COL[key]
    rbox(ax, x, y, w, h, face, edge, radius=0.2, lw=1.4, alpha=0.9)
    txt(ax, x, y, label, size=size, color="#ddeeff")

def divider(ax, y, color="#222244"):
    ax.axhline(y, xmin=0.01, xmax=0.99, color=color, lw=0.8, zorder=1)

# ══════════════════════════════════════════════════════════════════════
# TITLE BANNER
# ══════════════════════════════════════════════════════════════════════
rbox(ax, 11, 29.1, 21.5, 1.6, "#181b30", "#4466ff", radius=0.5, lw=3)
txt(ax, 11, 29.35, "California Housing  -  Complete MLOps Pipeline", size=19, color="#ffffff")
txt(ax, 11, 28.88,
    "Python 3  |  Git  |  GitHub  |  Jenkins  |  FastAPI  |  Uvicorn  |  Pytest  |  HTTPX  |  Evidently  |  Pandas  |  MLflow",
    size=9.5, color="#8899dd", weight="normal")

# ══════════════════════════════════════════════════════════════════════
# ROW 1  —  TOOLS & ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 27.6, 21, 1.5, "tools",
        "ENVIRONMENT  -  Tools Installed",
        "All tools verified before pipeline starts")

tools = ["Python 3.11", "VS Code", "Git 2.53", "GitHub Repo\nPRASADGAV/POE_MLops",
         "Java JDK 21", "Jenkins\n(port 8081)", "Pylance", "GitLens"]
for i, t in enumerate(tools):
    pill(ax, 1.5 + i*2.72, 27.0, 2.4, 0.72, "tools", t, size=8)

arr(ax, 11, 26.85, 11, 26.35)

# ══════════════════════════════════════════════════════════════════════
# ROW 2  —  GIT / GITHUB
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 25.9, 21, 1.4, "git",
        "GIT + GITHUB  -  Version Control",
        "git init  ->  git add  ->  git commit  ->  git push  ->  github.com/PRASADGAV/POE_MLops")

git_items = ["git init\n(local repo)", "3 commits\npushed", "Branch: main", ".gitignore\nconfigured",
             "GitHub remote\norigin set", "Jenkins pulls\nfrom GitHub"]
for i, g in enumerate(git_items):
    pill(ax, 2.0 + i*3.6, 25.3, 3.1, 0.72, "git", g, size=8)

arr(ax, 11, 25.2, 11, 24.65)

# ══════════════════════════════════════════════════════════════════════
# ROW 3  —  DATA
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 24.2, 21, 1.4, "data",
        "DATA SOURCE  -  carshare_mlops/data.py",
        "sklearn.datasets.fetch_california_housing()  |  20,640 rows  |  0 missing values  |  no CSV download")

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
for i, f in enumerate(features):
    pill(ax, 1.5 + i*2.72, 23.6, 2.4, 0.52, "data", f, size=8.5)

rbox(ax, 19.8, 23.6, 2.5, 0.52, "#3a0a0a", "#ee5555", radius=0.2, lw=1.4)
txt(ax, 19.8, 23.6, "TARGET\nMedHouseVal", size=7.5, color="#ffaaaa")

arr(ax, 11, 23.35, 11, 22.8)

# ══════════════════════════════════════════════════════════════════════
# ROW 4  —  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 22.4, 21, 1.4, "pre",
        "PREPROCESSING  -  carshare_mlops/data.py",
        "Split first  ->  threshold from training only  ->  binary label  ->  synthetic shift for drift")

pre = ["train_test_split\n80% / 20%  rs=42",
       "Threshold = median\nMedHouseVal on train\n= 1.7985",
       "high_price = 1\nif MedHouseVal\n>= threshold",
       "X_train 16,512\nX_test   4,128\ny_train / y_test",
       "make_synthetic_shift\n8 feature offsets\nfor Evidently demo"]
for i, p in enumerate(pre):
    pill(ax, 1.9 + i*4.3, 21.82, 3.8, 0.88, "pre", p, size=7.8)

arr(ax, 11, 21.4, 11, 20.85)

# ══════════════════════════════════════════════════════════════════════
# ROW 5  —  MODEL  (left) + EVALUATE (right)
# ══════════════════════════════════════════════════════════════════════
section(ax, 5.5, 20.4, 9.5, 1.4, "model",
        "MODEL TRAINING",
        "carshare_mlops/model.py")

params = ["n_estimators\n= 300", "max_depth\n= 10", "min_samples\n_leaf = 2", "random_state\n= 42"]
for i, p in enumerate(params):
    pill(ax, 1.5 + i*2.3, 19.82, 2.0, 0.72, "model", p, size=7.8)

section(ax, 16.5, 20.4, 9.5, 1.4, "model",
        "EVALUATION METRICS",
        "accuracy  |  precision  |  recall  |  F1  |  confusion matrix")

metrics = ["Accuracy\n88.25%", "Precision\n88.72%", "Recall\n87.46%", "F1 Score\n88.09%"]
for i, m in enumerate(metrics):
    pill(ax, 12.4 + i*2.3, 19.82, 2.0, 0.72, "model", m, size=7.8)

# train -> eval arrow
arr(ax, 10.3, 20.4, 11.7, 20.4, color="#cc66aa")

arr(ax, 11, 19.45, 11, 18.85)

# ══════════════════════════════════════════════════════════════════════
# ROW 6  —  MLFLOW  (left) + EVIDENTLY (right)
# ══════════════════════════════════════════════════════════════════════
# split
arr(ax, 11, 18.85, 5.5, 18.4, color=COL["mlflow"][1])
arr(ax, 11, 18.85, 16.5, 18.4, color=COL["evidently"][1])

section(ax, 5.5, 17.9, 9.5, 1.4, "mlflow",
        "MLFLOW TRACKING",
        "carshare_mlops/tracking.py  |  sqlite:///mlflow.db  |  port 5000")

mf = ["Experiment\nCaliforniaHousing\nHighPrice_Class",
      "log_params\nhyperparams\n+ dataset info",
      "log_metrics\nacc, prec\nrecall, F1",
      "log_model\nRF + signature\nskops trusted",
      "log_artifact\nHTML + JSON\n+ CSV"]
for i, m in enumerate(mf):
    pill(ax, 1.5 + i*1.9, 17.3, 1.65, 0.85, "mlflow", m, size=7.2)

section(ax, 16.5, 17.9, 9.5, 1.4, "evidently",
        "EVIDENTLY MONITORING",
        "carshare_mlops/monitoring.py  |  DataDriftPreset  |  8 features")

ev = ["Reference\ntest split\n(real data)",
      "Current batch\nsynthetic\nshifted",
      "DataDriftPreset\nstatistical\ntest per feature",
      "Drift detected\non all 8\nfeatures",
      "Save HTML\nevidently_housing\n_data_drift.html"]
for i, e in enumerate(ev):
    pill(ax, 11.9 + i*1.9, 17.3, 1.65, 0.85, "evidently", e, size=7.2)

arr(ax, 5.5, 17.2, 5.5, 16.65, color=COL["mlflow"][1])
arr(ax, 16.5, 17.2, 16.5, 16.65, color=COL["evidently"][1])

# ══════════════════════════════════════════════════════════════════════
# ROW 7  —  FASTAPI  
# ══════════════════════════════════════════════════════════════════════
arr(ax, 11, 17.2, 11, 16.65, color=COL["api"][1])

section(ax, 11, 16.2, 21, 1.4, "api",
        "FASTAPI + UVICORN  -  api.py  |  port 8000",
        "Model loaded at startup via lifespan  |  Uvicorn ASGI server  |  Interactive docs at /docs")

endpoints = ["GET /\nHealth check\nstatus: ok",
             "GET /info\nModel metadata\n8 features listed",
             "POST /predict\nSingle district\nreturns label + prob",
             "POST /predict/batch\nUp to 500 districts\nbatch results",
             "Swagger UI\nhttp://127.0.0.1\n:8000/docs"]
for i, e in enumerate(endpoints):
    pill(ax, 1.9 + i*4.3, 15.62, 3.8, 0.88, "api", e, size=7.8)

arr(ax, 11, 15.2, 11, 14.65)

# ══════════════════════════════════════════════════════════════════════
# ROW 8  —  PYTEST + HTTPX
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 14.2, 21, 1.4, "data",
        "PYTEST + HTTPX  -  tests/  (19 tests total)",
        "test_pipeline.py (7 tests)  +  test_api.py (12 HTTPX tests via TestClient)  -  all pass")

tests = ["Dataset schema\n& row count\n(20640 rows)",
         "Feature range\nvalidation\n(lat/lon/age)",
         "Threshold from\ntrain only\n(no leakage)",
         "Split\nreproducibility\n(rs=42)",
         "RF metrics\nall 0.0-1.0\n(validity)",
         "Drift batch\nnon-mutating\n(deterministic)",
         "Evidently\nreport created\n(> 1KB)",
         "API health\nGET /\n(200 ok)",
         "API info\nmodel metadata\n(8 features)",
         "Predict single\nhigh income\n-> high_price=1",
         "Predict low\nincome\n-> high_price=0",
         "Batch predict\n3 districts\ncount=3"]
cols_per_row = 6
for i, t in enumerate(tests):
    row = i // cols_per_row
    col = i % cols_per_row
    px = 2.0 + col * 3.6
    py = 13.62 - row * 1.0
    pill(ax, px, py, 3.1, 0.82, "data", t, size=7.2)

arr(ax, 11, 12.55, 11, 12.0)

# ══════════════════════════════════════════════════════════════════════
# ROW 9  —  JENKINS
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 11.55, 21, 1.4, "jenkins",
        "JENKINS CI  -  Jenkinsfile  |  http://localhost:8081",
        "Triggered by push to github.com/PRASADGAV/POE_MLops  |  6 pipeline stages  |  JUnit reports archived")

stages = ["Stage 1\nCheckout\nfrom GitHub",
          "Stage 2\nSetup venv\ninstall deps",
          "Stage 3\nLint\npy_compile x6",
          "Stage 4\nUnit Tests\nPipeline 7/7",
          "Stage 5\nUnit Tests\nAPI 12/12",
          "Stage 6\nRun ML\nPipeline",
          "Post\nArchive\nArtifacts"]
for i, s in enumerate(stages):
    face = "#1a0a0a" if i < 6 else "#0a1a0a"
    edge = "#ee4444" if i < 6 else "#44ee44"
    rbox(ax, 1.7 + i*2.95, 10.95, 2.6, 0.82, face, edge, radius=0.2, lw=1.6)
    txt(ax, 1.7 + i*2.95, 10.95, s, size=7.8, color="#ffdddd" if i < 6 else "#ddffdd")

# stage arrows
for i in range(6):
    arr(ax, 2.98 + i*2.95, 10.95, 3.1 + i*2.95, 10.95, color="#ee4444", lw=1.4)

arr(ax, 11, 10.5, 11, 9.95)

# ══════════════════════════════════════════════════════════════════════
# ROW 10  —  OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 9.5, 21, 1.4, "output",
        "OUTPUT ARTIFACTS  -  outputs/  +  mlartifacts/",
        "All files generated by main.py and archived by Jenkins")

outs = ["eda_summary\n.json",
        "classification\n_metrics.json",
        "test_predictions\n.csv",
        "synthetic_current\n_features.csv",
        "evidently_housing\n_data_drift.html",
        "mlflow.db\nSQLite store",
        "mlartifacts/\nmodel + logs"]
for i, o in enumerate(outs):
    pill(ax, 1.7 + i*3.1, 8.9, 2.7, 0.78, "output", o, size=7.8)

arr(ax, 11, 8.5, 11, 7.95)

# ══════════════════════════════════════════════════════════════════════
# ROW 11  —  VIEW RESULTS
# ══════════════════════════════════════════════════════════════════════
section(ax, 11, 7.5, 21, 1.4, "api",
        "VIEW RESULTS",
        "Open these in your browser after running the pipeline")

views = [
    ("MLflow UI\nhttp://127.0.0.1:5000", "mlflow"),
    ("FastAPI Docs\nhttp://127.0.0.1:8000/docs", "api"),
    ("Evidently Report\noutputs/evidently_housing\n_data_drift.html", "evidently"),
    ("Jenkins Dashboard\nhttp://localhost:8081", "jenkins"),
]
for i, (label, key) in enumerate(views):
    pill(ax, 2.8 + i*5.3, 6.9, 4.6, 0.95, key, label, size=8.5)

# ══════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════
rbox(ax, 11, 5.8, 21, 1.5, "#111320", "#333366", radius=0.4, lw=1.5)
txt(ax, 11, 6.38, "Legend", size=9.5, color="#8899cc")

legend = [
    ("tools",     "Environment"),
    ("git",       "Git / GitHub"),
    ("data",      "Data / Tests"),
    ("pre",       "Preprocessing"),
    ("model",     "Model"),
    ("mlflow",    "MLflow"),
    ("evidently", "Evidently"),
    ("api",       "FastAPI"),
    ("jenkins",   "Jenkins CI"),
    ("output",    "Outputs"),
]
for i, (key, name) in enumerate(legend):
    face, edge = COL[key]
    lx = 1.3 + i * 2.1
    rbox(ax, lx, 5.42, 0.4, 0.4, face, edge, radius=0.1, lw=1.2)
    txt(ax, lx + 0.6, 5.42, name, size=7.8, color="#ccccee", weight="normal", ha="left")

# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
rbox(ax, 11, 4.5, 21, 1.6, "#0f1120", "#222244", radius=0.4, lw=1.2)
txt(ax, 11, 4.9,
    "Dataset: sklearn.datasets.fetch_california_housing()  |  20,640 rows  |  8 features  |  RandomForestClassifier",
    size=8.5, color="#6677aa", weight="normal")
txt(ax, 11, 4.5,
    "Accuracy: 88.25%  |  Precision: 88.72%  |  Recall: 87.46%  |  F1: 88.09%  |  Tests: 19/19 passed",
    size=8.5, color="#6677aa", weight="normal")
txt(ax, 11, 4.1,
    "Jenkins: 6/6 stages passed  |  GitHub: github.com/PRASADGAV/POE_MLops  |  API: http://127.0.0.1:8000",
    size=8.5, color="#6677aa", weight="normal")

# confusion matrix mini
rbox(ax, 11, 3.0, 7, 1.5, "#111828", "#4455aa", radius=0.3, lw=1.5)
txt(ax, 11, 3.6, "Confusion Matrix  (Test Set  -  4,128 rows)", size=9, color="#aabbdd")
txt(ax, 9.0, 3.25, "Pred 0", size=8.5, color="#aaaacc")
txt(ax, 10.6, 3.25, "Pred 1", size=8.5, color="#aaaacc")
txt(ax, 7.9, 2.9,  "Actual 0", size=8.5, color="#aaaacc")
txt(ax, 7.9, 2.65, "Actual 1", size=8.5, color="#aaaacc")
txt(ax, 9.0, 2.9,  "1850", size=9, color="#88ffaa", weight="bold")
txt(ax, 10.6, 2.9, "228",  size=9, color="#ff8888", weight="bold")
txt(ax, 9.0, 2.65, "257",  size=9, color="#ff8888", weight="bold")
txt(ax, 10.6, 2.65,"1793", size=9, color="#88ffaa", weight="bold")

# ══════════════════════════════════════════════════════════════════════
# SAVE  —  project root
# ══════════════════════════════════════════════════════════════════════
out_path = "workflow_diagram.png"
fig.savefig(out_path, dpi=170, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none")
print(f"Saved: {out_path}")
plt.close()
