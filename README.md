# 💧 HydraAI — Probabilistic Early Warning System for Water Network Leak Detection

> **Live Dashboard:** [hydraai.streamlit.app](https://hydraai.streamlit.app)

A full end-to-end machine learning pipeline that simulates, labels, engineers features from, trains on, and deploys predictions from hydraulic network data — reframing leak detection as a **probabilistic forecasting problem** rather than a reactive alert system.

---

## 🚨 Problem Statement

Water distribution networks are large, complex infrastructure systems where even small leaks lead to:

- Significant water loss over time, especially critical in water-scarce regions
- Cascading infrastructure degradation
- High emergency repair costs and service disruption
- Reduced system reliability for downstream consumers

Traditional monitoring is **reactive** — it detects leaks after pressure drops become visible. By then, failure has already begun.

### HydraAI reframes this as:

> *"At any time t, what is the probability that a leak will occur within the next 4 hours at a given node?"*

This enables **preemptive maintenance** instead of emergency response.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.88 – 0.89 |
| Recall (leak detection) | 0.78 – 0.88 |
| Precision | 0.10 – 0.12 |
| Prediction horizon | 4 hours |
| Default alert threshold | 0.25 |

**On precision:** Low precision is intentional and appropriate. This is a high-recall early warning system. In water infrastructure, a missed leak is catastrophically more costly than a false alarm. The system is tuned to catch failures, not minimize crew dispatches.

---

## 🏗 System Architecture

```
Physics Simulation (WNTR)
        ↓
  Dataset Generation
  [generate_dataset.py]
        ↓
  Predictive Labeling
  [label_dataset.py]
  (4-hour look-ahead window)
        ↓
  Feature Engineering
  [feature_engineering.py]
  (delta, lag, rolling statistics)
        ↓
  Model Training (XGBoost)
  [XGBoost.py]
        ↓
  Inference + Temporal Risk Engine
  [streamlit_app.py]
  (smoothing → sustained alert logic)
        ↓
  Operational Dashboard
  (network map → node drill-down)
```

---

## ⚙️ Pipeline — Stage by Stage

### 1. Physics-Based Data Generation (`generate_dataset.py`)

Uses WNTR (Water Network Tool for Resilience) to run EPANET-based hydraulic simulations on the Net3 benchmark network (97 junction nodes).

For each node × leak size combination:
- A leak is injected at a **randomized start time** (2–8 hours into simulation) to prevent the model from memorizing fixed time patterns
- The simulation records pressure at every node over time
- Multiple leak sizes (0.0001, 0.0005 m²) are tested per node

Output: `dataset.csv` — time-series pressure readings with leak metadata per scenario.

---

### 2. Predictive Labeling (`label_dataset.py`)

Defines the supervised learning target using a **sliding 4-hour look-ahead window**:

```python
df["future_leak_4h_horizon"] = (
    (df["time"] <= df["leak_start_time"]) &
    (df["time"] + H >= df["leak_start_time"])
).astype(int)
```

**Why this matters:** Labeling the leak moment as `1` produces a detection model. Labeling the 4-hour window *before* the leak produces a prediction model. This distinction is the core of the predictive maintenance framing.

Output: `dataset_labeled.csv`

---

### 3. Feature Engineering (`feature_engineering.py`)

All features are computed using strictly past-only information — no future leakage.

| Feature | Signal Captured |
|---|---|
| `pressure_roll_mean` | Trend — gradual pressure drift |
| `pressure_roll_std` | Instability — increasing fluctuation |
| `pressure_roll_min/max` | Extremes — sustained low pressure |
| `pressure_lag_1`, `pressure_lag_3` | Short-term memory |
| `pressure_diff_1`, `pressure_diff_3` | Rate of change |
| `pressure_delta_from_roll` | Anomaly relative to own recent average |
| `pressure_delta_from_lag_1/3` | Sudden relative shift |

**Design choice:** Global pressure statistics (z-scores, network-wide mean) were deliberately excluded to avoid data leakage from full-network knowledge. The model learns from *local node dynamics*, not global state — which is closer to real sensor deployment conditions.

Cross-node simultaneity is removed by sampling one node per (scenario, timestep) pair to prevent near-duplicate rows from inflating training performance.

Output: `dataset_features.csv`

---

### 4. Model Training (`XGBoost.py`)

```python
model = XGBClassifier(
    scale_pos_weight=10,   # handles severe class imbalance
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Train/test split by scenario ID** (not random row split) to prevent temporal leakage — rows within the same scenario are highly correlated, so row-level splitting inflates evaluation metrics.

`scale_pos_weight=10` addresses the rarity of leak events (~10–15% of rows), biasing the model toward recall over precision — appropriate for safety-critical monitoring.

Outputs: `xgb_model.pkl`, `feature_list.pkl`

---

### 5. Inference + Temporal Risk Engine (`streamlit_app.py`)

The app performs inference only — no retraining. Pipeline per scenario:

1. Load scenario rows from `dataset_features.csv`
2. Align features using `feature_list.pkl` (preserves exact training order)
3. Run `model.predict_proba(X)[:, 1]` → continuous probability per (node, time)
4. Apply rolling mean smoothing (configurable window, default 6)
5. Fire alert only if smoothed probability exceeds threshold for **3+ consecutive timesteps**
6. Assign alert state: `ACTIVE` / `WARNING` / `STANDBY`

---

## 🖥 Operational Dashboard

**Live at:** [hydraai.streamlit.app](https://hydraai.streamlit.app)

The dashboard is designed as an operational monitoring platform, not a data science demo. The UX follows the same logic as real utility control centers: system-wide overview first, node-level investigation second.

### Network Map View (Primary)

The landing screen shows the full WNTR hydraulic network with all nodes rendered at their actual topological positions and colored by current smoothed risk:

- 🟢 **Green** — low risk (below threshold)
- 🟡 **Yellow** — warning (above threshold, not yet sustained)
- 🔴 **Red** — active alert (sustained above threshold for 3+ steps)

A KPI bar across the top shows: Active Alerts / Warnings / Mean Risk / % Above Threshold / Highest Risk Node.

An alert queue on the right lists flagged nodes sorted by severity, each with an **Inspect** button for one-click drill-down.

A node risk distribution histogram below the map shows the full probability spread across all nodes.

### Node Drill-Down View (On Demand)

Clicking **Inspect** on any node transitions to a full-page detail view:

- **Risk over time chart** — raw probability curve, smoothed curve, threshold line, and red-shaded sustained alert regions
- **Stats panel** — current risk, peak probability, duration above threshold, first/last alert times, % time in alert, monitoring window
- Navigation returns to the network map via **← Back to Network Map**

### Sidebar Controls

| Control | Range | Default | Effect |
|---|---|---|---|
| Scenario selector | All scenarios | — | Filters the dataset to a specific simulation run |
| Node selector | All nodes | — | Pre-selects node for drill-down |
| Alert threshold | 0.10 – 0.60 | 0.25 | Sets the probability cutoff for flagging |
| Smoothing window | 3 – 10 | 6 | Controls rolling average width |

**Note:** Threshold and smoothing controls affect the decision layer only. They do not modify the model or retrain anything.

---

## 🧱 Tech Stack

| Component | Technology |
|---|---|
| Hydraulic simulation | WNTR (EPA EPANET engine) |
| Data processing | Pandas, NumPy |
| Machine learning | XGBoost, Scikit-learn |
| Dashboard | Streamlit |
| Network visualization | Plotly (interactive), Matplotlib (static fallback) |
| Model persistence | Joblib |
| Deployment | Streamlit Community Cloud |

---

## 📦 Project Structure

```
HydraAI/
├── app/
│   └── streamlit_app.py          # Inference engine + full dashboard UI
├── data_pipeline/
│   ├── generate_dataset.py       # WNTR simulation → dataset.csv
│   ├── label_dataset.py          # 4-hour look-ahead labels
│   └── feature_engineering.py   # Temporal feature construction
├── models/
│   ├── xgb_model.pkl             # Trained XGBoost model
│   ├── feature_list.pkl          # Ordered feature schema for inference
│   └── XGBoost.py                # Training script
├── data/
│   └── dataset_features.csv      # Generated by running the pipeline (see below)
├── Net3.inp                      # EPANET benchmark network
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/tesfayeleni/HydraAI.git
cd HydraAI
```

### 2. Create and activate a virtual environment

```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the data pipeline

Run these in order from the project root:

```bash
cd data_pipeline
python generate_dataset.py       # → dataset.csv
python label_dataset.py          # → dataset_labeled.csv
python feature_engineering.py    # → ../data/dataset_features.csv
```

### 5. Train the model

```bash
cd ../models
python XGBoost.py                # → xgb_model.pkl, feature_list.pkl
```

### 6. Launch the dashboard

```bash
cd ..
streamlit run app/streamlit_app.py
```

---

## 📉 Limitations

- **Synthetic data only.** The pipeline is trained on WNTR hydraulic simulations, not real SCADA sensor readings. Generalization to live utility data is not yet validated.
- **Node-level modeling.** The model treats each node independently and does not capture inter-node pressure propagation through the pipe network.
- **Single network topology.** Trained and tested on Net3 only. Performance on networks with different topology, pipe material, or demand patterns is untested.
- **No uncertainty quantification.** The model outputs a point probability, not a confidence interval.

---

## 🚀 Future Work

**1. Real utility data integration**
Replace synthetic simulations with SCADA sensor readings from an operational water network. This is the single most impactful next step for real-world validity.

**2. Graph Neural Networks**
Model inter-node hydraulic dependencies using GNNs on the pipe network topology. Pressure propagation from a leak node to its neighbors carries information the current per-node model cannot see.

**3. Survival analysis framing**
Model time-to-failure as a distribution rather than a binary 4-hour horizon. More actionable for maintenance scheduling with variable lead-time requirements.

**4. Uncertainty quantification**
Bayesian XGBoost or conformal prediction wrappers to output calibrated confidence intervals alongside point estimates.

**5. Streaming inference**
Replace CSV-based batch inference with a real-time sensor ingestion pipeline (Kafka or MQTT) for continuous monitoring on live networks.

**6. Explainability layer**
Per-node SHAP values surfaced in the dashboard so operators understand *why* a node is flagged, not just *that* it is.

---

## 📚 References

- [WNTR Documentation](https://wntr.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [EPANET — US EPA](https://www.epa.gov/water-research/epanet)
- [Scikit-learn Documentation](https://scikit-learn.org/)
