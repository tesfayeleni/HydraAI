# 💧 HydraAI — Probabilistic Early Warning System for Water Network Leak Detection

---

## 🚨 Problem Statement

Water distribution networks are large, complex, and highly sensitive infrastructure systems. Even small leaks can lead to:

- Massive water loss over time  
- Increased operational and maintenance costs  
- Reduced system reliability and service quality  
- Severe impact in water-scarce regions (e.g., Middle East, North Africa)

Traditional monitoring systems are reactive, detecting leaks only after pressure drops become visible or physical damage occurs.

---

### 🔄 HydraAI reframes this problem as:

> A probabilistic early-warning forecasting problem

**“At any time t, what is the probability that a leak will occur within the next 4 hours?”**

This enables **preemptive maintenance instead of reactive repair**.

---

## 🧠 HydraAI Solution Overview

HydraAI is a full machine learning pipeline that simulates, labels, learns, and predicts leak risk in water distribution networks using physics-based simulations + temporal ML.

### Core Idea

Instead of detecting leaks after they happen, HydraAI learns:

- Subtle pre-failure pressure dynamics  
- Temporal degradation signals across nodes  
- Spatial inconsistencies in hydraulic behavior  

and converts them into a **continuous risk score over time**.

---

## 🏗 System Architecture

### 1. Physics-Based Data Generation (WNTR Simulation)

We use the EPA WNTR simulator to generate synthetic leak scenarios:

- Each scenario = leak at a specific node + leak size  
- Multiple scenarios created across:
  - All junction nodes  
  - Multiple leak severities  

Each simulation produces time-series pressure data.

📌 **Output (`dataset.csv`) contains:**

- time  
- node  
- pressure  
- leak_node  
- leak_size  
- scenario_id  
- leak_start_time  

---

### 2. Probabilistic Labeling (4-hour prediction horizon)

We convert the problem into a sliding window forecasting task:

> Label = 1 if a leak will occur within the next 4 hours at time *t*

This creates:

- `future_leak_4h_horizon ∈ {0, 1}`

Transforms raw simulation data into a predictive maintenance dataset.

---

### 3. Feature Engineering (Temporal + Local Dynamics)

HydraAI uses past-only information to avoid leakage and simulate real-world deployment conditions.

#### Temporal Features

- Rolling mean (pressure trend)  
- Rolling std (instability)  
- Rolling min/max (extremes)  
- lag_1, lag_3 (short-term memory)  
- diff_1, diff_3 (pressure change velocity)  

#### Key Design Choice (Important)

We removed:

- Global mean pressure  
- Global z-score features  

to avoid data leakage from full-network knowledge.

Instead, the model learns:

> “How pressure changes locally over time at each node”

---

### 4. Model Training (XGBoost Classifier)

We use XGBoost because:

- Strong performance on tabular + engineered features  
- Handles non-linear temporal interactions well  
- Robust to noisy simulation data  
- Efficient for large-scale datasets  

📌 Output:
- Probability of leak occurring within next 4 hours

---

### 5. Decision System (Risk Over Time)

Instead of binary classification, HydraAI outputs:

> Continuous leak probability per node over time

Operational logic:

- Sustained risk > threshold (e.g., 0.25)  
- Ignore single spikes  
- Trigger early warning alerts  

This aligns with real utility monitoring systems.

---

## 📊 Model Performance

Example evaluation results:

- ROC-AUC: ~0.88–0.89  
- Recall (leak detection): ~0.78–0.88  
- Precision: ~0.10–0.12 (expected in early-warning systems)

### Interpretation

This is intentional:

- High recall → catch most potential leaks  
- Low precision → acceptable false alarms in water-critical regions  

> In water utilities, missing a leak is far worse than a false alarm.

---

## 🧪 Streamlit Application

HydraAI includes a real-time monitoring dashboard:

### Features

- Upload simulation or sensor data  
- Predict leak probability per node  
- Generate risk-over-time curves  
- Visualize water network using WNTR  
- Highlight high-risk nodes in red  

### Outputs

- Node-level risk heatmap  
- Temporal risk evolution graph  
- Network visualization overlay (WNTR graph)  

---

## 🗺 Visualization Layer

Built using:

- `wntr.graphics` (network topology)  
- Matplotlib overlays  
- Node-level risk aggregation  

Enables engineers to:

> Visually identify vulnerable zones in the network in real time

---

## ⚙ Tech Stack

- Python (core pipeline)  
- WNTR (hydraulic simulation)  
- Pandas / NumPy (feature engineering)  
- XGBoost (ML model)  
- Scikit-learn (evaluation)  
- Streamlit (deployment UI)  
- Matplotlib (visualization)  
- Joblib (model persistence)  

---

## ⚙ Installation

### 1. Clone the repository
```bash
git clone https://github.com/tesfayeleni/HydraAI.git
cd HydraAI
```

### 2. Create and activate a virtual environment (recommended)

**Mac/Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. If you run into dependency issues
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the pipeline

```bash
python generate_dataset.py
python label_dataset.py
python feature_engineering.py
python XGBoost.py
streamlit run streamlit_app.py
```

---


## 🧪 Key Design Insight

HydraAI is not a classification problem.

It is:

> A spatiotemporal survival forecasting system over hydraulic networks

---

## 📉 Limitations

- Synthetic data (WNTR simulation, not real utility data)  
- Node-level modeling only (pipe-level extension future work)  
- Assumes known leak start distributions during training  
- Feature sensitivity depends on simulation realism  

---

## 🚀 Future Work (Research Direction)

HydraAI is designed to evolve into a publishable system:

### 1. Survival Analysis Extension
Model time-to-leak probability distributions instead of binary labels

### 2. Graph Neural Networks (GNNs)
Capture:
- Full network dependencies  
- Pressure propagation dynamics  

### 3. Real Utility Integration
Train on:
- SCADA sensor data  
- Real municipal pipelines  

### 4. Uncertainty Quantification
- Bayesian XGBoost or ensembles  
- Confidence intervals for predictions  

### 5. Edge Deployment
- Real-time monitoring system for utilities  
- Field engineer alerting system  

---

## 📚 References

- WNTR Python Library  
- Scikit-learn Documentation  
- XGBoost Documentation  