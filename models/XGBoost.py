import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset_features.csv")

# -----------------------------
# Define features and label
# -----------------------------
target = "future_leak_4h_horizon"

drop_cols = [
    "future_leak_4h_horizon",
    "future_leak",
    "scenario_id",
    "node",
    "leak_node",
    "time"
]

features = [
    "pressure_roll_mean",
    "pressure_roll_std",
    "pressure_roll_min",
    "pressure_roll_max",

    "pressure_lag_1",
    "pressure_lag_3",

    "pressure_diff_1",
    "pressure_diff_3",

    "pressure_delta_from_roll",
    "pressure_delta_from_lag_1",
    "pressure_delta_from_lag_3"
]

X = df[features]
y = df[target]

# -----------------------------
# Train-test split
# Avoid leakage by scenario
# -----------------------------
train_idx = df["scenario_id"] < df["scenario_id"].max() * 0.8

X_train, X_test = X[train_idx], X[~train_idx]
y_train, y_test = y[train_idx], y[~train_idx]

# -----------------------------
# XGBoost model
# -----------------------------
model = XGBClassifier(
    scale_pos_weight=10,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

# print("Label distribution:")
# print(df["future_leak_4h_horizon"].value_counts(normalize=True))
# print(df.groupby("future_leak_4h_horizon")["pressure_drop_from_baseline"].describe())

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predict probabilities
# -----------------------------
threshold = 0.25
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > threshold).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model + feature schema
# -----------------------------
joblib.dump(model, "xgb_model.pkl")
joblib.dump(features, "feature_list.pkl")

print("Model saved: xgb_model.pkl")
print("Feature list saved: feature_list.pkl")