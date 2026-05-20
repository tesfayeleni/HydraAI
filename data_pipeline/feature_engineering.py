import pandas as pd
import numpy as np

df = pd.read_csv("dataset_labeled.csv")

# ---------------------------------------------------
# STEP 1: sort properly
# ---------------------------------------------------
df = df.sort_values(["scenario_id", "node", "time"])

# ---------------------------------------------------
# STEP 2: REMOVE cross-node simultaneity (IMPORTANT FIX)
# ---------------------------------------------------
df = df.groupby(["scenario_id", "time"]).sample(1, random_state=42)

# ---------------------------------------------------
# STEP 3: rolling features (PAST ONLY)
# ---------------------------------------------------

WINDOW = 6

pressure_roll_mean = df.groupby(
    ["scenario_id", "node"]
)["pressure"].transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())

pressure_roll_std = df.groupby(
    ["scenario_id", "node"]
)["pressure"].transform(lambda x: x.rolling(WINDOW, min_periods=1).std())

pressure_roll_min = df.groupby(
    ["scenario_id", "node"]
)["pressure"].transform(lambda x: x.rolling(WINDOW, min_periods=1).min())

pressure_roll_max = df.groupby(
    ["scenario_id", "node"]
)["pressure"].transform(lambda x: x.rolling(WINDOW, min_periods=1).max())

# ---------------------------------------------------
# STEP 4: lag features (CRITICAL)
# ---------------------------------------------------

pressure_lag_1 = df.groupby(
    ["scenario_id", "node"]
)["pressure"].shift(1)

pressure_lag_3 = df.groupby(
    ["scenario_id", "node"]
)["pressure"].shift(3)

# ---------------------------------------------------
# STEP 5: short-term dynamics
# ---------------------------------------------------

pressure_diff_1 = df.groupby(
    ["scenario_id", "node"]
)["pressure"].diff()

pressure_diff_3 = df.groupby(
    ["scenario_id", "node"]
)["pressure"].diff(3)

# ---------------------------------------------------
# STEP 6: DELTA FEATURES (ONLY CHANGE SIGNALS)
# ---------------------------------------------------

pressure_delta_from_roll = df["pressure"] - pressure_roll_mean
pressure_delta_from_lag_1 = df["pressure"] - pressure_lag_1
pressure_delta_from_lag_3 = df["pressure"] - pressure_lag_3

# ---------------------------------------------------
# STEP 7: FINAL DATASET
# ---------------------------------------------------

df_features = pd.DataFrame({
    "scenario_id": df["scenario_id"],
    "node": df["node"],
    "time": df["time"],

    "pressure_roll_mean": pressure_roll_mean,
    "pressure_roll_std": pressure_roll_std,
    "pressure_roll_min": pressure_roll_min,
    "pressure_roll_max": pressure_roll_max,

    "pressure_lag_1": pressure_lag_1,
    "pressure_lag_3": pressure_lag_3,

    "pressure_diff_1": pressure_diff_1,
    "pressure_diff_3": pressure_diff_3,

    "pressure_delta_from_roll": pressure_delta_from_roll,
    "pressure_delta_from_lag_1": pressure_delta_from_lag_1,
    "pressure_delta_from_lag_3": pressure_delta_from_lag_3,

    "future_leak_4h_horizon": df["future_leak_4h_horizon"]
})

# ---------------------------------------------------
# STEP 8: cleanup
# ---------------------------------------------------
df_features = df_features.fillna(0)

df_features.to_csv("dataset_features.csv", index=False)

print("Feature engineering complete:", df_features.shape)