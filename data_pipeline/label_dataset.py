import pandas as pd

H = 4 * 3600

df = pd.read_csv("dataset.csv")

# -----------------------------------------
# Scenario-specific 4-hour prediction label
# -----------------------------------------
df["future_leak_4h_horizon"] = (
    (df["time"] <= df["leak_start_time"]) &
    (df["time"] + H >= df["leak_start_time"])
).astype(int)

# save
df.to_csv("dataset_labeled.csv", index=False)

print("Labeled dataset saved:", df.shape)