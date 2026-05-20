import wntr
import pandas as pd
import random

# -----------------------------
# CONFIG
# -----------------------------
wn_base = wntr.network.WaterNetworkModel("Net3.inp")

nodes = list(wn_base.junction_name_list)
leak_sizes = [0.0001, 0.0005]

all_data = []
scenario_id = 0

# -----------------------------
# GENERATE DATASET
# -----------------------------
for node in nodes:
    for size in leak_sizes:

        wn = wntr.network.WaterNetworkModel("Net3.inp")

        # 🔥 RANDOMIZED leak start time (important fix)
        leak_start = random.randint(2, 8) * 3600
        leak_end = leak_start + 8 * 3600

        junction = wn.get_node(node)

        junction.add_leak(
            wn,
            area=size,
            start_time=leak_start,
            end_time=leak_end
        )

        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        pressure = results.node["pressure"]

        df = pressure.reset_index().melt(
            id_vars="index",
            var_name="node",
            value_name="pressure"
        )

        df.rename(columns={"index": "time"}, inplace=True)

        df["leak_node"] = node
        df["leak_size"] = size
        df["scenario_id"] = scenario_id

        # 🔥 IMPORTANT: store leak timing for labeling
        df["leak_start_time"] = leak_start

        all_data.append(df)

        scenario_id += 1

# -----------------------------
# FINAL DATASET
# -----------------------------
final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv("dataset.csv", index=False)

print("Dataset created:", final_df.shape)