# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib  # to load your trained model
import wntr    # for network visualization

# --- LOAD NETWORK WITH CACHING ---
@st.cache_resource
def load_network(path):
    return wntr.network.WaterNetworkModel(path)

net = load_network("Net3.inp")  # load once and cache

# --- LOAD DATA ---
uploaded_file = st.file_uploader("Upload latest sensor/network data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # X_new = data.drop(columns=["node", "time"], errors='ignore').values  # remove non-feature columns

    X_new = data.values  

    
    # --- MAKE PREDICTIONS ---
    y_proba = rf_model.predict_proba(X_new)[:, 1]
    THRESHOLD = 0.5
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # --- DISPLAY PREDICTIONS ---
    st.write("Predicted leaks (1 = leak):")
    data["predicted_leak"] = y_pred
    st.dataframe(data)
    
    # --- NETWORK VISUALIZATION ---
    with st.spinner("Plotting network, please wait..."):
        # 1) Normalize node strings in your uploaded data
        data["node"] = data["node"].astype(str).str.strip()
    
        # 2) Aggregate leak predictions per node (collapse duplicates)
        agg_leaks = data.groupby("node", as_index=False)["predicted_leak"].max()
        st.write("Aggregated leak status per node (one row per node):")
        st.dataframe(agg_leaks.head(20))
    
        # 3) Quick check: counts of how many rows per node (just for debugging)
        st.write("Counts per node (first 10 rows) -- frequency, NOT leak status:")
        st.write(data["node"].value_counts().head(10))
    
        # 4) Build list of nodes predicted as leaks (one entry per node)
        leak_nodes_raw = agg_leaks.loc[agg_leaks["predicted_leak"] == 1, "node"].tolist()
        st.write("Nodes with leak==1 after aggregation:", leak_nodes_raw)
    
        # 5) Normalize network node names
        net_nodes = [str(n).strip() for n in net.node_name_list]
        st.write("Network nodes count:", len(net_nodes))
        st.write("Example network nodes:", net_nodes[:10])
    
        # 6) Find intersection between CSV leak nodes and network nodes
        valid_leaks = [n for n in leak_nodes_raw if n in net_nodes]
        unmatched = [n for n in leak_nodes_raw if n not in net_nodes]
        st.write("Valid leaks in network:", valid_leaks[:50])
        st.write("Unmatched nodes (not in network):", unmatched[:50])
    
        # 7) Build node -> leak value mapping (dict for WNTR)
        node_attr = {n: 0.0 for n in net_nodes}
        for n in valid_leaks:
            node_attr[n] = 1.0
    
        # 8) Debugging: check the node_attr dictionary before plotting
        st.write("Total nodes flagged as leak:", sum(v == 1.0 for v in node_attr.values()))
        st.write("Sample node_attr items:", list(node_attr.items())[:20])
    
        # 9) Plot the network
        fig, ax = plt.subplots(figsize=(10, 7))
        wntr.graphics.plot_network(
            net,
            node_attribute=node_attr,   # dict {node_name: 0 or 1}
            node_size=50,
            node_cmap=['lightgray', 'red'],  # 0->gray, 1->red
            node_range=[0, 1],
            link_width=1.2,
            show_plot=False,
            ax=ax
        )
    
        # 10) Add legend
        handles = [
            mpatches.Patch(color='red', label='Predicted leak'),
            mpatches.Patch(color='lightgray', label='Normal')
        ]
        ax.legend(handles=handles, loc='upper right')
    
        # 11) Show plot in Streamlit
        st.pyplot(fig)
