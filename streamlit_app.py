# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
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
    X_new = data.drop(columns=["node", "time"], errors='ignore').values  # remove non-feature columns

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
        # Ensure node names are strings and stripped
        data["node"] = data["node"].astype(str).str.strip()
        leak_nodes = data.loc[data["predicted_leak"] == 1, "node"].tolist()
        leak_nodes = [n.strip() for n in leak_nodes]
    
        # Only keep leak nodes that exist in the WNTR network
        valid_leaks = [n for n in leak_nodes if n in net.node_name_list]
        st.write(f"Predicted leak nodes (valid): {valid_leaks}")
        st.write(f"Network nodes count: {len(net.node_name_list)}")
    
        if valid_leaks:
            # Map *all* nodes to default color first
            node_colors = {node: 'skyblue' for node in net.node_name_list}
    
            # Highlight predicted leak nodes in red
            for node in valid_leaks:
                node_colors[node] = 'red'
    
            # Plot network
            fig, ax = plt.subplots(figsize=(10, 7))
            wntr.graphics.plot_network(
                net,
                node_attribute=node_colors,
                node_size=50,
                link_width=1.2,
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.warning("No predicted leak nodes match the network nodes. Nothing to highlight.")
