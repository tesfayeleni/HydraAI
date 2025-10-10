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
    # Load the water network model once and cache it for efficiency
    return wntr.network.WaterNetworkModel(path)

net = load_network("Net3.inp")  # load network file

# --- LOAD DATA ---
uploaded_file = st.file_uploader("Upload latest sensor/network data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Convert uploaded data to a NumPy array for model input
    X_new = data.values  

    # --- MAKE PREDICTIONS ---
    # Load trained Random Forest model
    rf_model = joblib.load("rf_model.pkl")  # trained RF model file
    y_proba = rf_model.predict_proba(X_new)[:, 1]  # probability of leak
    THRESHOLD = 0.5
    y_pred = (y_proba >= THRESHOLD).astype(int)  # apply threshold to get binary prediction

    # --- DISPLAY PREDICTIONS ---
    st.write("Predicted leaks (1 = leak):")
    data["predicted_leak"] = y_pred
    st.dataframe(data)  # show predictions with original data

    # --- NETWORK VISUALIZATION ---
    with st.spinner("Plotting network, please wait..."):
        # 1) Normalize node strings to ensure consistent matching
        data["node"] = data["node"].astype(str).str.strip()
        
        # 2) Aggregate leak predictions per node
        # max ensures that if any row indicates a leak, the node is flagged
        agg_leaks = data.groupby("node", as_index=False)["predicted_leak"].max()
        st.write("Aggregated leak status per node:")
        st.dataframe(agg_leaks)
    
        # 3) Identify nodes predicted as leaks
        leak_nodes_raw = agg_leaks.loc[agg_leaks["predicted_leak"] == 1, "node"].tolist()
    
        # 4) Ensure network node names are strings for matching
        net_nodes = [str(n).strip() for n in net.node_name_list]
    
        # 5) Filter valid leaks that exist in the network
        valid_leaks = [n for n in leak_nodes_raw if n in net_nodes]
        unmatched = [n for n in leak_nodes_raw if n not in net_nodes]
        st.write("Valid leaks:", valid_leaks)
        
        # TODO: In future iterations, implement logic to identify the primary cause for each leak
        # TODO: Implement automated NLP suggestions for each flagged node
    
        # 6) Build node -> leak value mapping for WNTR visualization
        node_attr = {n: 0.0 for n in net_nodes}  # default all nodes to 0
        for n in valid_leaks:
            node_attr[n] = 1.0  # mark leak nodes
    
        st.write("Total nodes flagged as leak:", sum(v == 1.0 for v in node_attr.values()))
        # Optional: sample of node_attr for debugging
        # print("Sample node_attr items:", list(node_attr.items())[:20])

        # NOTE: Current network visualization is static (non-interactive)
    
        # 7) Plot the network with leak nodes highlighted
        fig, ax = plt.subplots(figsize=(10, 7))
        wntr.graphics.plot_network(
            net,
            node_attribute=node_attr,   # dict {node_name: 0 or 1}
            node_size=50,
            node_cmap=['lightgray', 'red'],  # 0->gray, 1->red
            node_range=[0, 1],
            link_width=1.2,
            ax=ax
        )
    
        # 8) Add legend for leak vs normal nodes
        handles = [
            mpatches.Patch(color='red', label='Predicted leak'),
            mpatches.Patch(color='lightgray', label='Normal')
        ]
        ax.legend(handles=handles, loc='upper right')
    
        # 9) Show plot in Streamlit app
        st.pyplot(fig)
)
