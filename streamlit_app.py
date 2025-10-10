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
        # ensure node column exists and is string
        data["node"] = data["node"].astype(str).str.strip()
    
        # --- Aggregate leak predictions per node ---
        agg_leaks = data.groupby("node")["predicted_leak"].max().reset_index()
        st.write("Aggregated leak status per node (after combining duplicates):")
        st.dataframe(agg_leaks.head())
    
        # Create list of nodes predicted to have leaks
        leak_nodes_raw = agg_leaks.loc[agg_leaks["predicted_leak"] == 1, "node"].tolist()
        st.write("Final leak nodes after aggregation:", leak_nodes_raw)
    
        # network node list
        net_nodes = list(net.node_name_list)
        st.write(f"Network nodes count: {len(net_nodes)}")
        st.write("Example network nodes:", net_nodes[:10])
    
        # exact matches first
        valid_leaks = [n for n in leak_nodes_raw if n in net_nodes]
        unmatched = [n for n in leak_nodes_raw if n not in net_nodes]
    
            

        # create a numeric attribute for ALL nodes (0 = normal, 1 = leak)
        node_attr = pd.Series(0.0, index=net_nodes, dtype=float)
        for n in valid_leaks:
            node_attr.loc[n] = 1.0


        
        # quick sanity checks (will show in Streamlit)
        st.write("Number of nodes flagged as leak (in network):", int((node_attr == 1.0).sum()))
        st.write("Sample of nodes flagged:", node_attr[node_attr == 1.0].index.tolist()[:50])
    
        # Plot with numeric node_attribute and a two-color colormap
        fig, ax = plt.subplots(figsize=(10, 7))

        st.write(type(node_attr))
        st.write(len(node_attr))
        st.write(list(node_attr.items())[:10])

        
        wntr.graphics.plot_network(
            net,
            node_attribute=node_attr,        # numeric Series {nodeid: value}
            node_size=50,
            node_cmap=['lightgray', 'red'],  # map 0->lightgray, 1->red
            node_range=[0, 1],
            link_width=1.2,
            show_plot=False,                 # avoid plt.show(); we will use st.pyplot
            ax=ax
        )
    
        # add a simple legend
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color='red', label='Predicted leak'),
            mpatches.Patch(color='lightgray', label='Normal')
        ]
        ax.legend(handles=handles, loc='upper right')
    
        st.pyplot(fig)
