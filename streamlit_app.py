# streamlit_app.py

# ------------------------------
# IMPORT LIBRARIES
# ------------------------------
import streamlit as st        # Streamlit web app
import pandas as pd           # Data handling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib                 # Load trained Random Forest model
import wntr                   # Water network visualization

# ------------------------------
# LOAD NETWORK WITH CACHING
# ------------------------------
@st.cache_resource
def load_network(path):
    """
    Load and cache the water network model.
    Caching avoids reloading the network on every Streamlit rerun.
    """
    return wntr.network.WaterNetworkModel(path)

net = load_network("Net3.inp")  # load the network once

# ------------------------------
# FILE UPLOADER FOR NEW DATA
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload latest sensor/network data", type=["csv"]
)

if uploaded_file is not None:
    # Read CSV and convert to NumPy array for model
    data = pd.read_csv(uploaded_file)
    X_new = data.values  

    # --------------------------
    # MAKE PREDICTIONS
    # --------------------------
    rf_model = joblib.load("rf_model.pkl")  # Load trained Random Forest
    THRESHOLD = 0.5
    y_proba = rf_model.predict_proba(X_new)[:, 1]          # probability of leak
    y_pred = (y_proba >= THRESHOLD).astype(int)           # binary prediction

    # --------------------------
    # DISPLAY PREDICTIONS
    # --------------------------
    data["predicted_leak"] = y_pred
    st.write("Predicted leaks (1 = leak):")
    st.dataframe(data)  # display predictions

    # --------------------------
    # NETWORK VISUALIZATION
    # --------------------------
    with st.spinner("Plotting network, please wait..."):

        # 1) Normalize node names
        data["node"] = data["node"].astype(str).str.strip()

        # 2) Aggregate leak predictions per node
        agg_leaks = data.groupby("node", as_index=False)["predicted_leak"].max()
        st.write("Aggregated leak status per node:")
        st.dataframe(agg_leaks)

        # 3) Identify predicted leak nodes
        leak_nodes_raw = agg_leaks.loc[agg_leaks["predicted_leak"] == 1, "node"].tolist()

        # 4) Ensure network nodes are strings for matching
        net_nodes = [str(n).strip() for n in net.node_name_list]

        # 5) Filter valid leaks that exist in the network
        valid_leaks = [n for n in leak_nodes_raw if n in net_nodes]
        unmatched = [n for n in leak_nodes_raw if n not in net_nodes]
        st.write("Valid leaks:", valid_leaks)

        # TODO: Add primary cause for each leak in future
        # TODO: Implement NLP suggestions for flagged nodes

        # 6) Build node -> leak mapping
        node_attr = {n: 0.0 for n in net_nodes}  # default all nodes to 0
        for n in valid_leaks:
            node_attr[n] = 1.0  # mark leak nodes

        st.write("Total nodes flagged as leak:", sum(v == 1.0 for v in node_attr.values()))
        # Optional: sample node_attr for debugging
        # print("Sample node_attr items:", list(node_attr.items())[:20])

        # --------------------------
        # PLOT NETWORK
        # --------------------------
        fig, ax = plt.subplots(figsize=(10, 7))
        wntr.graphics.plot_network(
            net,
            node_attribute=node_attr,        # dict {node_name: 0 or 1}
            node_size=50,
            node_cmap=['lightgray', 'red'], # 0->gray, 1->red
            node_range=[0, 1],
            link_width=1.2,
            ax=ax
        )

        # 7) Add legend for leak vs normal nodes
        handles = [
            mpatches.Patch(color='red', label='Predicted leak'),
            mpatches.Patch(color='lightgray', label='Normal')
        ]
        ax.legend(handles=handles, loc='upper right')

        # 8) Display plot in Streamlit
        st.pyplot(fig)
)
