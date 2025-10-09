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

#--- LOAD MODEL ---
rf_model = joblib.load("rf_model.pkl")  # save after training using joblib.dump(rf_model, 'rf_model.pkl')

# --- LOAD DATA ---
uploaded_file = st.file_uploader("Upload latest sensor/network data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X_new = data.values  # assume proper preprocessing is already done
    
    # --- MAKE PREDICTIONS ---
    y_proba = rf_model.predict_proba(X_new)[:, 1]
    THRESHOLD = 0.5
    y_pred = (y_proba >= THRESHOLD).astype(int)
    
    # --- DISPLAY PREDICTIONS ---
    st.write("Predicted leaks (1 = leak):")
    data["predicted_leak"] = y_pred
    st.dataframe(data)


    
    # --- SHAP EXPLANATION ---
   # background = X_new[np.random.choice(X_new.shape[0], min(100, X_new.shape[0]), replace=False)]
    #explainer = shap.TreeExplainer(rf_model, data=background)
   # shap_values = explainer(X_new)
    
   # st.write("Feature importance (global):")
   # fig, ax = plt.subplots()
   # shap.summary_plot(shap_values.values, X_new, feature_names=data.columns, plot_type="bar", show=False)
#    st.pyplot(fig)


    
    # Optional: force plot for first flagged leak
    #idx = np.where(y_pred == 1)[0]
    #if len(idx) > 0:
        #i = idx[0]
       # st.write(f"SHAP explanation for leak at index {i}:")
       # shap_html = shap.force_plot(
        #    base_value=shap_values.base_values[i, 1],
         #   shap_values=shap_values.values[i, :, 1],
          #  features=X_new[i],
           # feature_names=data.columns,
            #matplotlib=False
       # )
       # st.components.v1.html(shap_html.html(), height=300)

    
# --- NETWORK VISUALIZATION ---
if uploaded_file is not None:
    with st.spinner("Plotting network, please wait..."):
        # Ensure nodes are strings
        data["node"] = data["node"].astype(str).strip()
        leak_nodes = data.loc[data["predicted_leak"] == 1, "node"].tolist()
        leak_nodes = [n.strip() for n in leak_nodes]
        valid_leaks = [n for n in leak_nodes if n in net.node_name_list]

        st.write("Predicted leak nodes:", leak_nodes)
        st.write("Network nodes count:", len(net.node_name_list))
        st.write("Valid leak nodes in network:", [n for n in leak_nodes if n in net.node_name_list])

        st.write("CSV leak nodes:", leak_nodes)
        st.write("Network nodes:", net.node_name_list[:10])  # preview
        st.write("Valid leak nodes:", valid_leaks)



        
        # Map node colors
        node_colors = {node: 'red' for node in leak_nodes if node in net.node_name_list}
        
        if valid_leaks:
            node_colors = {node: 'red' for node in valid_leaks}
            fig, ax = plt.subplots(figsize=(8, 6))
            wntr.graphics.plot_network(
                net,
                node_attribute=node_colors,
                node_size=50,
                link_width=1.5,
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.warning("No predicted leak nodes match the network nodes. Nothing to highlight.")

