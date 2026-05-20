"""
HydraAI — Hydraulic Risk Monitoring & Early Warning System
Streamlit Dashboard | Predictive Maintenance Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="HydraAI — Early Warning System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Industrial / Utility Aesthetic
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0e14;
    --bg-secondary: #111720;
    --bg-card: #141c26;
    --bg-card-border: #1e2d3d;
    --accent-blue: #00d4ff;
    --accent-cyan: #00ffcc;
    --accent-yellow: #ffd700;
    --accent-red: #ff3b3b;
    --accent-green: #00e676;
    --text-primary: #e8f4f8;
    --text-secondary: #7a9bb5;
    --text-muted: #4a6278;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--bg-card-border);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-blue);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
}

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: var(--font-mono);
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
}

.metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Alert Badges ── */
.badge-active {
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid var(--accent-red);
    color: var(--accent-red);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    text-transform: uppercase;
    display: inline-block;
}

.badge-warning {
    background: rgba(255, 215, 0, 0.12);
    border: 1px solid var(--accent-yellow);
    color: var(--accent-yellow);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    text-transform: uppercase;
    display: inline-block;
}

.badge-standby {
    background: rgba(0, 230, 118, 0.1);
    border: 1px solid var(--accent-green);
    color: var(--accent-green);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    text-transform: uppercase;
    display: inline-block;
}

/* ── Section Headers ── */
.section-header {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--bg-card-border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* ── Alert Row ── */
.alert-row {
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.alert-row-active {
    border-left: 3px solid var(--accent-red);
}

.alert-row-warning {
    border-left: 3px solid var(--accent-yellow);
}

.alert-row-standby {
    border-left: 3px solid var(--accent-green);
}

/* ── Logo / Title ── */
.hydra-logo {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--accent-blue);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
}

.hydra-tagline {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--bg-card-border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
}

div[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--accent-blue);
}

.stSelectbox label, .stSlider label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary) !important;
}

div[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border-color: var(--bg-card-border) !important;
    color: var(--text-primary) !important;
}

.stSlider .stSlider-thumb {
    background-color: var(--accent-blue) !important;
}

hr {
    border-color: var(--bg-card-border);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE PATH RESOLUTION
# ─────────────────────────────────────────────

UPLOAD_DIR = "/mnt/user-data/uploads"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."

def find_file(filename):
    """Search common locations for a file."""
    candidates = [
        os.path.join(UPLOAD_DIR, filename),
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(SCRIPT_DIR, "data", filename),
        os.path.join(SCRIPT_DIR, "models", filename),
        os.path.join(SCRIPT_DIR, "networks", filename),
        os.path.join(SCRIPT_DIR, "..", "models", filename),   # for .pkl files
        os.path.join(SCRIPT_DIR, "..", "data", filename),     # for .csv
        os.path.join(SCRIPT_DIR, "..", filename),             # for Net3.inp
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    path = find_file("xgb_model.pkl")
    if path is None:
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_feature_list():
    path = find_file("feature_list.pkl")
    if path is None:
        return None
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_dataset():
    path = find_file("dataset_features.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_network():
    """Load WNTR network — returns None if unavailable."""
    inp_path = find_file("Net3.inp")
    if inp_path is None:
        return None
    try:
        import wntr
        wn = wntr.network.WaterNetworkModel(inp_path)
        return wn
    except Exception:
        return None

# ─────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────

def run_inference(df_scenario, model, feature_list):
    """Run model inference on a filtered scenario dataframe."""
    X = df_scenario[feature_list].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    return probs

# ─────────────────────────────────────────────
# TEMPORAL RISK ENGINE
# ─────────────────────────────────────────────

def compute_smoothed_risk(prob_series, window=6):
    """Rolling mean smoothing over probability series."""
    return prob_series.rolling(window=window, min_periods=1).mean()

def compute_sustained_alerts(smoothed_series, threshold=0.25, min_consecutive=3):
    """
    Returns a boolean series: True where sustained alert is active.
    Alert triggers only after `min_consecutive` consecutive timesteps above threshold.
    """
    above = (smoothed_series > threshold).astype(int)
    # Rolling sum — if all `min_consecutive` recent steps are above threshold → alert
    sustained = above.rolling(window=min_consecutive, min_periods=min_consecutive).sum() == min_consecutive
    return sustained.fillna(False)

def get_alert_state(smoothed_prob, threshold):
    """Return alert state string based on smoothed probability."""
    if smoothed_prob >= threshold * 1.5:
        return "ACTIVE"
    elif smoothed_prob >= threshold:
        return "WARNING"
    else:
        return "STANDBY"

# ─────────────────────────────────────────────
# SCENARIO PROCESSING
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def process_scenario(_model, _feature_list, scenario_id, smoothing_window, threshold, _df_all):
    """
    Full pipeline for a given scenario_id:
    inference → smoothing → alert detection for all nodes.
    Returns a dataframe with per-(node, time) risk data.
    """
    df_scen = _df_all[_df_all["scenario_id"] == scenario_id].copy()
    df_scen = df_scen.sort_values(["node", "time"]).reset_index(drop=True)

    results = []
    for node, grp in df_scen.groupby("node"):
        grp = grp.copy().reset_index(drop=True)
        grp["leak_probability"] = run_inference(grp, _model, _feature_list)
        grp["risk_smoothed"] = compute_smoothed_risk(grp["leak_probability"], window=smoothing_window)
        grp["alert_active"] = compute_sustained_alerts(grp["risk_smoothed"], threshold=threshold)
        results.append(grp)

    return pd.concat(results, ignore_index=True)

# ─────────────────────────────────────────────
# ALERT PANEL DATA
# ─────────────────────────────────────────────

def build_alert_panel(df_processed, threshold):
    """Summarise per-node risk statistics for the alert panel."""
    rows = []
    for node, grp in df_processed.groupby("node"):
        current_smooth = grp["risk_smoothed"].iloc[-1]
        max_prob = grp["leak_probability"].max()
        duration_above = int((grp["risk_smoothed"] > threshold).sum())
        any_sustained = grp["alert_active"].any()
        state = get_alert_state(current_smooth, threshold)

        rows.append({
            "node": node,
            "current_smoothed": current_smooth,
            "max_probability": max_prob,
            "duration_above_threshold": duration_above,
            "sustained_alert": any_sustained,
            "alert_state": state,
        })

    panel = pd.DataFrame(rows)

    # Sort: ACTIVE first, then WARNING, then STANDBY, then by smoothed prob desc
    state_order = {"ACTIVE": 0, "WARNING": 1, "STANDBY": 2}
    panel["_sort_state"] = panel["alert_state"].map(state_order)
    panel = panel.sort_values(["_sort_state", "current_smoothed"], ascending=[True, False])
    panel = panel.drop(columns=["_sort_state"]).reset_index(drop=True)
    return panel

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0a0e14",
        "plot_bgcolor": "#0f1620",
        "font": {"color": "#e8f4f8", "family": "Space Mono, monospace"},
        "xaxis": {
            "gridcolor": "#1e2d3d",
            "linecolor": "#1e2d3d",
            "tickcolor": "#4a6278",
            "tickfont": {"color": "#7a9bb5", "size": 10},
            "title": {"font": {"color": "#7a9bb5"}},
        },
        "yaxis": {
            "gridcolor": "#1e2d3d",
            "linecolor": "#1e2d3d",
            "tickcolor": "#4a6278",
            "tickfont": {"color": "#7a9bb5", "size": 10},
            "title": {"font": {"color": "#7a9bb5"}},
        },
        "legend": {
            "bgcolor": "#141c26",
            "bordercolor": "#1e2d3d",
            "borderwidth": 1,
            "font": {"color": "#7a9bb5", "size": 10},
        },
        "hoverlabel": {
            "bgcolor": "#141c26",
            "bordercolor": "#1e2d3d",
            "font": {"color": "#e8f4f8", "size": 11},
        },
        "margin": {"l": 50, "r": 30, "t": 40, "b": 50},
    }
}


def plot_risk_over_time(node_df, node_name, threshold):
    """Interactive Plotly chart: raw + smoothed risk with alert highlights."""
    time_hours = node_df["time"] / 3600

    fig = go.Figure()

    # ── Sustained-risk fill regions ──
    alert_mask = node_df["alert_active"].values
    in_region = False
    region_start = None
    for i, (t, is_alert) in enumerate(zip(time_hours, alert_mask)):
        if is_alert and not in_region:
            region_start = t
            in_region = True
        elif not is_alert and in_region:
            fig.add_vrect(
                x0=region_start, x1=t,
                fillcolor="rgba(255,59,59,0.08)",
                layer="below", line_width=0,
            )
            in_region = False
    if in_region:
        fig.add_vrect(
            x0=region_start, x1=time_hours.iloc[-1],
            fillcolor="rgba(255,59,59,0.08)",
            layer="below", line_width=0,
        )

    # ── Raw probability ──
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=node_df["leak_probability"],
        mode="lines",
        name="Raw Probability",
        line=dict(color="rgba(0,212,255,0.35)", width=1),
        hovertemplate="<b>Time:</b> %{x:.2f}h<br><b>Raw:</b> %{y:.4f}<extra></extra>",
    ))

    # ── Smoothed risk ──
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=node_df["risk_smoothed"],
        mode="lines",
        name="Smoothed Risk",
        line=dict(color="#00d4ff", width=2.5),
        hovertemplate="<b>Time:</b> %{x:.2f}h<br><b>Smoothed:</b> %{y:.4f}<extra></extra>",
    ))

    # ── Threshold line ──
    fig.add_hline(
        y=threshold,
        line=dict(color="#ffd700", width=1.5, dash="dot"),
        annotation_text=f"Threshold ({threshold})",
        annotation_position="bottom right",
        annotation_font=dict(color="#ffd700", size=10, family="Space Mono"),
    )

    # ── Alert scatter ──
    alert_df = node_df[node_df["alert_active"]]
    if not alert_df.empty:
        fig.add_trace(go.Scatter(
            x=alert_df["time"] / 3600,
            y=alert_df["risk_smoothed"],
            mode="markers",
            name="Sustained Alert",
            marker=dict(color="#ff3b3b", size=6, symbol="circle",
                        line=dict(color="#ff8080", width=1)),
            hovertemplate="<b>ALERT</b><br>Time: %{x:.2f}h<br>Risk: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(
            text=f"<span style='font-family:Space Mono;font-size:13px;color:#7a9bb5'>NODE</span> "
                 f"<span style='font-family:Space Mono;font-size:13px;color:#00d4ff'>{node_name}</span>"
                 f"<span style='font-family:Space Mono;font-size:11px;color:#4a6278'> — Leak Risk Timeline</span>",
            x=0, xanchor="left",
        ),
        xaxis_title="Time (hours)",
        yaxis_title="Leak Probability",
        yaxis=dict(**PLOTLY_TEMPLATE["layout"]["yaxis"], range=[-0.02, 1.02]),
        height=400,
        hovermode="x unified",
    )
    return fig


def plot_network_risk(wn, alert_panel_df, threshold):
    """
    Matplotlib WNTR network plot with node coloring based on smoothed risk.
    Returns a matplotlib figure.
    """
    try:
        import wntr

        node_risk = dict(zip(alert_panel_df["node"], alert_panel_df["current_smoothed"]))

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0a0e14")
        ax.set_facecolor("#0f1620")

        # Plot pipes (edges)
        pos = {}
        for name, node in wn.nodes():
            pos[name] = (node.coordinates[0], node.coordinates[1])

        for pipe_name, pipe in wn.pipes():
            n1, n2 = pipe.start_node_name, pipe.end_node_name
            if n1 in pos and n2 in pos:
                x_vals = [pos[n1][0], pos[n2][0]]
                y_vals = [pos[n1][1], pos[n2][1]]
                ax.plot(x_vals, y_vals, color="#1e2d3d", linewidth=1, zorder=1)

        # Plot nodes
        for node_name, (x, y) in pos.items():
            risk = node_risk.get(node_name, 0.0)
            if risk >= threshold * 1.5:
                color = "#ff3b3b"
                size = 80
                edgecolor = "#ff8080"
            elif risk >= threshold:
                color = "#ffd700"
                size = 60
                edgecolor = "#ffe680"
            else:
                t = min(risk / threshold, 1.0) if threshold > 0 else 0
                # Interpolate green → yellow-green
                r = int(0 + t * 100)
                g = int(230 - t * 50)
                b = int(118 - t * 80)
                color = f"#{r:02x}{g:02x}{b:02x}"
                size = 40
                edgecolor = "#2a3d50"

            ax.scatter(x, y, c=color, s=size, zorder=3,
                       edgecolors=edgecolor, linewidths=0.8)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff3b3b',
                   markeredgecolor='#ff8080', markersize=9, label='ACTIVE ALERT'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd700',
                   markeredgecolor='#ffe680', markersize=8, label='WARNING'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00e676',
                   markeredgecolor='#2a3d50', markersize=7, label='STANDBY / LOW'),
        ]
        legend = ax.legend(
            handles=legend_elements, loc='lower left',
            facecolor='#141c26', edgecolor='#1e2d3d',
            labelcolor='#7a9bb5',
            prop={'family': 'monospace', 'size': 8},
        )

        ax.set_title("HYDRAULIC NETWORK — SPATIAL RISK MAP",
                     color="#7a9bb5", fontsize=9,
                     fontfamily="monospace", loc="left", pad=10)
        ax.tick_params(colors="#4a6278", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d3d")

        plt.tight_layout()
        return fig

    except Exception as e:
        return None


def plot_risk_histogram(alert_panel_df, threshold):
    """Distribution of current smoothed risk across all nodes."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=alert_panel_df["current_smoothed"],
        nbinsx=30,
        marker=dict(
            color=alert_panel_df["current_smoothed"].apply(
                lambda v: "#ff3b3b" if v >= threshold * 1.5
                else ("#ffd700" if v >= threshold else "#00d4ff")
            ),
            line=dict(color="#0a0e14", width=0.5),
        ),
        hovertemplate="Risk: %{x:.3f}<br>Nodes: %{y}<extra></extra>",
        name="Node Risk Distribution",
    ))
    fig.add_vline(
        x=threshold,
        line=dict(color="#ffd700", width=1.5, dash="dot"),
        annotation_text=f"Threshold",
        annotation_font=dict(color="#ffd700", size=9, family="Space Mono"),
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(
            text="<span style='font-family:Space Mono;font-size:11px;color:#7a9bb5'>NODE RISK DISTRIBUTION</span>",
            x=0, xanchor="left",
        ),
        xaxis_title="Smoothed Probability",
        yaxis_title="Node Count",
        height=280,
        showlegend=False,
        bargap=0.05,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class='hydra-logo'>💧 HydraAI</div>
    <div class='hydra-tagline'>Hydraulic Early Warning System</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Load data for sidebar controls
    df_all = load_dataset()
    model = load_model()
    feature_list = load_feature_list()

    if df_all is None or model is None or feature_list is None:
        st.error("⚠️ Required files not found. Please ensure dataset_features.csv, xgb_model.pkl, and feature_list.pkl are available.")
        st.stop()

    st.markdown("### 📡 Scenario")
    scenarios = sorted(df_all["scenario_id"].unique())
    selected_scenario = st.selectbox("Scenario ID", scenarios, index=0, label_visibility="collapsed")

    st.markdown("### 🔵 Node")
    nodes_in_scenario = sorted(df_all[df_all["scenario_id"] == selected_scenario]["node"].unique())
    selected_node = st.selectbox("Node", nodes_in_scenario, index=0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Risk Engine Parameters")

    threshold = st.slider(
        "Alert Threshold",
        min_value=0.10, max_value=0.60, value=0.25, step=0.01,
        help="Smoothed probability threshold for triggering alerts"
    )

    smoothing_window = st.slider(
        "Smoothing Window (timesteps)",
        min_value=3, max_value=10, value=6, step=1,
        help="Rolling mean window for temporal smoothing"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Space Mono;font-size:0.6rem;color:#4a6278;line-height:1.6em'>"
        "P(leak) = probability of leak<br>within next 4 hours<br><br>"
        "Alerts trigger only after<br>≥3 consecutive steps<br>above threshold."
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# ── Header ──
st.markdown("""
<div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:0.2rem'>
  <span style='font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#00d4ff;letter-spacing:0.04em'>HydraAI</span>
  <span style='font-family:Space Mono;font-size:0.75rem;color:#4a6278;letter-spacing:0.1em'>HYDRAULIC RISK MONITORING & EARLY WARNING SYSTEM</span>
</div>
<div style='font-size:0.8rem;color:#7a9bb5;margin-bottom:1.5rem'>
  Real-time predictive maintenance platform for water distribution networks
</div>
""", unsafe_allow_html=True)

# ── Run inference for selected scenario ──
with st.spinner("Running inference pipeline…"):
    df_processed = process_scenario(
        model, feature_list,
        selected_scenario, smoothing_window, threshold,
        df_all
    )

alert_panel_df = build_alert_panel(df_processed, threshold)

# ── Node-specific data ──
node_df = df_processed[df_processed["node"] == selected_node].copy().reset_index(drop=True)
node_alert_row = alert_panel_df[alert_panel_df["node"] == selected_node]

current_smooth = node_df["risk_smoothed"].iloc[-1] if not node_df.empty else 0.0
max_prob = node_df["leak_probability"].max() if not node_df.empty else 0.0
alert_state = get_alert_state(current_smooth, threshold)
duration = int((node_df["risk_smoothed"] > threshold).sum()) if not node_df.empty else 0

# ── Top KPI Row ──
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    badge = (
        f"<span class='badge-active'>ACTIVE</span>" if alert_state == "ACTIVE"
        else f"<span class='badge-warning'>WARNING</span>" if alert_state == "WARNING"
        else f"<span class='badge-standby'>STANDBY</span>"
    )
    color = "#ff3b3b" if alert_state == "ACTIVE" else "#ffd700" if alert_state == "WARNING" else "#00e676"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Alert State</div>
      <div class='metric-value' style='color:{color};font-size:1.2rem'>{alert_state}</div>
      <div class='metric-sub'>Node {selected_node}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Current Risk</div>
      <div class='metric-value' style='color:#00d4ff'>{current_smooth:.3f}</div>
      <div class='metric-sub'>Smoothed probability</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Peak Probability</div>
      <div class='metric-value' style='color:#00ffcc'>{max_prob:.3f}</div>
      <div class='metric-sub'>Max observed</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Duration Above Thr.</div>
      <div class='metric-value' style='color:#ffd700'>{duration}</div>
      <div class='metric-sub'>Timesteps</div>
    </div>""", unsafe_allow_html=True)

with col5:
    active_nodes = int((alert_panel_df["alert_state"] == "ACTIVE").sum())
    warn_nodes = int((alert_panel_df["alert_state"] == "WARNING").sum())
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Network Alerts</div>
      <div class='metric-value' style='color:#ff3b3b'>{active_nodes} <span style='font-size:1rem;color:#ffd700'>/ {warn_nodes}</span></div>
      <div class='metric-sub'>Active / Warning</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Layout: Risk Chart + Network ──
chart_col, net_col = st.columns([3, 2], gap="medium")

with chart_col:
    st.markdown("<div class='section-header'>📈 Risk Over Time — Selected Node</div>", unsafe_allow_html=True)
    if not node_df.empty:
        fig_risk = plot_risk_over_time(node_df, selected_node, threshold)
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data available for this node/scenario combination.")

with net_col:
    st.markdown("<div class='section-header'>🗺️ Hydraulic Network — Spatial Risk</div>", unsafe_allow_html=True)
    wn = load_network()
    if wn is not None:
        net_fig = plot_network_risk(wn, alert_panel_df, threshold)
        if net_fig is not None:
            st.pyplot(net_fig, use_container_width=True)
        else:
            st.info("Network visualization unavailable.")
    else:
        # Fallback: scatter plot of risk across nodes
        st.markdown(
            "<div style='font-family:Space Mono;font-size:0.65rem;color:#4a6278;"
            "padding:0.5rem 0 0.5rem 0'>Net3.inp not found — showing risk scatter</div>",
            unsafe_allow_html=True
        )
        top_nodes = alert_panel_df.head(30)
        color_map = top_nodes["alert_state"].map(
            {"ACTIVE": "#ff3b3b", "WARNING": "#ffd700", "STANDBY": "#00d4ff"}
        )
        fig_scatter = go.Figure(go.Bar(
            x=top_nodes["node"],
            y=top_nodes["current_smoothed"],
            marker_color=color_map,
            hovertemplate="<b>Node:</b> %{x}<br><b>Risk:</b> %{y:.4f}<extra></extra>",
        ))
        fig_scatter.add_hline(y=threshold, line=dict(color="#ffd700", width=1.5, dash="dot"))
        fig_scatter.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            xaxis_title="Node", yaxis_title="Smoothed Risk",
            height=380, title=dict(
                text="<span style='font-family:Space Mono;font-size:11px;color:#7a9bb5'>TOP 30 NODES BY RISK</span>",
                x=0, xanchor="left"
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ── Bottom Row: Alert Panel + Distribution ──
alert_col, dist_col = st.columns([3, 2], gap="medium")

with alert_col:
    st.markdown("<div class='section-header'>🚨 Alert Panel — All Nodes</div>", unsafe_allow_html=True)

    # Top 15 nodes by alert priority
    display_panel = alert_panel_df.head(15)

    for _, row in display_panel.iterrows():
        state = row["alert_state"]
        row_class = {
            "ACTIVE": "alert-row alert-row-active",
            "WARNING": "alert-row alert-row-warning",
            "STANDBY": "alert-row alert-row-standby",
        }.get(state, "alert-row")
        badge_html = {
            "ACTIVE": "<span class='badge-active'>ACTIVE</span>",
            "WARNING": "<span class='badge-warning'>WARNING</span>",
            "STANDBY": "<span class='badge-standby'>STANDBY</span>",
        }.get(state, "")

        highlight = " border: 1px solid #1e3050;" if row["node"] == selected_node else ""

        st.markdown(f"""
        <div class='{row_class}' style='font-family:Space Mono;{highlight}'>
          <div>
            <span style='font-size:0.8rem;color:#e8f4f8;font-weight:600'>{row['node']}</span>
            <span style='font-size:0.65rem;color:#4a6278;margin-left:0.6rem'>
              {'▶ selected' if row['node'] == selected_node else ''}
            </span>
          </div>
          <div style='display:flex;gap:1.5rem;align-items:center;font-size:0.7rem;color:#7a9bb5;margin-top:0.3rem'>
            <span>Risk: <span style='color:#00d4ff'>{row['current_smoothed']:.3f}</span></span>
            <span>Peak: <span style='color:#00ffcc'>{row['max_probability']:.3f}</span></span>
            <span>Duration: <span style='color:#ffd700'>{row['duration_above_threshold']}</span></span>
          </div>
          <div style='margin-top:0.3rem'>{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)

with dist_col:
    st.markdown("<div class='section-header'>📊 Network Risk Distribution</div>", unsafe_allow_html=True)
    fig_hist = plot_risk_histogram(alert_panel_df, threshold)
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Summary stats
    st.markdown("<br>", unsafe_allow_html=True)
    total_nodes = len(alert_panel_df)
    pct_above = 100 * (alert_panel_df["current_smoothed"] > threshold).mean()
    mean_risk = alert_panel_df["current_smoothed"].mean()

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class='metric-card' style='padding:0.8rem;'>
          <div class='metric-label' style='font-size:0.6rem'>Total Nodes</div>
          <div class='metric-value' style='font-size:1.2rem;color:#7a9bb5'>{total_nodes}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class='metric-card' style='padding:0.8rem;'>
          <div class='metric-label' style='font-size:0.6rem'>Above Thr.</div>
          <div class='metric-value' style='font-size:1.2rem;color:#ffd700'>{pct_above:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class='metric-card' style='padding:0.8rem;'>
          <div class='metric-label' style='font-size:0.6rem'>Mean Risk</div>
          <div class='metric-value' style='font-size:1.2rem;color:#00d4ff'>{mean_risk:.3f}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid #1e2d3d;padding-top:0.8rem;
font-family:Space Mono;font-size:0.6rem;color:#4a6278;
display:flex;justify-content:space-between;'>
  <span>HydraAI — Hydraulic Early Warning System</span>
  <span>XGBoost · WNTR · Streamlit</span>
  <span>P(leak | next 4h)</span>
</div>
""", unsafe_allow_html=True)