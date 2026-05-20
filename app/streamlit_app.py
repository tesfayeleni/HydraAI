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
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # app/ -> HydraAI/

def find_file(filename):
    """Search common locations for a file, walking up to project root."""
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(PARENT_DIR, filename),
        os.path.join(PARENT_DIR, "data", filename),
        os.path.join(PARENT_DIR, "models", filename),
        os.path.join(PARENT_DIR, "networks", filename),
        os.path.join(SCRIPT_DIR, "data", filename),
        os.path.join(SCRIPT_DIR, "models", filename),
        os.path.join(UPLOAD_DIR, filename),
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

def apply_dark_theme(fig, height=400, extra=None):
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0f1620",
        font=dict(color="#e8f4f8", family="Space Mono, monospace"),
        xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickcolor="#4a6278", tickfont=dict(color="#7a9bb5", size=10),
                   title_font=dict(color="#7a9bb5")),
        yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickcolor="#4a6278", tickfont=dict(color="#7a9bb5", size=10),
                   title_font=dict(color="#7a9bb5")),
        legend=dict(bgcolor="#141c26", bordercolor="#1e2d3d",
                    borderwidth=1, font=dict(color="#7a9bb5", size=10)),
        hoverlabel=dict(bgcolor="#141c26", bordercolor="#1e2d3d",
                        font=dict(color="#e8f4f8", size=11)),
        margin=dict(l=50, r=30, t=40, b=50),
        height=height,
    )
    if extra:
        fig.update_layout(**extra)
    return fig


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

    apply_dark_theme(fig, height=400, extra=dict(
        title=dict(
            text=f"<span style='font-family:Space Mono;font-size:13px;color:#7a9bb5'>NODE</span> "
                 f"<span style='font-family:Space Mono;font-size:13px;color:#00d4ff'>{node_name}</span>"
                 f"<span style='font-family:Space Mono;font-size:11px;color:#4a6278'> — Leak Risk Timeline</span>",
            x=0, xanchor="left",
        ),
        xaxis=dict(title="Time (hours)", gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickfont=dict(color="#7a9bb5", size=10), title_font=dict(color="#7a9bb5")),
        yaxis=dict(title="Leak Probability", gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickfont=dict(color="#7a9bb5", size=10), title_font=dict(color="#7a9bb5"),
                   range=[-0.02, 1.02]),
        hovermode="x unified",
    ))
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
    apply_dark_theme(fig, height=280, extra=dict(
        title=dict(
            text="<span style='font-family:Space Mono;font-size:11px;color:#7a9bb5'>NODE RISK DISTRIBUTION</span>",
            x=0, xanchor="left",
        ),
        xaxis=dict(title="Smoothed Probability", gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickfont=dict(color="#7a9bb5", size=10), title_font=dict(color="#7a9bb5")),
        yaxis=dict(title="Node Count", gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickfont=dict(color="#7a9bb5", size=10), title_font=dict(color="#7a9bb5")),
        showlegend=False,
        bargap=0.05,
    ))
    return fig


# ─────────────────────────────────────────────
# NETWORK OVERVIEW VISUALIZATION (full-width Plotly)
# ─────────────────────────────────────────────

def plot_network_overview(wn, alert_panel_df, threshold, selected_node=None):
    """
    Full-width interactive Plotly network map.
    Nodes colored by smoothed risk. Clicking/hovering shows node info.
    """
    node_risk = dict(zip(alert_panel_df["node"], alert_panel_df["current_smoothed"]))
    node_state = dict(zip(alert_panel_df["node"], alert_panel_df["alert_state"]))
    node_peak = dict(zip(alert_panel_df["node"], alert_panel_df["max_probability"]))
    node_dur = dict(zip(alert_panel_df["node"], alert_panel_df["duration_above_threshold"]))

    pos = {}
    for name, node in wn.nodes():
        pos[name] = (node.coordinates[0], node.coordinates[1])

    fig = go.Figure()

    # ── Pipe edges ──
    edge_x, edge_y = [], []
    for pipe_name, pipe in wn.pipes():
        n1, n2 = pipe.start_node_name, pipe.end_node_name
        if n1 in pos and n2 in pos:
            edge_x += [pos[n1][0], pos[n2][0], None]
            edge_y += [pos[n1][1], pos[n2][1], None]

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#1e2d3d", width=1.5),
        hoverinfo="none",
        showlegend=False,
        name="pipes",
    ))

    # ── Node groups by state ──
    groups = {
        "ACTIVE":  {"color": "#ff3b3b", "size": 14, "edge": "#ff8080", "label": "🔴 ACTIVE ALERT"},
        "WARNING": {"color": "#ffd700", "size": 11, "edge": "#ffe680", "label": "🟡 WARNING"},
        "STANDBY": {"color": "#00e676", "size": 8,  "edge": "#00b050", "label": "🟢 STANDBY"},
    }

    for state, style in groups.items():
        xs, ys, texts, names = [], [], [], []
        for node_name, (x, y) in pos.items():
            if node_state.get(node_name, "STANDBY") != state:
                continue
            risk = node_risk.get(node_name, 0.0)
            peak = node_peak.get(node_name, 0.0)
            dur = node_dur.get(node_name, 0)
            xs.append(x)
            ys.append(y)
            names.append(node_name)
            texts.append(
                f"<b>Node {node_name}</b><br>"
                f"State: <b>{state}</b><br>"
                f"Current Risk: {risk:.4f}<br>"
                f"Peak: {peak:.4f}<br>"
                f"Duration: {dur} steps"
            )

        # Highlight selected node
        marker_colors = []
        marker_sizes = []
        marker_line_colors = []
        marker_line_widths = []
        for n in names:
            if n == selected_node:
                marker_colors.append("#ffffff")
                marker_sizes.append(style["size"] + 6)
                marker_line_colors.append("#00d4ff")
                marker_line_widths.append(2.5)
            else:
                marker_colors.append(style["color"])
                marker_sizes.append(style["size"])
                marker_line_colors.append(style["edge"])
                marker_line_widths.append(1)

        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=style["label"],
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    color=marker_colors,
                    size=marker_sizes,
                    line=dict(
                        color=marker_line_colors,
                        width=marker_line_widths,
                    ),
                ),
            ))

    fig.update_layout(
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0a0e14",
        font=dict(color="#e8f4f8", family="Space Mono, monospace"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False, fixedrange=True),
        legend=dict(
            bgcolor="#141c26", bordercolor="#1e2d3d", borderwidth=1,
            font=dict(color="#7a9bb5", size=11, family="Space Mono"),
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        hoverlabel=dict(
            bgcolor="#141c26", bordercolor="#1e2d3d",
            font=dict(color="#e8f4f8", size=12, family="Space Mono"),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
        hovermode="closest",
    )
    return fig


def plot_network_fallback(alert_panel_df, threshold, selected_node=None):
    """Plotly bar chart fallback when Net3.inp is unavailable."""
    panel = alert_panel_df.copy()
    colors = panel["alert_state"].map(
        {"ACTIVE": "#ff3b3b", "WARNING": "#ffd700", "STANDBY": "#00d4ff"}
    ).tolist()
    # Highlight selected
    if selected_node:
        colors = [
            "#ffffff" if n == selected_node else c
            for n, c in zip(panel["node"], colors)
        ]

    fig = go.Figure(go.Bar(
        x=panel["node"],
        y=panel["current_smoothed"],
        marker_color=colors,
        hovertemplate="<b>Node %{x}</b><br>Risk: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line=dict(color="#ffd700", width=1.5, dash="dot"),
                  annotation_text="Threshold", annotation_font=dict(color="#ffd700", size=9))
    apply_dark_theme(fig, height=420, extra=dict(
        xaxis=dict(title="Node", gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickfont=dict(color="#7a9bb5", size=9), title_font=dict(color="#7a9bb5"),
                   tickangle=-45),
        yaxis=dict(title="Smoothed Risk", gridcolor="#1e2d3d", linecolor="#1e2d3d",
                   tickfont=dict(color="#7a9bb5", size=10), title_font=dict(color="#7a9bb5"),
                   range=[0, max(alert_panel_df["current_smoothed"].max() * 1.2, threshold * 2)]),
        bargap=0.2,
        showlegend=False,
    ))
    return fig


# ─────────────────────────────────────────────
# SESSION STATE — navigation
# ─────────────────────────────────────────────

if "view" not in st.session_state:
    st.session_state.view = "map"          # "map" | "node"
if "drill_node" not in st.session_state:
    st.session_state.drill_node = None

def go_to_node(node_name):
    st.session_state.drill_node = node_name
    st.session_state.view = "node"

def go_to_map():
    st.session_state.view = "map"

# ─────────────────────────────────────────────
# SIDEBAR  (always visible)
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class='hydra-logo'>💧 HydraAI</div>
    <div class='hydra-tagline'>Hydraulic Early Warning System</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    df_all = load_dataset()
    model  = load_model()
    feature_list = load_feature_list()

    if df_all is None or model is None or feature_list is None:
        st.error("⚠️ Required files not found. Ensure dataset_features.csv, xgb_model.pkl, and feature_list.pkl are available.")
        st.stop()

    st.markdown("### 📡 Scenario")
    scenarios = sorted(df_all["scenario_id"].unique())
    selected_scenario = st.selectbox(
        "Scenario ID", scenarios, index=0, label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Risk Engine")
    threshold = st.slider(
        "Alert Threshold", min_value=0.10, max_value=0.60,
        value=0.25, step=0.01,
        help="Probability above which a node is flagged. Does not affect model predictions."
    )
    smoothing_window = st.slider(
        "Smoothing Window (timesteps)", min_value=3, max_value=10,
        value=6, step=1,
        help="Rolling mean window — wider = fewer false alarms, slower response."
    )
    st.markdown(
        "<div style='font-family:Space Mono;font-size:0.6rem;color:#4a6278;line-height:1.7'>"
        "These controls adjust <b style='color:#7a9bb5'>alert flagging only</b>.<br>"
        "The XGBoost model is not retrained.<br><br>"
        "P(leak) = probability of leak<br>within next 4 hours.<br><br>"
        "Alerts trigger after ≥3<br>consecutive steps above threshold."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Process scenario
    with st.spinner("Running inference…"):
        df_processed = process_scenario(
            model, feature_list,
            selected_scenario, smoothing_window, threshold,
            df_all
        )
    alert_panel_df = build_alert_panel(df_processed, threshold)

    # Node jump list — sorted by risk
    st.markdown("### 🔵 Jump to Node")
    sorted_nodes = alert_panel_df["node"].tolist()

    def node_label(n):
        row = alert_panel_df.loc[alert_panel_df["node"] == n].iloc[0]
        icon = "🔴" if row["alert_state"] == "ACTIVE" else "🟡" if row["alert_state"] == "WARNING" else "🟢"
        return f"{icon} {n}  ({row['current_smoothed']:.3f})"

    jump_node = st.selectbox(
        "Select node", sorted_nodes,
        format_func=node_label,
        label_visibility="collapsed",
        key="sidebar_node_select"
    )
    if st.button("🔍 Inspect Node", use_container_width=True):
        go_to_node(jump_node)
        st.rerun()

    if st.session_state.view == "node":
        if st.button("← Back to Network Map", use_container_width=True):
            go_to_map()
            st.rerun()


# ─────────────────────────────────────────────
# SHARED — load network + compute globals
# ─────────────────────────────────────────────

wn = load_network()

active_nodes  = int((alert_panel_df["alert_state"] == "ACTIVE").sum())
warn_nodes    = int((alert_panel_df["alert_state"] == "WARNING").sum())
total_nodes   = len(alert_panel_df)
standby_nodes = total_nodes - active_nodes - warn_nodes
mean_risk     = alert_panel_df["current_smoothed"].mean()
pct_above     = 100 * (alert_panel_df["current_smoothed"] > threshold).mean()
peak_node     = alert_panel_df.iloc[0]["node"] if not alert_panel_df.empty else "—"
peak_val      = alert_panel_df.iloc[0]["current_smoothed"] if not alert_panel_df.empty else 0.0

def kpi(label, value, color, sub=""):
    return (
        f"<div class='metric-card' style='padding:0.8rem 1rem;'>"
        f"<div class='metric-label' style='font-size:0.6rem'>{label}</div>"
        f"<div class='metric-value' style='font-size:1.3rem;color:{color}'>{value}</div>"
        + (f"<div class='metric-sub'>{sub}</div>" if sub else "")
        + "</div>"
    )

# ─────────────────────────────────────────────
# HEADER BAR (always shown)
# ─────────────────────────────────────────────

view_label = "NETWORK MAP" if st.session_state.view == "map" else f"NODE DRILL-DOWN · {st.session_state.drill_node}"
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
margin-bottom:1rem;padding-bottom:0.8rem;border-bottom:1px solid #1e2d3d;'>
  <div>
    <span style='font-family:Space Mono;font-size:1.4rem;font-weight:700;
    color:#00d4ff;letter-spacing:0.04em'>HydraAI</span>
    <span style='font-family:Space Mono;font-size:0.7rem;color:#4a6278;
    letter-spacing:0.1em;margin-left:0.8rem'>{view_label}</span>
  </div>
  <div style='display:flex;gap:1.5rem;font-family:Space Mono;font-size:0.7rem;'>
    <span style='color:#4a6278'>SCENARIO <span style='color:#e8f4f8'>{selected_scenario}</span></span>
    <span style='color:#4a6278'>NODES <span style='color:#e8f4f8'>{total_nodes}</span></span>
    <span style='color:#4a6278'>MEAN RISK <span style='color:#00d4ff'>{mean_risk:.3f}</span></span>
    <span style='color:#4a6278'>ABOVE THR <span style='color:#ffd700'>{pct_above:.1f}%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip (always shown) ──
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.markdown(kpi("Active Alerts",    active_nodes,        "#ff3b3b", "sustained risk"),   unsafe_allow_html=True)
with k2: st.markdown(kpi("Warnings",         warn_nodes,          "#ffd700", "above threshold"),  unsafe_allow_html=True)
with k3: st.markdown(kpi("Standby",          standby_nodes,       "#00e676", "low risk"),         unsafe_allow_html=True)
with k4: st.markdown(kpi("Mean Risk",        f"{mean_risk:.3f}",  "#00d4ff", "smoothed avg"),     unsafe_allow_html=True)
with k5: st.markdown(kpi("Above Threshold",  f"{pct_above:.1f}%", "#ffd700", f"thr={threshold}"), unsafe_allow_html=True)
with k6: st.markdown(kpi("Highest Risk",     peak_node,           "#ff3b3b", f"{peak_val:.3f}"),  unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# VIEW A — NETWORK MAP (default landing page)
# ══════════════════════════════════════════════════════

if st.session_state.view == "map":

    st.markdown(
        "<div class='section-header'>🗺️ Network Overview — All Nodes · Spatial Risk Map"
        " &nbsp;<span style='font-size:0.6rem;color:#4a6278'>"
        "hover a node for details · use sidebar to inspect</span></div>",
        unsafe_allow_html=True
    )

    net_col_main, alert_sidebar_col = st.columns([3, 1], gap="medium")

    with net_col_main:
        if wn is not None:
            net_fig = plot_network_overview(wn, alert_panel_df, threshold,
                                            st.session_state.drill_node)
            st.plotly_chart(net_fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(
                "<div style='font-family:Space Mono;font-size:0.65rem;color:#4a6278;"
                "padding:0.3rem 0 0.6rem'>Net3.inp not found — showing risk bar chart</div>",
                unsafe_allow_html=True
            )
            st.plotly_chart(
                plot_network_fallback(alert_panel_df, threshold, st.session_state.drill_node),
                use_container_width=True, config={"displayModeBar": False}
            )

    with alert_sidebar_col:
        st.markdown(
            "<div class='section-header' style='margin-top:0'>🚨 Alert Queue</div>",
            unsafe_allow_html=True
        )
        display_panel = alert_panel_df[
            alert_panel_df["alert_state"].isin(["ACTIVE", "WARNING"])
        ].head(10)
        if display_panel.empty:
            display_panel = alert_panel_df.head(8)

        for _, row in display_panel.iterrows():
            s = row["alert_state"]
            bc  = "#ff3b3b" if s == "ACTIVE" else "#ffd700" if s == "WARNING" else "#00e676"
            bdg = "badge-active" if s == "ACTIVE" else "badge-warning" if s == "WARNING" else "badge-standby"
            st.markdown(f"""
            <div style='border-left:3px solid {bc};background:#141c26;
            border-radius:0 5px 5px 0;padding:0.55rem 0.8rem;margin-bottom:0.4rem;
            font-family:Space Mono;border-top:1px solid #1e2d3d;
            border-right:1px solid #1e2d3d;border-bottom:1px solid #1e2d3d;'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:0.75rem;color:#e8f4f8;font-weight:700'>{row["node"]}</span>
                <span class='{bdg}'>{s}</span>
              </div>
              <div style='font-size:0.65rem;color:#7a9bb5;margin-top:0.3rem'>
                Risk <span style='color:#00d4ff'>{row["current_smoothed"]:.3f}</span>
                &nbsp;·&nbsp; {row["duration_above_threshold"]}×
              </div>
            </div>
            """, unsafe_allow_html=True)
            # Inspect button per alert row
            if st.button(f"Inspect {row['node']}", key=f"btn_{row['node']}",
                         use_container_width=True):
                go_to_node(row["node"])
                st.rerun()

    # ── Risk Distribution — full width, below the network map + alert queue ──
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>📊 Network Risk Distribution — All Nodes</div>",
        unsafe_allow_html=True
    )
    fig_hist = plot_risk_histogram(alert_panel_df, threshold)
    fig_hist.update_layout(height=240)
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════
# VIEW B — NODE DRILL-DOWN
# ══════════════════════════════════════════════════════

else:
    drill_node = st.session_state.drill_node

    # ── Back button at top ──
    if st.button("← Back to Network Map", key="back_top"):
        go_to_map()
        st.rerun()

    # ── Get node data — FIX: filter from df_processed which has smoothed cols ──
    node_df = (
        df_processed[df_processed["node"] == drill_node]
        .sort_values("time")
        .reset_index(drop=True)
    )

    node_row = alert_panel_df[alert_panel_df["node"] == drill_node]
    if node_row.empty:
        st.warning(f"No data found for node {drill_node}.")
        st.stop()
    node_row = node_row.iloc[0]

    s_color = "#ff3b3b" if node_row["alert_state"] == "ACTIVE" else "#ffd700" if node_row["alert_state"] == "WARNING" else "#00e676"
    bdg_cls = "badge-active" if node_row["alert_state"] == "ACTIVE" else "badge-warning" if node_row["alert_state"] == "WARNING" else "badge-standby"

    # ── Node title bar ──
    st.markdown(f"""
    <div style='border-top:2px solid {s_color};border-radius:4px 4px 0 0;
    background:#141c26;padding:0.8rem 1.2rem;margin-bottom:1rem;
    display:flex;align-items:center;gap:1rem;'>
      <span style='font-family:Space Mono;font-size:0.65rem;letter-spacing:0.15em;
      text-transform:uppercase;color:#7a9bb5'>Node Drill-Down</span>
      <span style='font-family:Space Mono;font-size:1rem;font-weight:700;color:#e8f4f8'>{drill_node}</span>
      <span class='{bdg_cls}'>{node_row["alert_state"]}</span>
      <span style='font-family:Space Mono;font-size:0.65rem;color:#4a6278;margin-left:auto'>
        Scenario {selected_scenario} · {len(node_df)} timesteps · {node_df["time"].max()/3600:.1f}h window
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── 4-KPI row for this node ──
    nk1, nk2, nk3, nk4 = st.columns(4)
    with nk1: st.markdown(kpi("Alert State",     node_row["alert_state"],                            s_color,    "current"),           unsafe_allow_html=True)
    with nk2: st.markdown(kpi("Current Risk",    f"{node_row['current_smoothed']:.4f}",            "#00d4ff",  "smoothed prob"),      unsafe_allow_html=True)
    with nk3: st.markdown(kpi("Peak Probability",f"{node_row['max_probability']:.4f}",             "#00ffcc",  "max observed"),       unsafe_allow_html=True)
    with nk4: st.markdown(kpi("Duration Above",  f"{node_row['duration_above_threshold']} steps",  "#ffd700",  f"thr={threshold}"),   unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Main drill-down layout ──
    chart_col, stats_col = st.columns([3, 1], gap="medium")

    with chart_col:
        st.markdown(
            "<div class='section-header'>📈 Leak Risk Timeline</div>",
            unsafe_allow_html=True
        )
        if not node_df.empty and len(node_df) > 1:
            fig_risk = plot_risk_over_time(node_df, drill_node, threshold)
            # Make it taller for drill-down view
            fig_risk.update_layout(height=460)
            st.plotly_chart(fig_risk, use_container_width=True,
                            config={"displayModeBar": True})
        elif len(node_df) == 1:
            st.warning(
                f"Node **{drill_node}** has only 1 timestep in this scenario. "
                "This is a data artifact from how `feature_engineering.py` "
                "samples one node per (scenario, time) step — this node "
                "appears in very few rows. Try a different scenario or node."
            )
        else:
            st.info("No data available for this node/scenario combination.")

    with stats_col:
        st.markdown(
            "<div class='section-header'>📋 Node Statistics</div>",
            unsafe_allow_html=True
        )

        if not node_df.empty:
            alert_events   = node_df[node_df["alert_active"]]["time"].values / 3600
            sustained_count= int(node_df["alert_active"].sum())
            first_alert    = f"{alert_events[0]:.1f}h"  if len(alert_events) > 0 else "None"
            last_alert     = f"{alert_events[-1]:.1f}h" if len(alert_events) > 0 else "None"
            time_span      = node_df["time"].max() / 3600
            n_steps        = len(node_df)
            frac_alert     = sustained_count / n_steps if n_steps > 0 else 0

            stats = [
                ("Alert State",        node_row["alert_state"],                          s_color),
                ("Current Risk",       f"{node_row['current_smoothed']:.4f}",          "#00d4ff"),
                ("Peak Probability",   f"{node_row['max_probability']:.4f}",           "#00ffcc"),
                ("Duration Above Thr", f"{node_row['duration_above_threshold']} steps","#ffd700"),
                ("Sustained Alerts",   f"{sustained_count} timesteps",                   "#ff8080"),
                ("% Time in Alert",    f"{frac_alert*100:.1f}%",                         "#ff8080"),
                ("First Alert",        first_alert,                                       "#7a9bb5"),
                ("Last Alert",         last_alert,                                        "#7a9bb5"),
                ("Total Timesteps",    str(n_steps),                                      "#4a6278"),
                ("Monitoring Window",  f"{time_span:.1f}h",                               "#4a6278"),
                ("Smoothing Window",   f"{smoothing_window} steps",                       "#4a6278"),
                ("Threshold",          f"{threshold}",                                    "#4a6278"),
            ]
            for label, val, color in stats:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:baseline;
                padding:0.35rem 0;border-bottom:1px solid #1a2530;font-family:Space Mono;'>
                  <span style='font-size:0.58rem;color:#4a6278;text-transform:uppercase;
                  letter-spacing:0.05em'>{label}</span>
                  <span style='font-size:0.72rem;color:{color};font-weight:600'>{val}</span>
                </div>""", unsafe_allow_html=True)

            # Risk bar
            frac_disp = min(node_row["current_smoothed"], 1.0)
            st.markdown(f"""
            <div style='margin-top:1.2rem;font-family:Space Mono;'>
              <div style='font-size:0.6rem;color:#4a6278;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:0.5rem'>Risk Level</div>
              <div style='background:#1e2d3d;border-radius:3px;height:8px;overflow:hidden;'>
                <div style='background:{s_color};width:{frac_disp*100:.1f}%;
                height:100%;border-radius:3px;'></div>
              </div>
              <div style='display:flex;justify-content:space-between;font-size:0.55rem;
              color:#4a6278;margin-top:0.25rem'>
                <span>0.0</span><span>Thr {threshold}</span><span>1.0</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Alert timeline mini-chart ──
        if not node_df.empty and len(node_df) > 1:
            st.markdown(
                "<div class='section-header' style='margin-top:1.5rem'>⏱ Alert Timeline</div>",
                unsafe_allow_html=True
            )
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(
                x=node_df["time"] / 3600,
                y=node_df["risk_smoothed"],
                mode="lines", line=dict(color="#00d4ff", width=1.5),
                showlegend=False,
                hovertemplate="t=%{x:.1f}h · risk=%{y:.3f}<extra></extra>",
            ))
            # Alert band
            alert_mask_vals = node_df["alert_active"].astype(float).values
            fig_mini.add_trace(go.Scatter(
                x=node_df["time"] / 3600,
                y=alert_mask_vals * node_df["risk_smoothed"].max(),
                mode="lines", fill="tozeroy",
                fillcolor="rgba(255,59,59,0.12)",
                line=dict(color="rgba(255,59,59,0)", width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_mini.add_hline(
                y=threshold,
                line=dict(color="#ffd700", width=1, dash="dot")
            )
            fig_mini.update_layout(
                paper_bgcolor="#0a0e14", plot_bgcolor="#0f1620",
                margin=dict(l=30, r=10, t=10, b=30), height=160,
                xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d",
                           tickfont=dict(color="#7a9bb5", size=8),
                           title=dict(text="Hours", font=dict(color="#7a9bb5", size=9))),
                yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d",
                           tickfont=dict(color="#7a9bb5", size=8)),
                hoverlabel=dict(bgcolor="#141c26", bordercolor="#1e2d3d",
                                font=dict(color="#e8f4f8", size=10)),
            )
            st.plotly_chart(fig_mini, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Bottom back button ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("← Back to Network Map", key="back_bottom"):
        go_to_map()
        st.rerun()


# ── Footer ──
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid #1e2d3d;padding-top:0.8rem;
font-family:Space Mono;font-size:0.6rem;color:#4a6278;
display:flex;justify-content:space-between;'>
  <span>HydraAI — Hydraulic Early Warning System</span>
  <span>XGBoost · WNTR · Streamlit</span>
  <span>P(leak | next 4h)</span>
</div>
""", unsafe_allow_html=True)
