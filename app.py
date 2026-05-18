import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os

# Page config 
st.set_page_config(
    page_title="SENTINEL // NIDS",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Share+Tech+Mono&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f0f4f8;
    color: #1a2535;
}
.stApp { background-color: #f0f4f8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0b1e3d;
    border-right: 3px solid #1565c0;
}
[data-testid="stSidebar"] * { color: #cfd8e3 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #90caf9 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 2px;
}
[data-testid="stSidebar"] .stRadio label { color: #cfd8e3 !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a5f; }

/* Headers */
h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: #0b1e3d !important;
    letter-spacing: 3px !important;
    text-transform: uppercase;
}
h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    color: #0d47a1 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border-top: 4px solid #1565c0;
    border-radius: 4px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0d47a1;
    font-family: 'Share Tech Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #546e7a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
    font-family: 'Barlow Condensed', sans-serif;
}

/* Alert cards */
.alert-danger {
    background: #fff5f5;
    border-left: 5px solid #c62828;
    border-radius: 2px;
    padding: 12px 18px;
    margin: 6px 0;
    color: #b71c1c;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.88rem;
}
.alert-safe {
    background: #f1f8f4;
    border-left: 5px solid #2e7d32;
    border-radius: 2px;
    padding: 12px 18px;
    margin: 6px 0;
    color: #1b5e20;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.88rem;
}
.alert-info {
    background: #e8f0fe;
    border-left: 5px solid #1565c0;
    border-radius: 2px;
    padding: 12px 18px;
    margin: 6px 0;
    color: #0d47a1;
    font-size: 0.88rem;
}

/* Buttons */
.stButton > button {
    background: #1565c0;
    color: #ffffff;
    border: none;
    border-radius: 3px;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 32px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #0d47a1;
    box-shadow: 0 4px 16px rgba(21,101,192,0.3);
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: #ffffff;
    border-color: #90a4ae;
    color: #1a2535;
    border-radius: 3px;
}

/* Divider */
hr { border-color: #cfd8dc; }

/* Top banner stripe */
.top-banner {
    background: linear-gradient(90deg, #0b1e3d 0%, #1565c0 60%, #0b1e3d 100%);
    padding: 12px 24px;
    border-radius: 4px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.banner-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 5px;
    text-transform: uppercase;
}
.banner-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #90caf9;
    letter-spacing: 2px;
}
.status-dot {
    width: 10px; height: 10px;
    background: #43a047;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px #43a047;
}
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 3px;
    color: #78909c;
    text-transform: uppercase;
    margin-bottom: 4px;
    border-bottom: 1px solid #cfd8dc;
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    notebooks = os.path.join(base, "notebooks")
    model        = joblib.load(os.path.join(notebooks, "xgb_model.pkl"))
    scaler       = joblib.load(os.path.join(notebooks, "scaler.pkl"))
    le           = joblib.load(os.path.join(notebooks, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(notebooks, "feature_cols.pkl"))
    return model, scaler, le, feature_cols

try:
    model, scaler, le, feature_cols = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model load error: {e}")

# Attack colour map 
ATTACK_COLORS = {
    "BENIGN":        "#2e7d32",
    "DDoS":          "#c62828",
    "PortScan":      "#e65100",
    "Bot":           "#6a1b9a",
    "Infiltration":  "#f57f17",
    "Web Attack":    "#ad1457",
    "DoS":           "#b71c1c",
}

CHART_BG   = "#ffffff"
CHART_TEXT = "#1a2535"
CHART_GRID = "#eceff1"
ACCENT     = "#1565c0"

def get_attack_color(label):
    for k, v in ATTACK_COLORS.items():
        if k.upper() in label.upper():
            return v
    return "#546e7a"

def style_ax(ax, fig):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_GRID)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.title.set_color(ACCENT)
    ax.grid(axis='x', color=CHART_GRID, linewidth=0.6)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##SENTINEL")
    st.markdown("#### Network Intrusion Detection System")
    st.markdown("---")
    st.markdown('<div class="section-label" style="color:#90caf9">Operation Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "",
        ["📊 CSV Analysis", "🔬 Manual Detection", "📈 Model Insights"],
        index=0
    )
    st.markdown("---")
    st.markdown('<div class="section-label" style="color:#90caf9">System Status</div>', unsafe_allow_html=True)
    if model_loaded:
        st.markdown('<span class="status-dot"></span> **MODEL ONLINE**', unsafe_allow_html=True)
    else:
        st.markdown(' **MODEL OFFLINE**')
    st.markdown("")
    st.markdown('<div class="section-label" style="color:#90caf9">Intelligence</div>', unsafe_allow_html=True)
    st.markdown("- Algorithm: `XGBoost`")
    st.markdown("- Dataset: `CICIDS 2017`")
    st.markdown("- Engine: `SHAP Explainability`")
    st.markdown("- Threats: `DDoS · PortScan · Bot · WebAttack`")
    st.markdown("---")
    st.markdown("<small style='color:#546e7a'>SENTINEL v1.0 // Nilkanth Changawala</small>", unsafe_allow_html=True)

#  Header banner 
st.markdown("""
<div class="top-banner">
    <div>
        <div class="banner-title">🔷 SENTINEL</div>
        <div class="banner-sub">NETWORK INTRUSION DETECTION SYSTEM &nbsp;·&nbsp; XGBOOST + SHAP INTELLIGENCE ENGINE</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("")


# MODE 1 — CSV Analysis
 
if mode == "📊 CSV Analysis":
    st.markdown("## 📊 Batch Traffic Analysis")
    st.markdown('<div class="alert-info">Upload a network traffic CSV file — all flows will be classified and threats identified.</div>', unsafe_allow_html=True)
    st.markdown("")

    uploaded = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

    if uploaded:
        with st.spinner("🔍 Analysing traffic flows..."):
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing[:5]}...")
            else:
                X = df[feature_cols].copy()
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                X.fillna(0, inplace=True)
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                labels = le.inverse_transform(preds)
                df["Prediction"] = labels

                total      = len(df)
                attacks    = (df["Prediction"] != "BENIGN").sum()
                benign     = (df["Prediction"] == "BENIGN").sum()
                attack_pct = round(attacks / total * 100, 1)

                # ── Metrics ──
                st.markdown('<div class="section-label">Threat Summary</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Flows</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2e7d32">{benign:,}</div><div class="metric-label">Benign</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#c62828">{attacks:,}</div><div class="metric-label">Threats Detected</div></div>', unsafe_allow_html=True)
                with c4:
                    threat_color = "#c62828" if attack_pct > 5 else "#e65100" if attack_pct > 1 else "#2e7d32"
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{threat_color}">{attack_pct}%</div><div class="metric-label">Threat Rate</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                # ── Charts ──
                col_left, col_right = st.columns(2)
                counts = df["Prediction"].value_counts()
                bar_colors = [get_attack_color(l) for l in counts.index]

                with col_left:
                    st.markdown('<div class="section-label">Attack Type Breakdown</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    style_ax(ax, fig)
                    ax.barh(counts.index, counts.values, color=bar_colors, height=0.6)
                    ax.set_xlabel("Flow Count", color=CHART_TEXT)
                    ax.set_title("Traffic Classification", color=ACCENT, fontsize=11, fontfamily='sans-serif')
                    st.pyplot(fig)
                    plt.close()

                with col_right:
                    st.markdown('<div class="section-label">Traffic Distribution</div>', unsafe_allow_html=True)
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    fig2.patch.set_facecolor(CHART_BG)
                    ax2.set_facecolor(CHART_BG)
                    wedges, texts, autotexts = ax2.pie(
                        counts.values,
                        colors=bar_colors,
                        autopct='%1.1f%%',
                        pctdistance=0.78,
                        startangle=90,
                        textprops={'color': CHART_TEXT, 'fontsize': 9},
                        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
                    )
                    ax2.legend(wedges, counts.index, loc="center left",
                               bbox_to_anchor=(-0.3, 0, 0.5, 1),
                               fontsize=8, frameon=False, labelcolor=CHART_TEXT)
                    ax2.set_title("Proportion by Category", color=ACCENT, fontsize=11)
                    st.pyplot(fig2)
                    plt.close()

                st.markdown("---")

                # ── Alerts ──
                st.markdown('<div class="section-label">Threat Alerts</div>', unsafe_allow_html=True)
                attack_df = df[df["Prediction"] != "BENIGN"][["Prediction"] + feature_cols[:5]].head(20)
                if len(attack_df) == 0:
                    st.markdown('<div class="alert-safe">✅ CLEAR — No hostile traffic detected in this sample.</div>', unsafe_allow_html=True)
                else:
                    for _, row in attack_df.iterrows():
                        st.markdown(f'<div class="alert-danger">⚠ THREAT DETECTED &nbsp;|&nbsp; <strong>{row["Prediction"]}</strong></div>', unsafe_allow_html=True)

                st.markdown("---")

                # ── SHAP ──
                st.markdown('<div class="section-label">SHAP Intelligence — Why was traffic flagged?</div>', unsafe_allow_html=True)
                with st.spinner("Computing SHAP feature attribution..."):
                    try:
                        explainer  = shap.TreeExplainer(model)
                        sample     = X_scaled[:min(100, len(X_scaled))]
                        shap_values = explainer.shap_values(sample)

                        if isinstance(shap_values, list):
                            shap_mean = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
                        else:
                            shap_mean = np.abs(shap_values).mean(axis=0)

                        top_n = 15
                        top_idx      = np.argsort(shap_mean)[-top_n:][::-1]
                        top_features = [feature_cols[i] for i in top_idx]
                        top_values   = shap_mean[top_idx]

                        fig3, ax3 = plt.subplots(figsize=(8, 5))
                        style_ax(ax3, fig3)
                        colors_shap = [ACCENT] * top_n
                        ax3.barh(top_features[::-1], top_values[::-1], color=colors_shap, height=0.6)
                        ax3.set_xlabel("Mean |SHAP Value|", color=CHART_TEXT)
                        ax3.set_title("Top Features Driving Threat Predictions", color=ACCENT, fontsize=12)
                        st.pyplot(fig3)
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP computation skipped: {e}")

                st.markdown("---")
                csv_out = df[["Prediction"] + feature_cols[:10]].to_csv(index=False)
                st.download_button("⬇ Export Results (CSV)", csv_out, "sentinel_results.csv", "text/csv")

# MODE 2 — Manual Detection

elif mode == "🔬 Manual Detection":
    st.markdown("## 🔬 Manual Flow Analysis")
    st.markdown('<div class="alert-info">Enter network flow parameters manually to classify a single connection.</div>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('<div class="section-label">Flow Parameters</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        dest_port     = st.number_input("Destination Port",    0, 65535,    80)
        flow_duration = st.number_input("Flow Duration (μs)",  0, 10000000, 1000)
        fwd_packets   = st.number_input("Total Fwd Packets",   0, 10000,    10)
        bwd_packets   = st.number_input("Total Bwd Packets",   0, 10000,    5)
    with col2:
        fwd_bytes     = st.number_input("Total Fwd Bytes",     0, 1000000,  500)
        bwd_bytes     = st.number_input("Total Bwd Bytes",     0, 1000000,  200)
        flow_bytes_s  = st.number_input("Flow Bytes/s",        0.0, 10000000.0, 1000.0)
        flow_pkt_s    = st.number_input("Flow Packets/s",      0.0, 100000.0,   10.0)
    with col3:
        syn_flag      = st.number_input("SYN Flag Count",      0, 100, 1)
        ack_flag      = st.number_input("ACK Flag Count",      0, 100, 5)
        psh_flag      = st.number_input("PSH Flag Count",      0, 100, 2)
        fin_flag      = st.number_input("FIN Flag Count",      0, 100, 1)

    st.markdown("")
    if st.button("🔍 ANALYSE FLOW"):
        row = {col: 0.0 for col in feature_cols}
        mapping = {
            "Destination Port":              dest_port,
            "Flow Duration":                 flow_duration,
            "Total Fwd Packets":             fwd_packets,
            "Total Backward Packets":        bwd_packets,
            "Total Length of Fwd Packets":   fwd_bytes,
            "Total Length of Bwd Packets":   bwd_bytes,
            "Flow Bytes/s":                  flow_bytes_s,
            "Flow Packets/s":                flow_pkt_s,
            "SYN Flag Count":                syn_flag,
            "ACK Flag Count":                ack_flag,
            "PSH Flag Count":                psh_flag,
            "FIN Flag Count":                fin_flag,
        }
        for k, v in mapping.items():
            if k in row:
                row[k] = float(v)

        X        = pd.DataFrame([row])[feature_cols]
        X_scaled = scaler.transform(X)
        pred     = model.predict(X_scaled)[0]
        label    = le.inverse_transform([pred])[0]
        proba    = model.predict_proba(X_scaled)[0]
        confidence = round(max(proba) * 100, 2)

        st.markdown("---")
        st.markdown('<div class="section-label">Classification Result</div>', unsafe_allow_html=True)
        if label == "BENIGN":
            st.markdown(f'<div class="alert-safe">✅ STATUS: CLEAR &nbsp;|&nbsp; Classification: <strong>BENIGN</strong> &nbsp;|&nbsp; Confidence: {confidence}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-danger">⚠ STATUS: THREAT &nbsp;|&nbsp; Classification: <strong>{label}</strong> &nbsp;|&nbsp; Confidence: {confidence}%</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-label">Probability Breakdown by Threat Class</div>', unsafe_allow_html=True)
        classes  = le.classes_
        prob_df  = pd.DataFrame({"Attack Type": classes, "Probability": proba}).sort_values("Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        style_ax(ax, fig)
        bar_colors = [get_attack_color(c) for c in prob_df["Attack Type"]]
        ax.barh(prob_df["Attack Type"], prob_df["Probability"], color=bar_colors, height=0.55)
        ax.set_xlabel("Probability", color=CHART_TEXT)
        ax.set_title("Threat Class Probabilities", color=ACCENT, fontsize=11)
        st.pyplot(fig)
        plt.close()

        st.markdown("")
        st.markdown('<div class="section-label">SHAP Feature Attribution</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP explanation..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_scaled)
                sv = shap_vals[pred] if isinstance(shap_vals, list) else shap_vals[0]

                feat_importance = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)
                top    = feat_importance[:10]
                names  = [f[0] for f in top]
                values = [f[1] for f in top]
                colors = ['#c62828' if v > 0 else '#2e7d32' for v in values]

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                style_ax(ax2, fig2)
                ax2.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
                ax2.axvline(0, color='#90a4ae', linewidth=1)
                ax2.set_xlabel("SHAP Value  (red → threat signal  |  green → benign signal)", color=CHART_TEXT)
                ax2.set_title("Feature Contributions to This Classification", color=ACCENT, fontsize=11)
                st.pyplot(fig2)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")

# MODE 3 — Model Insights

elif mode == "📈 Model Insights":
    st.markdown("## 📈 System Intelligence")
    st.markdown("")

    st.markdown('<div class="section-label">System Specifications</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">XGBoost</div><div class="metric-label">Algorithm</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">CICIDS 2017</div><div class="metric-label">Training Dataset</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color:#2e7d32; font-size:1.4rem">OPERATIONAL</div><div class="metric-label">System Status</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Threat Classification Matrix</div>', unsafe_allow_html=True)
    attack_info = {
        "BENIGN":       ("Normal authorised network traffic", "#2e7d32"),
        "DDoS":         ("Distributed Denial of Service — coordinated traffic flood", "#c62828"),
        "PortScan":     ("Reconnaissance — systematic scanning for open ports", "#e65100"),
        "Bot":          ("Automated malicious bot activity on the network", "#6a1b9a"),
        "Web Attack":   ("SQL injection, XSS, and brute force against web services", "#ad1457"),
        "Infiltration": ("Internal network infiltration and lateral movement", "#f57f17"),
        "DoS":          ("Single-source Denial of Service attack variants", "#b71c1c"),
    }
    for attack, (desc, color) in attack_info.items():
        st.markdown(
            f'<div style="border-left:4px solid {color}; padding:10px 16px; margin:5px 0; '
            f'background:#ffffff; border-radius:2px; box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
            f'<strong style="color:{color}; font-family:\'Barlow Condensed\',sans-serif; '
            f'font-size:1.05rem; letter-spacing:1px">{attack}</strong>'
            f'<span style="color:#546e7a; font-size:0.9rem; margin-left:12px">{desc}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-label">SHAP Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    SHAP (SHapley Additive exPlanations) provides analyst-grade reasoning for every classification:

    - 🔴 **Red bars** — feature increases threat probability
    - 🟢 **Green bars** — feature supports benign classification
    - **Bar magnitude** — strength of that feature's influence on the decision

    This transforms raw ML predictions into interpretable, actionable intelligence — the standard expected in real Security Operations Centres.
    """)

    st.markdown("---")
    st.markdown('<div class="section-label">Technology Stack</div>', unsafe_allow_html=True)
    cols   = st.columns(4)
    techs  = [("Python", "#1565c0"), ("XGBoost", "#e65100"), ("SHAP", "#6a1b9a"), ("Streamlit", "#c62828")]
    for col, (tech, color) in zip(cols, techs):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}; font-size:1.5rem">{tech}</div></div>', unsafe_allow_html=True)