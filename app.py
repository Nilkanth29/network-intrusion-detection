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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #c8d6e5;
}
.stApp { background-color: #0a0e1a; }

h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: #00f5d4; }

.metric-card {
    background: linear-gradient(135deg, #0d1b2a, #1a2744);
    border: 1px solid #00f5d4;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,245,212,0.15);
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #00f5d4; font-family: 'Share Tech Mono', monospace; }
.metric-label { font-size: 0.85rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }

.alert-danger {
    background: linear-gradient(135deg, #2d0a0a, #4a0f0f);
    border: 1px solid #ff4757;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 8px 0;
    color: #ff6b7a;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    box-shadow: 0 0 10px rgba(255,71,87,0.2);
}
.alert-safe {
    background: linear-gradient(135deg, #0a2d1a, #0f3d26);
    border: 1px solid #2ed573;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 8px 0;
    color: #7bed9f;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
}
.stButton > button {
    background: linear-gradient(135deg, #00f5d4, #00b4d8);
    color: #0a0e1a;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    padding: 10px 30px;
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(0,245,212,0.4);
}
.sidebar .sidebar-content { background-color: #0d1b2a; }
[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1a2744; }
.stSelectbox > div > div { background-color: #0d1b2a; border-color: #00f5d4; color: #c8d6e5; }
.stNumberInput > div > div > input { background-color: #0d1b2a; border-color: #00f5d4; color: #c8d6e5; }
hr { border-color: #1a2744; }
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
    st.error(f"⚠️ Could not load model: {e}")

# ── Attack colour map ─────────────────────────────────────────────────────────
ATTACK_COLORS = {
    "BENIGN":        "#2ed573",
    "DDoS":          "#ff4757",
    "PortScan":      "#ffa502",
    "Bot":           "#ff6348",
    "Infiltration":  "#eccc68",
    "Web Attack":    "#ff6b81",
    "DoS":           "#ff4757",
}

def get_attack_color(label):
    for k, v in ATTACK_COLORS.items():
        if k.upper() in label.upper():
            return v
    return "#a29bfe"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ NIDS Dashboard")
    st.markdown("---")
    mode = st.radio(
        "Select Mode",
        ["📊 CSV Analysis", "🔬 Manual Detection", "📈 Model Insights"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Algorithm: `XGBoost`")
    st.markdown("- Dataset: `CICIDS 2017`")
    st.markdown("- Classes: `DDoS, PortScan, Bot, Web Attack, BENIGN`")
    st.markdown("---")
    st.markdown("<small style='color:#7f8c8d'>Built by Nilkanth Changawala</small>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Network Intrusion Detection System")
st.markdown("*Real-time network traffic classification powered by XGBoost + SHAP Explainability*")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — CSV Analysis
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📊 CSV Analysis":
    st.markdown("## 📊 CSV Batch Analysis")
    st.markdown("Upload a network traffic CSV file to classify all flows.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        with st.spinner("Analysing traffic..."):
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()

            # Align columns
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

                # ── Summary metrics ──
                total     = len(df)
                attacks   = (df["Prediction"] != "BENIGN").sum()
                benign    = (df["Prediction"] == "BENIGN").sum()
                attack_pct = round(attacks / total * 100, 1)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Flows</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2ed573">{benign:,}</div><div class="metric-label">Benign</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff4757">{attacks:,}</div><div class="metric-label">Attacks Detected</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ffa502">{attack_pct}%</div><div class="metric-label">Attack Rate</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                # ── Attack breakdown chart ──
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("### Attack Type Breakdown")
                    counts = df["Prediction"].value_counts()
                    colors = [get_attack_color(l) for l in counts.index]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#0d1b2a')
                    ax.set_facecolor('#0d1b2a')
                    bars = ax.barh(counts.index, counts.values, color=colors)
                    ax.set_xlabel("Count", color='#c8d6e5')
                    ax.tick_params(colors='#c8d6e5')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#1a2744')
                    st.pyplot(fig)
                    plt.close()

                with col_right:
                    st.markdown("### Traffic Distribution")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    fig2.patch.set_facecolor('#0d1b2a')
                    ax2.set_facecolor('#0d1b2a')
                    wedge_colors = [get_attack_color(l) for l in counts.index]
                    wedges, texts, autotexts = ax2.pie(
                        counts.values, 
                        colors=wedge_colors,
                        autopct='%1.1f%%',
                        pctdistance=0.75,
                        startangle=90,
                        textprops={'color': '#c8d6e5'}
                    )
                    ax2.legend(
                        wedges, counts.index,
                        loc="center left",
                        bbox_to_anchor=(-0.3, 0, 0.5, 1),
                        fontsize=8,
                        frameon=False,
                        labelcolor='#c8d6e5'
                    )
                    st.pyplot(fig2)
                    plt.close()

                st.markdown("---")

                # ── Attack alerts ──
                st.markdown("### 🚨 Detected Attacks")
                attack_df = df[df["Prediction"] != "BENIGN"][["Prediction"] + feature_cols[:5]].head(20)
                if len(attack_df) == 0:
                    st.markdown('<div class="alert-safe">✅ No attacks detected in this traffic sample.</div>', unsafe_allow_html=True)
                else:
                    for _, row in attack_df.iterrows():
                        color = get_attack_color(row["Prediction"])
                        st.markdown(f'<div class="alert-danger">⚠️ <strong>{row["Prediction"]}</strong> detected</div>', unsafe_allow_html=True)

                # ── SHAP Explanation ──
                st.markdown("---")
                st.markdown("### 🔍 SHAP Explainability — Why was traffic flagged?")
                with st.spinner("Computing SHAP values..."):
                    try:
                        explainer = shap.TreeExplainer(model)
                        sample = X_scaled[:min(100, len(X_scaled))]
                        shap_values = explainer.shap_values(sample)

                        if isinstance(shap_values, list):
                            shap_mean = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
                        else:
                            shap_mean = np.abs(shap_values).mean(axis=0)

                        top_n = 15
                        top_idx = np.argsort(shap_mean)[-top_n:][::-1]
                        top_features = [feature_cols[i] for i in top_idx]
                        top_values   = shap_mean[top_idx]

                        fig3, ax3 = plt.subplots(figsize=(8, 5))
                        fig3.patch.set_facecolor('#0d1b2a')
                        ax3.set_facecolor('#0d1b2a')
                        bars = ax3.barh(top_features[::-1], top_values[::-1], color='#00f5d4')
                        ax3.set_xlabel("Mean |SHAP value|", color='#c8d6e5')
                        ax3.set_title("Top Features Driving Predictions", color='#00f5d4', fontsize=13)
                        ax3.tick_params(colors='#c8d6e5')
                        for spine in ax3.spines.values():
                            spine.set_edgecolor('#1a2744')
                        st.pyplot(fig3)
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP computation skipped: {e}")

                # ── Download results ──
                st.markdown("---")
                csv_out = df[["Prediction"] + feature_cols[:10]].to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", csv_out, "nids_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Manual Detection
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔬 Manual Detection":
    st.markdown("## 🔬 Manual Flow Detection")
    st.markdown("Enter network flow features manually to classify a single connection.")

    st.markdown("### Common Network Flow Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        dest_port        = st.number_input("Destination Port", 0, 65535, 80)
        flow_duration    = st.number_input("Flow Duration (μs)", 0, 10000000, 1000)
        fwd_packets      = st.number_input("Total Fwd Packets", 0, 10000, 10)
        bwd_packets      = st.number_input("Total Bwd Packets", 0, 10000, 5)
    with col2:
        fwd_bytes        = st.number_input("Total Fwd Bytes", 0, 1000000, 500)
        bwd_bytes        = st.number_input("Total Bwd Bytes", 0, 1000000, 200)
        flow_bytes_s     = st.number_input("Flow Bytes/s", 0.0, 10000000.0, 1000.0)
        flow_packets_s   = st.number_input("Flow Packets/s", 0.0, 100000.0, 10.0)
    with col3:
        syn_flag         = st.number_input("SYN Flag Count", 0, 100, 1)
        ack_flag         = st.number_input("ACK Flag Count", 0, 100, 5)
        psh_flag         = st.number_input("PSH Flag Count", 0, 100, 2)
        fin_flag         = st.number_input("FIN Flag Count", 0, 100, 1)

    if st.button("🔍 Analyse Flow"):
        # Build feature row with zeros for all features, fill in known ones
        row = {col: 0.0 for col in feature_cols}
        mapping = {
            "Destination Port": dest_port,
            "Flow Duration": flow_duration,
            "Total Fwd Packets": fwd_packets,
            "Total Backward Packets": bwd_packets,
            "Total Length of Fwd Packets": fwd_bytes,
            "Total Length of Bwd Packets": bwd_bytes,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_packets_s,
            "SYN Flag Count": syn_flag,
            "ACK Flag Count": ack_flag,
            "PSH Flag Count": psh_flag,
            "FIN Flag Count": fin_flag,
        }
        for k, v in mapping.items():
            if k in row:
                row[k] = float(v)

        X = pd.DataFrame([row])[feature_cols]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        label = le.inverse_transform([pred])[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = round(max(proba) * 100, 2)

        st.markdown("---")
        color = get_attack_color(label)
        if label == "BENIGN":
            st.markdown(f'<div class="alert-safe">✅ <strong>BENIGN</strong> — Normal traffic detected | Confidence: {confidence}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-danger">🚨 <strong>ATTACK DETECTED: {label}</strong> | Confidence: {confidence}%</div>', unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("### Prediction Confidence Breakdown")
        classes = le.classes_
        prob_df = pd.DataFrame({"Attack Type": classes, "Probability": proba}).sort_values("Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        bar_colors = [get_attack_color(c) for c in prob_df["Attack Type"]]
        ax.barh(prob_df["Attack Type"], prob_df["Probability"], color=bar_colors)
        ax.set_xlabel("Probability", color='#c8d6e5')
        ax.tick_params(colors='#c8d6e5')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2744')
        st.pyplot(fig)
        plt.close()

        # SHAP for this single prediction
        st.markdown("### 🔍 Why this prediction?")
        with st.spinner("Computing SHAP explanation..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    sv = shap_vals[pred]
                else:
                    sv = shap_vals[0]

                feat_importance = list(zip(feature_cols, sv))
                feat_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                top = feat_importance[:10]

                names  = [f[0] for f in top]
                values = [f[1] for f in top]
                colors = ['#ff4757' if v > 0 else '#2ed573' for v in values]

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor('#0d1b2a')
                ax2.set_facecolor('#0d1b2a')
                ax2.barh(names[::-1], values[::-1], color=colors[::-1])
                ax2.axvline(0, color='#c8d6e5', linewidth=0.8)
                ax2.set_xlabel("SHAP Value (red = increases attack probability)", color='#c8d6e5')
                ax2.set_title("Feature Contributions to This Prediction", color='#00f5d4')
                ax2.tick_params(colors='#c8d6e5')
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#1a2744')
                st.pyplot(fig2)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Model Insights":
    st.markdown("## 📈 Model Insights")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">XGBoost</div><div class="metric-label">Algorithm</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">CICIDS 2017</div><div class="metric-label">Dataset</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value" style="color:#2ed573">Research Grade</div><div class="metric-label">Dataset Quality</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Attack Classes Detected")
    attack_info = {
        "BENIGN":       ("Normal network traffic", "#2ed573"),
        "DDoS":         ("Distributed Denial of Service — floods target with traffic", "#ff4757"),
        "PortScan":     ("Reconnaissance attack scanning for open ports", "#ffa502"),
        "Bot":          ("Automated malicious bot activity", "#ff6348"),
        "Web Attack":   ("SQL injection, XSS, brute force web attacks", "#ff6b81"),
        "Infiltration": ("Internal network infiltration attempts", "#eccc68"),
        "DoS":          ("Denial of Service attack variants", "#a29bfe"),
    }
    for attack, (desc, color) in attack_info.items():
        st.markdown(f'<div style="border-left: 4px solid {color}; padding: 10px 15px; margin: 6px 0; background: #0d1b2a; border-radius: 4px;"><strong style="color:{color}">{attack}</strong> — {desc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How SHAP Explainability Works")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) explains **why** the model made a specific prediction:
    
    - 🔴 **Red bars** → feature pushes prediction toward "Attack"
    - 🟢 **Green bars** → feature pushes prediction toward "Benign"  
    - **Bar length** → how much influence that feature had
    
    This is what makes this project advanced — not just detecting attacks, but explaining the reasoning behind each alert like a real SOC analyst tool.
    """)

    st.markdown("---")
    st.markdown("### Tech Stack")
    cols = st.columns(4)
    techs = [("Python", "#3776ab"), ("XGBoost", "#ff6348"), ("SHAP", "#00f5d4"), ("Streamlit", "#ff4b4b")]
    for col, (tech, color) in zip(cols, techs):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}; font-size:1.4rem">{tech}</div></div>', unsafe_allow_html=True)
