"""
app.py — SONA (Security of Network Access) — Streamlit Dashboard
=================================================================
Run with:  streamlit run dashboard/app.py

Make sure you have trained the model first:
    cd sona
    python src/train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from predict import load_artifacts, predict_single, EXAMPLE_NORMAL, EXAMPLE_DOS
from preprocess import CATEGORICAL_COLS

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SONA — Security of Network Access",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .attack-badge  { background:#FEE2E2; color:#991B1B; padding:4px 12px;
                     border-radius:20px; font-weight:600; font-size:14px; }
    .normal-badge  { background:#D1FAE5; color:#065F46; padding:4px 12px;
                     border-radius:20px; font-weight:600; font-size:14px; }
    .metric-card   { background:#F8FAFC; border:1px solid #E2E8F0;
                     border-radius:8px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        return load_artifacts(model_dir=os.path.join(
            os.path.dirname(__file__), "..", "models"
        ))
    except FileNotFoundError:
        return None


artifacts = load_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ SONA")
    st.markdown("Security of Network Access")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🔍 Live prediction", "📊 Model performance", "📖 About this project"],
        label_visibility="collapsed",
    )
    st.divider()
    if artifacts:
        st.success("Model loaded ✓")
        st.caption(f"Labels: {', '.join(artifacts['label_names'])}")
    else:
        st.error("No trained model found.\nRun `python src/train.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Live prediction
# ─────────────────────────────────────────────────────────────────────────────

if page == "🔍 Live prediction":
    st.title("🔍 Live traffic prediction")
    st.markdown("Enter network connection features below, or load an example, and the model will classify the traffic.")

    if not artifacts:
        st.error("Please train the model first: `cd sona && python src/train.py`")
        st.stop()

    col_load, _ = st.columns([2, 6])
    with col_load:
        example = st.selectbox("Load example", ["Custom", "Normal traffic", "DoS attack"])

    defaults = (
        EXAMPLE_NORMAL if example == "Normal traffic"
        else EXAMPLE_DOS if example == "DoS attack"
        else EXAMPLE_NORMAL
    )

    st.subheader("Connection features")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Basic**")
        duration    = st.number_input("duration (s)", 0, 60000, int(defaults["duration"]))
        protocol    = st.selectbox("protocol_type", ["tcp","udp","icmp"],
                        index=["tcp","udp","icmp"].index(defaults["protocol_type"]))
        service     = st.selectbox("service",
                        ["http","ftp","smtp","ssh","dns","pop3","https","other"],
                        index=0)
        flag        = st.selectbox("flag",
                        ["SF","S0","S1","S2","S3","REJ","RSTO","RSTR","SH","OTH"],
                        index=["SF","S0","S1","S2","S3","REJ","RSTO","RSTR","SH","OTH"].index(
                            defaults["flag"]) if defaults["flag"] in
                            ["SF","S0","S1","S2","S3","REJ","RSTO","RSTR","SH","OTH"] else 0)
        src_bytes   = st.number_input("src_bytes",  0, 10_000_000, int(defaults["src_bytes"]))
        dst_bytes   = st.number_input("dst_bytes",  0, 10_000_000, int(defaults["dst_bytes"]))
        logged_in   = st.selectbox("logged_in", [0, 1], index=int(defaults["logged_in"]))

    with c2:
        st.markdown("**Connection stats**")
        count         = st.slider("count",           0, 512, int(defaults["count"]))
        srv_count     = st.slider("srv_count",       0, 512, int(defaults["srv_count"]))
        serror_rate   = st.slider("serror_rate",     0.0, 1.0, float(defaults["serror_rate"]))
        rerror_rate   = st.slider("rerror_rate",     0.0, 1.0, float(defaults["rerror_rate"]))
        same_srv_rate = st.slider("same_srv_rate",   0.0, 1.0, float(defaults["same_srv_rate"]))
        diff_srv_rate = st.slider("diff_srv_rate",   0.0, 1.0, float(defaults["diff_srv_rate"]))

    with c3:
        st.markdown("**Host stats**")
        dst_host_count         = st.slider("dst_host_count",         0, 255, int(defaults["dst_host_count"]))
        dst_host_srv_count     = st.slider("dst_host_srv_count",     0, 255, int(defaults["dst_host_srv_count"]))
        dst_host_same_srv_rate = st.slider("dst_host_same_srv_rate", 0.0, 1.0, float(defaults["dst_host_same_srv_rate"]))
        dst_host_serror_rate   = st.slider("dst_host_serror_rate",   0.0, 1.0, float(defaults["dst_host_serror_rate"]))
        dst_host_rerror_rate   = st.slider("dst_host_rerror_rate",   0.0, 1.0, float(defaults["dst_host_rerror_rate"]))
        num_failed_logins      = st.number_input("num_failed_logins", 0, 10, int(defaults["num_failed_logins"]))
        root_shell             = st.selectbox("root_shell", [0, 1], index=int(defaults["root_shell"]))

    # Build record from inputs
    record = {**defaults}
    record.update({
        "duration": duration, "protocol_type": protocol,
        "service": service, "flag": flag,
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "logged_in": logged_in, "count": count, "srv_count": srv_count,
        "serror_rate": serror_rate, "rerror_rate": rerror_rate,
        "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
        "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": dst_host_same_srv_rate,
        "dst_host_serror_rate": dst_host_serror_rate,
        "dst_host_rerror_rate": dst_host_rerror_rate,
        "num_failed_logins": num_failed_logins, "root_shell": root_shell,
    })

    st.divider()
    if st.button("🔍 Analyse this connection", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            result = predict_single(record, artifacts)

        st.subheader("Prediction result")
        r1, r2, r3 = st.columns(3)

        badge = (
            f'<span class="attack-badge">⚠️ {result["predicted_class"]}</span>'
            if result["is_attack"]
            else f'<span class="normal-badge">✅ {result["predicted_class"]}</span>'
        )
        r1.markdown(f"**Classification**<br>{badge}", unsafe_allow_html=True)
        r2.metric("Confidence", f"{result['confidence']:.1%}")
        r3.metric("Threat level", "HIGH" if result["is_attack"] else "LOW")

        # Probability bar chart
        proba_df = pd.DataFrame(
            list(result["all_probabilities"].items()),
            columns=["Class", "Probability"]
        ).sort_values("Probability", ascending=True)

        fig = px.bar(
            proba_df, x="Probability", y="Class",
            orientation="h", range_x=[0, 1],
            color="Probability",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            title="Prediction probabilities per class"
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Model performance
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Model performance":
    st.title("📊 Model performance")

    # Load saved plots if they exist
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")

    if not os.path.exists(plot_dir) or not os.listdir(plot_dir):
        st.info("No plots found. Run `python src/train.py` first to generate performance charts.")
        st.code("cd sona\npython src/train.py")
        st.stop()

    # Model comparison
    cmp_path = os.path.join(plot_dir, "model_comparison.png")
    if os.path.exists(cmp_path):
        st.subheader("Model comparison")
        st.image(cmp_path, use_column_width=True)

    # Confusion matrices
    st.subheader("Confusion matrices")
    cm_files = sorted([f for f in os.listdir(plot_dir) if f.startswith("cm_")])
    if cm_files:
        tabs = st.tabs([f.replace("cm_", "").replace(".png", "").replace("_", " ").title()
                        for f in cm_files])
        for tab, fname in zip(tabs, cm_files):
            with tab:
                st.image(os.path.join(plot_dir, fname), use_column_width=True)

    # Feature importance
    st.subheader("Feature importance")
    fi_files = sorted([f for f in os.listdir(plot_dir) if f.startswith("fi_")])
    if fi_files:
        tabs = st.tabs([f.replace("fi_", "").replace(".png", "").replace("_", " ").title()
                        for f in fi_files])
        for tab, fname in zip(tabs, fi_files):
            with tab:
                st.image(os.path.join(plot_dir, fname), use_column_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: About
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📖 About this project":
    st.title("📖 About SONA")

    st.markdown("""
    ## SONA — Security of Network Access

    SONA is an ML-powered network intrusion detection system that analyses
    network connection records and classifies them as normal or one of four
    attack categories in real time.

    ### Dataset — NSL-KDD
    The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset,
    widely used in cybersecurity research. It contains:
    - **125,973** training records
    - **22,544** test records
    - **41 features** per connection
    - **5 classes**: Normal, DoS, Probe, R2L, U2R

    ### Attack categories

    | Category | Description | Examples |
    |---|---|---|
    | Normal | Legitimate traffic | HTTP browsing, FTP transfers |
    | DoS | Denial of Service — floods the target | Neptune, Smurf, Pod |
    | Probe | Network scanning / reconnaissance | Nmap, Portsweep, Satan |
    | R2L | Remote to Local — gain local access | Guess_passwd, FTP_write |
    | U2R | User to Root — privilege escalation | Buffer overflow, Rootkit |

    ### Models trained

    | Model | Strengths |
    |---|---|
    | Logistic Regression | Fast baseline, interpretable |
    | Decision Tree | Visualizable, easy to explain |
    | Random Forest | Robust, handles class imbalance |
    | XGBoost | Best overall performance |

    ### Tech stack
    Python · Pandas · Scikit-learn · XGBoost · imbalanced-learn · Streamlit · Plotly

    ### Download dataset
    [NSL-KDD — University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
    """)
