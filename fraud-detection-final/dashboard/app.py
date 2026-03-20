"""
dashboard/app.py — Premium Fraud Detection Dashboard (Streamlit)

Run:  streamlit run dashboard/app.py

Improvements vs original:
  - Full UI redesign with professional dark theme
  - API_URL loaded from environment variable (no hardcoded localhost)
  - Model metrics loaded dynamically from API (not hardcoded strings)
  - Built-in onboarding tutorial page
  - Batch CSV upload and scoring
  - Feature importance sidebar visualization
  - Proper error handling and user feedback
  - All state management via session_state
"""

import os
import time as time_module

import pandas as pd
import plotly.graph_objects as go
import random
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark, professional theme ─────────────────────────────────────
st.markdown("""
<style>
/* ── Base & font ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ───────────────────────────── */
.stApp {
    background: #0d0f1a;
    color: #e4e6f0;
}

/* ── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111320 !important;
    border-right: 1px solid rgba(99,102,241,0.15);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #9198b5;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 4px;
}

/* ── Metric cards ─────────────────────────────── */
[data-testid="metric-container"] {
    background: #161929 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #6b7db3 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e4e6f0 !important;
    font-size: 26px !important;
    font-weight: 600 !important;
}

/* ── Buttons ──────────────────────────────────── */
.stButton > button {
    background: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #4f46e5 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e2235 !important;
    color: #9198b5 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

/* ── Inputs & selects ─────────────────────────── */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #1a1d2e !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #e4e6f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Cards ─────────────────────────────────────── */
.fraud-card {
    background: #161929;
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.result-safe {
    background: linear-gradient(135deg, #0f2920 0%, #112a1e 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.result-medium {
    background: linear-gradient(135deg, #2a200a 0%, #1f1a08 100%);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.result-fraud {
    background: linear-gradient(135deg, #2a0d0d 0%, #1f0b0b 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.verdict-title {
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 6px 0;
}
.verdict-sub {
    font-size: 14px;
    color: #9198b5;
    margin: 0;
}
.prob-text {
    font-size: 40px;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    margin: 0.5rem 0;
}
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 6px;
}
.tag-low    { background: rgba(16,185,129,0.15);  color: #10b981; border: 1px solid rgba(16,185,129,0.3);  }
.tag-medium { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.3);  }
.tag-high   { background: rgba(239,68,68,0.15);   color: #ef4444; border: 1px solid rgba(239,68,68,0.3);   }
.section-header {
    font-size: 11px;
    font-weight: 600;
    color: #6b7db3;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(99,102,241,0.1);
}
.latency-badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
}
/* ── Divider ──────────────────────────────────── */
hr {
    border-color: rgba(99,102,241,0.12) !important;
}
/* ── Tab styling ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #161929 !important;
    border-radius: 8px !important;
    color: #6b7db3 !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    border-color: rgba(99,102,241,0.4) !important;
}
/* ── File uploader ────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #161929 !important;
    border: 1px dashed rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
}
/* ── DataFrame ────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}
/* ── Info / warning / error boxes ─────────────── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
/* ── Scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
/* ── Remove default padding ───────────────────── */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
/* ── Plotly charts background ─────────────────── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "amount":    149.99,
        "time_val":  0.0,
        "merchant_category": "Other",
        "transaction_location": "Home",
        "transaction_type": "In-person (Chip)",
        "is_new_device": False,
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _api_alive() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _make_gauge(prob: float, risk: str) -> go.Figure:
    colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    bar_color = colors.get(risk, "#6366f1")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#e4e6f0", "family": "DM Mono"}},
        title={"text": "Fraud Risk Score", "font": {"size": 14, "color": "#6b7db3", "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#4a5280"}, "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20],  "color": "rgba(16,185,129,0.08)"},
                {"range": [20, 60], "color": "rgba(245,158,11,0.08)"},
                {"range": [60, 100],"color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return fig


def _make_feature_bar(v_vals: list) -> go.Figure:
    """Show the top contributing V features (absolute value = magnitude)."""
    labels = [f"V{i+1}" for i in range(28)]
    values = [abs(v) for v in v_vals]
    top_idx = sorted(range(28), key=lambda i: values[i], reverse=True)[:10]
    top_labels = [labels[i] for i in top_idx]
    top_values = [v_vals[i] for i in top_idx]
    bar_colors = ["#ef4444" if v < 0 else "#10b981" for v in top_values]
    fig = go.Figure(go.Bar(
        x=[abs(v) for v in top_values],
        y=top_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in top_values],
        textposition="outside",
        textfont={"size": 11, "color": "#9198b5", "family": "DM Mono"},
    ))
    fig.update_layout(
        title={"text": "Top 10 Feature Magnitudes", "font": {"size": 13, "color": "#6b7db3"}},
        height=300,
        margin=dict(t=40, b=10, l=20, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"gridcolor": "rgba(99,102,241,0.08)", "color": "#4a5280"},
        yaxis={"color": "#9198b5"},
        font={"family": "DM Mono"},
    )
    return fig


def _risk_card(result: dict, latency: float):
    risk = result["risk_level"]
    prob = result["fraud_probability"]
    rec  = result["recommendation"]
    css_class = {"LOW": "result-safe", "MEDIUM": "result-medium", "HIGH": "result-fraud"}.get(risk, "fraud-card")
    tag_class  = {"LOW": "tag-low",    "MEDIUM": "tag-medium",    "HIGH": "tag-high"}.get(risk, "tag-low")
    prob_color = {"LOW": "#10b981",    "MEDIUM": "#f59e0b",       "HIGH": "#ef4444"}.get(risk, "#6366f1")

    st.markdown(f"""
    <div class="{css_class}">
        <span class="tag {tag_class}">{risk} RISK</span>
        <span class="latency-badge">{latency} ms</span>
        <div class="prob-text" style="color:{prob_color}">{prob*100:.2f}%</div>
        <p class="verdict-sub">{rec}</p>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(99,102,241,0.15)">
        <div style="font-size:28px">🛡️</div>
        <div>
            <div style="font-size:17px;font-weight:600;color:#e4e6f0">FraudShield</div>
            <div style="font-size:11px;color:#6b7db3">v1.0 · XGBoost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    alive = _api_alive()
    status_color = "#10b981" if alive else "#ef4444"
    status_text  = "API Connected" if alive else "API Offline"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;background:#161929;
                border:1px solid rgba(99,102,241,0.15);border-radius:8px;
                padding:8px 12px;margin-bottom:1.5rem">
        <div style="width:8px;height:8px;border-radius:50%;background:{status_color};
                    box-shadow:0 0 6px {status_color}"></div>
        <span style="font-size:12px;color:{status_color};font-weight:500">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("NAVIGATION")
    page = st.radio(
        "",
        ["🔍  Score Transaction", "📦  Batch Analysis", "📊  Model Performance", "🎓  Tutorial"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("ABOUT")
    st.caption(f"API endpoint: `{API_URL}`")
    st.caption("Dataset: Kaggle Credit Card Fraud (284,807 transactions, 0.17% fraud)")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#4a5280;font-size:10px;letter-spacing:0.05em">
        © 2026 · DEVELOPED BY<br>
        <span style="color:#6366f1;font-weight:600;font-size:11px">DEVANSH SHARMA</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Score a single transaction
# ═══════════════════════════════════════════════════════════════════════════════
if "Score" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:600;color:#e4e6f0;margin-bottom:4px">
        Score a Transaction
    </h1>
    <p style="color:#6b7db3;font-size:14px;margin-bottom:1.5rem">
        Enter transaction details and click <strong style="color:#a5b4fc">Run Analysis</strong> to get a fraud score.
    </p>
    """, unsafe_allow_html=True)

    # ── Quick-fill generators ──────────────────────────────────────────────────
    col_l, col_m, col_r = st.columns([1, 1, 2])

    with col_l:
        st.markdown('<p class="section-header">Quick Fill</p>', unsafe_allow_html=True)
        if st.button("🎲 Generate Legit", use_container_width=True):
            st.session_state.merchant_category = "Groceries"
            st.session_state.transaction_location = "Home"
            st.session_state.transaction_type = "In-person (Chip)"
            st.session_state.is_new_device = False
            st.session_state.amount = round(random.uniform(10, 300), 2)
            st.session_state.time_val = round(random.uniform(0, 172800), 0)
            st.rerun()

        if st.button("🚨 Generate Fraud", use_container_width=True):
            st.session_state.merchant_category = "Luxury Goods"
            st.session_state.transaction_location = "International (High-risk)"
            st.session_state.transaction_type = "Manual Entry"
            st.session_state.is_new_device = True
            st.session_state.amount = round(random.uniform(400, 2000), 2)
            st.session_state.time_val = round(random.uniform(0, 172800), 0)
            st.rerun()

        if st.button("⬜ Reset Defaults", use_container_width=True):
            st.session_state.merchant_category = "Other"
            st.session_state.transaction_location = "Home"
            st.session_state.transaction_type = "In-person (Chip)"
            st.session_state.is_new_device = False
            st.session_state.amount = 149.99
            st.session_state.time_val = 0.0
            st.rerun()

    with col_m:
        st.markdown('<p class="section-header">Transaction Details</p>', unsafe_allow_html=True)
        amount    = st.number_input("Amount ($)", min_value=0.01, value=float(st.session_state.amount),  step=0.01, format="%.2f")
        time_input = st.number_input("Time (seconds)", min_value=0.0, value=float(st.session_state.time_val), step=1.0)

        st.markdown("""
        <p style="font-size:11px;color:#4a5280;margin-top:8px">
        <strong style="color:#6b7db3">Amount</strong>: Transaction value in USD<br>
        <strong style="color:#6b7db3">Time</strong>: Seconds since dataset start (0–172,800)
        </p>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-header">User-Friendly Details</p>', unsafe_allow_html=True)
        st.caption("These fields are used by the AI to assess risk when technical PCA data is unavailable.")
        
        m_categories = ["Groceries", "Electronics", "Luxury Goods", "Entertainment", "Online Services", "Gas Station", "Pharmacy", "Other"]
        l_locations = ["Home", "Domestic (Different City)", "International (Low-risk)", "International (High-risk)"]
        t_types = ["In-person (Chip)", "In-person (Swipe)", "Online", "Recurring", "Manual Entry"]
        
        merchant = st.selectbox("Merchant Category", m_categories, index=m_categories.index(st.session_state.merchant_category))
        location = st.selectbox("Transaction Location", l_locations, index=l_locations.index(st.session_state.transaction_location))
        t_type   = st.selectbox("Transaction Type", t_types, index=t_types.index(st.session_state.transaction_type))
        new_device = st.checkbox("New Device or Session", value=st.session_state.is_new_device)

    st.divider()

    # ── Analysis button & results ──────────────────────────────────────────────
    run_col, _ = st.columns([1, 2])
    with run_col:
        run = st.button("🔮  Run Analysis", type="primary", use_container_width=True)

    if run:
        if not alive:
            st.error("❌ Cannot reach API. Make sure the FastAPI server is running:\n```\nuvicorn api.main:app --reload\n```")
        else:
            payload = {
                "Amount": amount,
                "Time": time_input,
                "merchant_category": merchant,
                "transaction_location": location,
                "transaction_type": t_type,
                "is_new_device": new_device
            }

            with st.spinner("Analyzing transaction…"):
                t0  = time_module.time()
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    wall = round((time_module.time() - t0) * 1000, 1)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.last_result = result

                        # Results layout
                        res_a, res_b = st.columns([1, 1])
                        with res_a:
                            _risk_card(result, wall)
                        with res_b:
                            st.plotly_chart(_make_gauge(result["fraud_probability"], result["risk_level"]), use_container_width=True)

                        # Note about synthetic data
                        st.info("💡 **Note:** Since PCA features (V1-V28) were not provided, the AI generated a synthetic risk profile based on your user-friendly inputs.")

                        # Raw JSON expander
                        with st.expander("View raw API response"):
                            st.json(result)
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is Uvicorn running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch analysis (CSV upload)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:600;color:#e4e6f0;margin-bottom:4px">
        Batch Transaction Analysis
    </h1>
    <p style="color:#6b7db3;font-size:14px;margin-bottom:1.5rem">
        Upload a CSV file to score up to 100 transactions at once.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fraud-card">
        <p style="font-size:13px;color:#9198b5;margin:0">
        <strong style="color:#a5b4fc">CSV format:</strong>
        One row per transaction. Required columns: <code style="font-family:'DM Mono';color:#818cf8">V1</code> through
        <code style="font-family:'DM Mono';color:#818cf8">V28</code>,
        <code style="font-family:'DM Mono';color:#818cf8">Amount</code>,
        <code style="font-family:'DM Mono';color:#818cf8">Time</code>.
        Maximum 100 rows per upload.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
            missing  = [c for c in required if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            elif len(df) > 100:
                st.warning(f"File has {len(df)} rows. Only the first 100 will be scored.")
                df = df.head(100)
            else:
                st.success(f"Loaded {len(df)} transactions.")
                st.dataframe(df.head(5), use_container_width=True)

                if st.button("🚀 Score All Transactions", type="primary"):
                    transactions = []
                    for _, row in df.iterrows():
                        tx = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
                        tx["Amount"] = float(row["Amount"])
                        tx["Time"]   = float(row["Time"])
                        transactions.append(tx)

                    with st.spinner(f"Scoring {len(transactions)} transactions…"):
                        try:
                            resp = requests.post(f"{API_URL}/predict/batch", json=transactions, timeout=30)
                            if resp.status_code == 200:
                                results = resp.json()
                                df_out = df.copy()
                                df_out["fraud_probability"] = [r["fraud_probability"] for r in results]
                                df_out["risk_level"]        = [r["risk_level"]        for r in results]
                                df_out["is_fraud"]          = [r["is_fraud"]           for r in results]
                                df_out["recommendation"]    = [r["recommendation"]     for r in results]

                                # Summary metrics
                                total  = len(df_out)
                                frauds = df_out["is_fraud"].sum()
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Total Scored",  total)
                                m2.metric("Flagged Fraud", int(frauds))
                                m3.metric("Fraud Rate",    f"{frauds/total*100:.1f}%")
                                m4.metric("High Risk",     int((df_out["risk_level"] == "HIGH").sum()))

                                # Results table with colour coding
                                st.markdown('<p class="section-header">Detailed Results</p>', unsafe_allow_html=True)

                                def _color_risk(val):
                                    colors = {"LOW": "color:#10b981", "MEDIUM": "color:#f59e0b", "HIGH": "color:#ef4444"}
                                    return colors.get(val, "")

                                display_cols = ["Amount", "fraud_probability", "risk_level", "is_fraud", "recommendation"]
                                st.dataframe(
                                    df_out[display_cols].style.applymap(_color_risk, subset=["risk_level"]),
                                    use_container_width=True,
                                )

                                # Download
                                csv_out = df_out.to_csv(index=False)
                                st.download_button(
                                    "⬇️  Download Results CSV",
                                    data=csv_out,
                                    file_name="fraud_results.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.error(f"API error: {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:600;color:#e4e6f0;margin-bottom:4px">
        Model Performance
    </h1>
    <p style="color:#6b7db3;font-size:14px;margin-bottom:1.5rem">
        Live metrics loaded from the API — always reflects the currently deployed model.
    </p>
    """, unsafe_allow_html=True)

    try:
        resp = requests.get(f"{API_URL}/metrics/model", timeout=5)
        if resp.status_code == 200:
            m = resp.json()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC",   f"{m.get('roc_auc', 'N/A')}")
            c2.metric("PR-AUC",    f"{m.get('pr_auc', 'N/A')}")
            c3.metric("Precision", f"{m.get('precision', 'N/A')}")
            c4.metric("Recall",    f"{m.get('recall', 'N/A')}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("F1-Score",       f"{m.get('f1_score', 'N/A')}")
            c6.metric("Fraud Catch Rate", f"{m.get('fraud_catch_rate', 'N/A')}%")
            c7.metric("False Alarm Rate", f"{m.get('false_alarm_rate', 'N/A')}%")
            c8.metric("Dataset Size",   f"{m.get('dataset', {}).get('total_rows', 'N/A'):,}" if m.get('dataset') else "N/A")

            # Confusion matrix viz
            st.divider()
            st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
            cm = m.get("confusion_matrix", {})
            if cm:
                tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
                fig_cm = go.Figure(go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=["Predicted: Legit", "Predicted: Fraud"],
                    y=["Actual: Legit", "Actual: Fraud"],
                    text=[[f"TN\n{tn:,}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]],
                    texttemplate="%{text}",
                    colorscale=[[0, "#161929"], [0.5, "#4338ca"], [1, "#6366f1"]],
                    showscale=False,
                    textfont={"size": 14, "color": "#e4e6f0", "family": "DM Mono"},
                ))
                fig_cm.update_layout(
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "DM Sans", "color": "#9198b5"},
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                col_cm, col_exp = st.columns([1, 1])
                with col_cm:
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col_exp:
                    st.markdown(f"""
                    <div class="fraud-card" style="margin-top:1rem">
                    <p style="color:#9198b5;font-size:13px;line-height:1.8">
                    <strong style="color:#10b981">TN {tn:,}</strong> — Legit transactions correctly approved<br>
                    <strong style="color:#ef4444">FP {fp}</strong> — Legit transactions incorrectly blocked (false alarms)<br>
                    <strong style="color:#f59e0b">FN {fn}</strong> — Fraud transactions that slipped through (missed)<br>
                    <strong style="color:#6366f1">TP {tp}</strong> — Fraud transactions correctly blocked
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("View raw metrics JSON"):
                st.json(m)
        else:
            st.warning("Metrics file not found. Run `python model/train.py` first.")
    except Exception as e:
        st.error(f"Could not load metrics from API: {e}")
        st.info("Make sure the FastAPI server is running: `uvicorn api.main:app --reload`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Tutorial
# ═══════════════════════════════════════════════════════════════════════════════
elif "Tutorial" in page:
    st.markdown("""
    <h1 style="font-size:26px;font-weight:600;color:#e4e6f0;margin-bottom:4px">
        Getting Started Guide
    </h1>
    <p style="color:#6b7db3;font-size:14px;margin-bottom:1.5rem">
        Everything you need to understand and use FraudShield from scratch.
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🏗️  System Overview", "🔍  Score a Transaction", "📦  Batch Scoring", "🧠  Understanding the Output"])

    with tab1:
        st.markdown("""
        <div class="fraud-card">
        <h3 style="color:#a5b4fc;margin-top:0">What is FraudShield?</h3>
        <p style="color:#9198b5;font-size:14px;line-height:1.8">
        FraudShield is an end-to-end machine learning system that detects fraudulent credit card
        transactions in real time. It consists of three components:
        </p>
        </div>
        """, unsafe_allow_html=True)

        arch1, arch2, arch3 = st.columns(3)
        with arch1:
            st.markdown("""
            <div class="fraud-card" style="text-align:center">
            <div style="font-size:32px;margin-bottom:8px">🧠</div>
            <div style="font-weight:600;color:#e4e6f0;margin-bottom:6px">ML Model</div>
            <div style="font-size:12px;color:#6b7db3">XGBoost classifier trained on 284,807 transactions with SMOTE balancing</div>
            </div>
            """, unsafe_allow_html=True)
        with arch2:
            st.markdown("""
            <div class="fraud-card" style="text-align:center">
            <div style="font-size:32px;margin-bottom:8px">⚡</div>
            <div style="font-weight:600;color:#e4e6f0;margin-bottom:6px">FastAPI Backend</div>
            <div style="font-size:12px;color:#6b7db3">REST API that serves predictions in under 5ms with full validation</div>
            </div>
            """, unsafe_allow_html=True)
        with arch3:
            st.markdown("""
            <div class="fraud-card" style="text-align:center">
            <div style="font-size:32px;margin-bottom:8px">📊</div>
            <div style="font-weight:600;color:#e4e6f0;margin-bottom:6px">This Dashboard</div>
            <div style="font-size:12px;color:#6b7db3">Streamlit interface for interactive scoring and model monitoring</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="fraud-card" style="margin-top:1rem">
        <h4 style="color:#a5b4fc;margin-top:0">Setup in 4 steps</h4>
        <ol style="color:#9198b5;font-size:13px;line-height:2">
        <li>Download <code style="color:#818cf8">creditcard.csv</code> from
            <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" style="color:#6366f1">Kaggle</a>
            and place it in the <code style="color:#818cf8">data/</code> folder</li>
        <li>Run <code style="color:#818cf8">python model/train.py</code> to train and save the model</li>
        <li>Run <code style="color:#818cf8">uvicorn api.main:app --reload</code> in one terminal</li>
        <li>Run <code style="color:#818cf8">streamlit run dashboard/app.py</code> in another terminal</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="fraud-card">
        <h3 style="color:#a5b4fc;margin-top:0">How to Score a Transaction</h3>

        <p style="color:#9198b5;font-size:13px;margin-bottom:1rem">
        Navigate to <strong style="color:#e4e6f0">🔍 Score Transaction</strong> in the sidebar. You will see three sections:
        </p>

        <h4 style="color:#e4e6f0">1. Quick Fill buttons</h4>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        • <strong style="color:#10b981">Generate Legit</strong> — fills all 30 fields with values typical of a normal transaction<br>
        • <strong style="color:#ef4444">Generate Fraud</strong> — fills fields with values that pattern-match known fraud (extreme negative V12, V14, V17)<br>
        • <strong style="color:#6b7db3">Reset to Zeros</strong> — clears everything
        </p>

        <h4 style="color:#e4e6f0">2. Transaction Details</h4>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        • <strong style="color:#a5b4fc">Amount</strong> — The transaction dollar value (must be > 0)<br>
        • <strong style="color:#a5b4fc">Time</strong> — Seconds elapsed since the first transaction in the dataset. For testing, any value between 0 and 172,800 is fine.
        </p>

        <h4 style="color:#e4e6f0">3. User-Friendly Details</h4>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        Since technical PCA features (V1–V28) are often not available to end-users, FraudShield allows you to enter descriptive information about the transaction:
        <br>• <strong style="color:#a5b4fc">Merchant Category</strong> — The type of business (e.g., Luxury Goods carry higher risk)
        <br>• <strong style="color:#a5b4fc">Location</strong> — Where the transaction occurred relative to the user's home city
        <br>• <strong style="color:#a5b4fc">Transaction Type</strong> — How the payment was processed (e.g., Manual Entry is riskier)
        <br>• <strong style="color:#a5b4fc">New Device</strong> — Whether this card is being used on an unrecognized device
        </p>

        <h4 style="color:#e4e6f0">4. Reading the result</h4>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        After clicking <strong>Run Analysis</strong>, you'll see a risk card and gauge chart:<br>
        • <strong style="color:#10b981">GREEN / LOW</strong> — Approve the transaction<br>
        • <strong style="color:#f59e0b">AMBER / MEDIUM</strong> — Send for manual review<br>
        • <strong style="color:#ef4444">RED / HIGH</strong> — Block the transaction immediately
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="fraud-card">
        <h3 style="color:#a5b4fc;margin-top:0">Batch Scoring via CSV</h3>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        Use the <strong style="color:#e4e6f0">📦 Batch Analysis</strong> page to score many transactions at once.
        </p>

        <h4 style="color:#e4e6f0">CSV format requirements</h4>
        <p style="color:#9198b5;font-size:13px">Your CSV must have these exact column headers:</p>
        <code style="color:#818cf8;font-size:12px;display:block;background:#0d0f1a;padding:8px;border-radius:6px;margin:8px 0">
        V1,V2,V3,...,V28,Amount,Time
        </code>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        • Maximum 100 rows per upload<br>
        • All V1–V28 values should be floating point numbers<br>
        • Amount must be greater than 0<br>
        • Time must be 0 or greater<br>
        • The output CSV adds columns: <code style="color:#818cf8">fraud_probability</code>, <code style="color:#818cf8">risk_level</code>, <code style="color:#818cf8">is_fraud</code>, <code style="color:#818cf8">recommendation</code>
        </p>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="fraud-card">
        <h3 style="color:#a5b4fc;margin-top:0">Understanding the Output</h3>

        <h4 style="color:#e4e6f0">Fraud Probability</h4>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        A number between 0 and 1 representing the model's confidence that the transaction is fraudulent.
        0.02 means 2% probability of fraud. 0.95 means 95% probability of fraud.
        </p>

        <h4 style="color:#e4e6f0">Risk Levels</h4>
        </div>
        """, unsafe_allow_html=True)

        rl1, rl2, rl3 = st.columns(3)
        with rl1:
            st.markdown("""
            <div class="result-safe" style="text-align:center">
            <span class="tag tag-low">LOW</span>
            <div style="color:#9198b5;font-size:12px;margin-top:8px">Probability &lt; 20%<br>Safe to approve</div>
            </div>
            """, unsafe_allow_html=True)
        with rl2:
            st.markdown("""
            <div class="result-medium" style="text-align:center">
            <span class="tag tag-medium">MEDIUM</span>
            <div style="color:#9198b5;font-size:12px;margin-top:8px">Probability 20–60%<br>Manual review required</div>
            </div>
            """, unsafe_allow_html=True)
        with rl3:
            st.markdown("""
            <div class="result-fraud" style="text-align:center">
            <span class="tag tag-high">HIGH</span>
            <div style="color:#9198b5;font-size:12px;margin-top:8px">Probability &gt; 60%<br>Block immediately</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="fraud-card" style="margin-top:1rem">
        <h4 style="color:#e4e6f0;margin-top:0">Feature Importance Chart</h4>
        <p style="color:#9198b5;font-size:13px;line-height:1.8">
        After scoring, a bar chart shows which V features had the highest absolute values in your transaction.
        <strong style="color:#ef4444">Red bars</strong> = negative values (associated with fraud risk),
        <strong style="color:#10b981">Green bars</strong> = positive values (associated with legitimacy).
        The most fraud-predictive features in this dataset are <strong style="color:#a5b4fc">V14, V12, and V17</strong> —
        very negative values in these features strongly indicate fraud.
        </p>
        </div>
        """, unsafe_allow_html=True)
