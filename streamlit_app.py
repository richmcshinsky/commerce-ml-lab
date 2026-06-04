"""Commerce ML Lab — Portfolio Overview

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Commerce ML Lab",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.caption("Richard McShinsky · [GitHub](https://github.com/richmcshinsky/commerce-ml-lab)")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.title("Commerce ML Lab")
st.subheader("End-to-end ML systems for e-commerce operations")

st.markdown(
    "Three production-ready projects, each connecting a **prediction** to an **operational decision**. "
    "The goal isn't better models — it's better outcomes."
)

st.divider()

# ── Project cards ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### 📦 Demand Forecasting")
    st.metric(
        label="vs. seasonal naive baseline",
        value="+19% WMAPE improvement",
        help="WMAPE 0.058 vs 0.072 · M5 Walmart CA_1 · 3,049 SKUs · 28-day horizon",
    )
    st.markdown(
        "A single global LightGBM model across 3,049 SKUs feeding an "
        "**(s, S) inventory reorder policy**. Quantile regression provides 80% prediction "
        "intervals that connect directly to safety-stock sizing. One model beats 3,049 "
        "individual models at a fraction of the compute and eliminates cold-start on low-volume SKUs."
    )

with col2:
    st.markdown("### 🔁 Returns Intelligence")
    st.metric(
        label="Fraud Precision@50",
        value="~40–60%",
        help="Graph features detect address-sharing rings that behavioural features miss entirely",
    )
    st.markdown(
        "Three models, one schema: **return likelihood**, **fraud detection**, and "
        "**exchange recommendations**. Graph features — shared delivery addresses, "
        "payment hash overlap, network component size — expose organised fraud rings "
        "that look individually normal on behavioural signals alone."
    )

with col3:
    st.markdown("### 🎯 Checkout Uplift")
    st.metric(
        label="vs. propensity targeting",
        value="Uplift > Propensity",
        help="T-learner Qini coefficient exceeds propensity model Qini at every budget level",
    )
    st.markdown(
        "Propensity models tell you who will convert — uplift models tell you who you "
        "can actually **influence**. T/S-learner CATE estimation targets persuadables "
        "and skips sure-things, who would convert regardless and just drain discount budget."
    )

st.divider()

# ── Principles ────────────────────────────────────────────────────────────────
st.markdown("#### Design principles applied across every project")

p1, p2, p3, p4 = st.columns(4)
p1.info("**Baseline before model**\nEvery technique earns its complexity against the simplest possible alternative.")
p2.info("**Prediction → policy**\nForecasts feed reorder rules. Scores feed approval thresholds. Uplift feeds budget allocation.")
p3.info("**Cost-aware thresholds**\nDecision cutoffs connect to stated business costs — not arbitrary F1 or accuracy targets.")
p4.info("**Honest limitations**\nEvery project carries a limitations section and a 'what I'd do with more time' list.")
