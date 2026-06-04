"""Commerce ML Lab — Home page."""
from __future__ import annotations

import streamlit as st

st.title("Commerce ML Lab")
st.markdown("**Richard McShinsky** · End-to-end ML for e-commerce operations")
st.divider()

st.markdown(
    "Each project connects a model to a specific **operational decision** — "
    "reorder now vs. hold, approve vs. flag, who to target with a discount. "
    "Predictions without policies don't move metrics."
)

st.divider()

# ── Project cards ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("#### 📦 Demand Forecasting")
        ma, mb = st.columns(2)
        ma.metric("WMAPE gain", "+19%", help="0.058 vs 0.072 seasonal naive · M5 Walmart CA_1 · 28-day horizon")
        mb.metric("SKUs", "3,049", help="One global model — no per-SKU fitting")
        st.markdown(
            "One LightGBM model trained globally feeds an **(s, S) reorder policy**. "
            "80% prediction intervals from quantile regression size safety stock directly — "
            "connecting the forecast to an inventory action."
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 🔁 Returns Intelligence")
        ma, mb = st.columns(2)
        ma.metric("Fraud Precision@50", "40–60%", help="Graph features detect address-sharing rings")
        mb.metric("Rule coverage", "~70%", help="Returns resolved by heuristic layer, no ML needed")
        st.markdown(
            "Three models, one schema: fraud detection, return likelihood, and exchange recommendations. "
            "**Graph features** — shared addresses, payment hash rings — expose organised fraud "
            "that behavioural signals miss entirely."
        )

with col3:
    with st.container(border=True):
        st.markdown("#### 🎯 Checkout Uplift")
        ma, mb = st.columns(2)
        ma.metric("Targeting", "Uplift", help="T-learner Qini > propensity Qini at every budget level")
        mb.metric("Method", "T/S-learner", help="CATE estimation — who you can influence, not who will convert")
        st.markdown(
            "Propensity models find who will convert. Uplift models find who you can **influence**. "
            "T-learner CATE scores target persuadables and skip sure-things "
            "who would convert regardless and just drain discount budget."
        )

st.divider()

# ── Principles ─────────────────────────────────────────────────────────────────
st.markdown("#### Four principles applied across every project")
p1, p2, p3, p4 = st.columns(4)
p1.info("**Baseline first**\n\nEvery model earns its complexity vs. the simplest possible alternative.")
p2.info("**Predictions → policies**\n\nForecasts feed reorder rules. Scores feed approval thresholds.")
p3.info("**Cost-aware thresholds**\n\nDecision cutoffs tie to stated business costs, not arbitrary F1 targets.")
p4.info("**Honest limitations**\n\nEvery project carries a limitations section and a future-work list.")
