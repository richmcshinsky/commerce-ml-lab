"""Commerce ML Lab — Home page."""
from __future__ import annotations

import streamlit as st

st.title("Commerce ML Lab")
st.markdown(
    "**Richard McShinsky** · [GitHub](https://github.com/richmcshinsky/commerce-ml-lab)"
)

st.markdown(
    "Four end-to-end ML systems built on one principle: **every model connects to a specific "
    "operational decision.** Predictions that don't change behaviour don't create value."
)

st.divider()

# ── 2×2 project cards ──────────────────────────────────────────────────────────
row1_l, row1_r = st.columns(2, gap="large")
row2_l, row2_r = st.columns(2, gap="large")

with row1_l:
    with st.container(border=True):
        st.markdown("### 📦 Demand Forecasting")
        st.markdown("**+19% accuracy** over seasonal naive · 3,049 SKUs · 28-day horizon")
        st.divider()
        st.markdown(
            "A single LightGBM model trained across all SKUs feeds a live **(s, S) reorder policy**. "
            "The model's 80% prediction intervals size safety stock directly — "
            "turning the forecast into a concrete 'reorder now or hold' decision for every SKU."
        )

with row1_r:
    with st.container(border=True):
        st.markdown("### 🔁 Returns Intelligence")
        st.markdown("**Fraud PR-AUC 0.560** (23× random baseline) · ~70% of returns resolved by rules alone")
        st.divider()
        st.markdown(
            "Three models share one data schema: fraud detection, return likelihood, and exchange recommendations. "
            "Graph features — shared delivery addresses, payment hash rings — expose organised fraud rings "
            "that look individually normal on every other signal."
        )

with row2_l:
    with st.container(border=True):
        st.markdown("### 🎯 Checkout Uplift")
        st.markdown("**Persuadables: +12.3% treatment effect** · Sure-things: +1.0% despite 15.9% propensity")
        st.divider()
        st.markdown(
            "Propensity scores find who will convert. Uplift models find who you can actually **influence**. "
            "Sure-things have the highest propensity but near-zero treatment effect — "
            "targeting them wastes every discount dollar on sales that would have happened anyway."
        )

with row2_r:
    with st.container(border=True):
        st.markdown("### 🚚 Shipping Optimisation")
        st.markdown("**+16.4% expected margin** vs flat-rate · conversion rate also improves (+1.3 pp)")
        st.divider()
        st.markdown(
            "Per-session shipping price selection using a causal elasticity model. "
            "Sure-things absorb a premium express rate. Persuadables convert with free shipping — "
            "and the order still profits after the shipping cost. "
            "The result: more margin **and** higher conversion. One flat rate achieves neither."
        )

st.divider()

# ── Principles ─────────────────────────────────────────────────────────────────
st.markdown("#### Principles applied across every project")
p1, p2, p3, p4 = st.columns(4)
p1.info("**Baseline first**\n\nEvery technique proves itself against the simplest possible alternative.")
p2.info("**Predictions → policies**\n\nForecasts feed reorder rules. Scores feed fraud thresholds. CATE estimates feed pricing decisions.")
p3.info("**Cost-aware**\n\nDecision cutoffs tie to stated business costs, not F1 targets chosen without context.")
p4.info("**Honest limits**\n\nEvery project has a limitations section. Synthetic data is always labelled as such.")
