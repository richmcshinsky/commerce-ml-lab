"""P4 — Shipping Price Optimisation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "projects/04_shipping_optimization/src"))

# ── Constants ──────────────────────────────────────────────────────────────────

SHIPPING_OPTIONS = [
    {"name": "Free Shipping", "price": 0.00,  "days": "5–7 days", "label": "free"},
    {"name": "Standard",      "price": 4.99,  "days": "3–5 days", "label": "standard"},
    {"name": "Expedited",     "price": 7.99,  "days": "2–3 days", "label": "expedited"},
    {"name": "Express",       "price": 12.99, "days": "1 day",    "label": "express"},
]

MARGIN_RATE = 0.35
SHIP_COST   = 4.50
FLAT_PRICE  = 4.99

SEGMENT_PARAMS = {
    "sure_thing":   {"base_logit": 2.3,  "sensitivity": 0.06,  "color": "#1565C0"},
    "persuadable":  {"base_logit": 0.35, "sensitivity": 0.28,  "color": "#2E7D32"},
    "lost_cause":   {"base_logit": -2.7, "sensitivity": 0.02,  "color": "#757575"},
    "sleeping_dog": {"base_logit": -0.35,"sensitivity": -0.07, "color": "#E65100"},
}

SEGMENT_LABELS = {
    "sure_thing":   "Sure-thing",
    "persuadable":  "Persuadable",
    "lost_cause":   "Lost cause",
    "sleeping_dog": "Sleeping dog",
}

SEGMENT_WHY = {
    "sure_thing":   "Converts regardless of price — charging more recovers shipping cost without losing the sale",
    "persuadable":  "Price-sensitive — free shipping converts an otherwise-lost order that still turns a profit",
    "lost_cause":   "Won't convert at any price — charging more at least recovers cost if they do",
    "sleeping_dog": "Counter-intuitive: higher price slightly increases conversion (perceived quality signal)",
}


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))


def p_convert(segment: str, price: float, cart_value: float, is_returning: bool) -> float:
    """Parametric conversion probability — works without a trained model."""
    p = SEGMENT_PARAMS[segment]
    adj = 0.003 * (cart_value - 85.0) + 0.25 * float(is_returning)
    return float(_sigmoid(p["base_logit"] + adj - p["sensitivity"] * price))


def expected_margin(segment: str, price: float, cart_value: float, is_returning: bool) -> float:
    return p_convert(segment, price, cart_value, is_returning) * (
        cart_value * MARGIN_RATE + price - SHIP_COST
    )


def recommend(segment: str, cart_value: float, is_returning: bool) -> dict:
    return max(
        SHIPPING_OPTIONS,
        key=lambda o: expected_margin(segment, o["price"], cart_value, is_returning),
    )


def fmt_price(price: float) -> str:
    """Format a price as a dollar string safe for Streamlit markdown (escaped dollar sign)."""
    return "Free" if price == 0.0 else f"\\${price:.2f}"


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("🚚 Shipping Price Optimisation")
st.markdown(
    "Which shipping option should you show — and at what price — to maximise margin "
    "without hurting conversion? The same session data that feeds the intent model "
    "drives a per-session pricing decision."
)

# Core insight — rewritten without paired dollar signs to avoid KaTeX rendering
st.info(
    "**Core insight:** A flat shipping rate is optimal for no one. "
    "Sure-things will convert even at an express premium — charging them less gives away margin. "
    "Persuadables only convert with low or no shipping cost — charge them the standard rate "
    "and the order is lost entirely. "
    "Segment-aware pricing recovers both.",
    icon="💡",
)

st.divider()

# ── Session configurator ───────────────────────────────────────────────────────
st.subheader("Session Configurator")
st.caption(
    "Adjust the session below. The optimiser evaluates all four shipping tiers "
    "and returns the one that maximises expected margin."
)

col_seg, col_cart, col_ret, col_depth = st.columns([2, 2, 1, 1])
with col_seg:
    segment = st.selectbox(
        "Customer segment",
        list(SEGMENT_PARAMS.keys()),
        format_func=lambda s: SEGMENT_LABELS[s],
        help="In production this is predicted from session behaviour — here you can select it directly.",
    )
with col_cart:
    cart_value = st.slider("Cart value", 15, 400, 85, step=5, format="$%d")
with col_ret:
    is_returning = st.toggle("Returning customer", value=True)
with col_depth:
    session_depth = st.number_input("Pages visited", min_value=1, max_value=30, value=8)

# Compute recommendation
prices  = [o["price"] for o in SHIPPING_OPTIONS]
p_vals  = [p_convert(segment, p, cart_value, is_returning) for p in prices]
em_vals = [expected_margin(segment, p, cart_value, is_returning) for p in prices]
best_opt = recommend(segment, cart_value, is_returning)

best_em  = expected_margin(segment, best_opt["price"], cart_value, is_returning)
best_p   = p_convert(segment, best_opt["price"], cart_value, is_returning)
flat_em  = expected_margin(segment, FLAT_PRICE, cart_value, is_returning)
flat_p   = p_convert(segment, FLAT_PRICE, cart_value, is_returning)
em_delta = best_em - flat_em
p_delta  = best_p - flat_p

# ── Recommendation card ────────────────────────────────────────────────────────
rec_col, metric_col = st.columns([3, 2])

with rec_col:
    delta_sign = "+" if em_delta >= 0 else ""
    price_display = fmt_price(best_opt["price"])
    # Use escaped dollar signs (\$) to prevent KaTeX from treating them as math delimiters
    st.success(
        f"### 🚚 Recommended: {best_opt['name']}\n\n"
        f"**Price:** {price_display} &nbsp;·&nbsp; **Delivery:** {best_opt['days']}\n\n"
        f"**P(convert):** {best_p:.0%} &nbsp;·&nbsp; "
        f"**Expected margin:** \\${best_em:.2f} "
        f"({delta_sign}\\${abs(em_delta):.2f} vs standard rate)"
    )

with metric_col:
    m1, m2 = st.columns(2)
    m1.metric("Expected margin", f"${best_em:.2f}", f"{em_delta:+.2f} vs flat rate")
    m2.metric("P(convert)", f"{best_p:.0%}", f"{p_delta:+.0%} vs flat rate")

# ── Price optimisation chart ───────────────────────────────────────────────────
fig_surf = go.Figure()

bar_colors = [
    "#1565C0" if o["price"] == best_opt["price"] else "#CFD8DC"
    for o in SHIPPING_OPTIONS
]
fig_surf.add_trace(go.Bar(
    x=[o["name"] for o in SHIPPING_OPTIONS],
    y=em_vals,
    text=[f"${v:.2f}" for v in em_vals],
    textposition="outside",
    marker_color=bar_colors,
    name="Expected margin ($)",
    hovertemplate="<b>%{x}</b><br>Expected margin: $%{y:.2f}<extra></extra>",
))
fig_surf.add_trace(go.Scatter(
    x=[o["name"] for o in SHIPPING_OPTIONS],
    y=[v * 100 for v in p_vals],
    mode="lines+markers",
    name="P(convert) %",
    yaxis="y2",
    line=dict(color="#E53935", width=2.5),
    marker=dict(size=9),
    hovertemplate="<b>%{x}</b><br>P(convert): %{y:.1f}%<extra></extra>",
))
fig_surf.update_layout(
    height=320,
    margin=dict(t=10, b=40, l=50, r=60),
    yaxis=dict(title="Expected margin ($)"),
    yaxis2=dict(title="P(convert) %", overlaying="y", side="right", showgrid=False, range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_surf, use_container_width=True)
st.caption(
    f"Blue bar = recommended option. "
    f"The optimal price is where the margin curve peaks for this segment. "
    f"For a **{SEGMENT_LABELS[segment].lower()}** with a "
    f"\\${cart_value} cart, that is **{best_opt['name']}**."
)

st.divider()

# ── Why prices differ by segment ───────────────────────────────────────────────
st.subheader("Why the Optimal Price Differs by Segment")
st.markdown(
    "Each line below shows expected margin at every price point for one segment type. "
    "The star marks the maximum. The dashed line is the flat \\${:.2f} standard rate — "
    "notice it hits the peak for no segment.".format(FLAT_PRICE)
)

prices_fine = np.linspace(0, 13, 100)
fig_seg = go.Figure()

for seg, params in SEGMENT_PARAMS.items():
    em_curve = [expected_margin(seg, float(p), cart_value, True) for p in prices_fine]
    fig_seg.add_trace(go.Scatter(
        x=prices_fine, y=em_curve,
        name=SEGMENT_LABELS[seg],
        line=dict(color=params["color"], width=2.2),
        hovertemplate=(
            f"<b>{SEGMENT_LABELS[seg]}</b><br>"
            "Price: $%{x:.2f}<br>E[margin]: $%{y:.2f}<extra></extra>"
        ),
    ))
    best_p_seg = float(max(prices_fine, key=lambda p: expected_margin(seg, float(p), cart_value, True)))
    best_em_seg = expected_margin(seg, best_p_seg, cart_value, True)
    fig_seg.add_trace(go.Scatter(
        x=[best_p_seg], y=[best_em_seg],
        mode="markers",
        marker=dict(color=params["color"], size=12, symbol="star"),
        showlegend=False,
        hovertemplate=(
            f"<b>{SEGMENT_LABELS[seg]} optimum</b><br>"
            "Price: $%{x:.2f}<br>E[margin]: $%{y:.2f}<extra></extra>"
        ),
    ))

fig_seg.add_vline(
    x=FLAT_PRICE, line_dash="dash", line_color="#757575", opacity=0.7,
    annotation_text=f"Flat rate ${FLAT_PRICE}", annotation_position="top right",
)
fig_seg.update_layout(
    height=360,
    xaxis_title="Shipping price charged to customer ($)",
    yaxis_title="Expected margin per session ($)",
    margin=dict(t=10, b=40, l=50, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_seg, use_container_width=True)

st.divider()

# ── Segment strategy table ─────────────────────────────────────────────────────
st.subheader("Pricing Strategy by Segment")

rows = []
for seg in SEGMENT_PARAMS:
    best_seg = recommend(seg, cart_value, True)
    rows.append({
        "Segment": SEGMENT_LABELS[seg],
        "Recommended option": f"{best_seg['name']} (${best_seg['price']:.2f})",
        "P(convert) at flat rate": f"{p_convert(seg, FLAT_PRICE, cart_value, True):.0%}",
        "P(convert) optimised": f"{p_convert(seg, best_seg['price'], cart_value, True):.0%}",
        "Margin gain vs flat rate": f"${expected_margin(seg, best_seg['price'], cart_value, True) - expected_margin(seg, FLAT_PRICE, cart_value, True):+.2f}",
        "Business logic": SEGMENT_WHY[seg],
    })

st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "Adjust cart value and segment above — the recommendations update in real time."
)

st.divider()

# ── Results (if training has been run) ─────────────────────────────────────────
RESULTS = ROOT / "projects/04_shipping_optimization/results"
metrics_path  = RESULTS / "shipping_metrics.csv"
policy_png    = RESULTS / "shipping_policy_comparison.png"
elasticity_png = RESULTS / "shipping_elasticity_curves.png"
price_dist_png = RESULTS / "shipping_price_distribution.png"

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path).set_index("metric")["value"]
    st.subheader("Model Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Elasticity model PR-AUC",
        f"{float(metrics.get('elasticity_pr_auc', 0)):.3f}",
        help=f"Random baseline: {float(metrics.get('random_baseline_pr_auc', 0)):.3f}",
    )
    m2.metric(
        "Margin improvement",
        f"+{float(metrics.get('margin_improvement_pct', 0)):.1f}%",
        help="Optimised policy vs flat-rate standard shipping",
    )
    m3.metric(
        "Conversion rate shift",
        f"{(float(metrics.get('optimised_mean_conversion', 0)) - float(metrics.get('flat_rate_mean_conversion', 0)))*100:+.1f} pp",
        help="Percentage-point change — near zero by design",
    )
    m4.metric(
        "Training sessions",
        f"{int(float(metrics.get('n_train', 0))):,}",
        help="Randomly price-assigned checkout sessions (A/B test simulation)",
    )

with st.expander("📊 Training charts & methodology", expanded=False):
    if elasticity_png.exists() or policy_png.exists():
        ch1, ch2 = st.columns(2)
        if elasticity_png.exists():
            ch1.image(str(elasticity_png),
                      caption="Price elasticity and expected margin curves by segment",
                      use_container_width=True)
        if policy_png.exists():
            ch2.image(str(policy_png),
                      caption="Policy comparison: flat rate vs optimised vs always-free vs always-express",
                      use_container_width=True)
        if price_dist_png.exists():
            st.image(str(price_dist_png),
                     caption="Recommended tier by segment under the optimised policy",
                     use_container_width=True)
    else:
        st.info("Run `make train-shipping` to generate training charts.", icon="ℹ️")

    st.markdown(
        """
        **Why train on A/B test data?**

        Each session in the training set was randomly assigned one shipping price from the menu.
        Because price assignment is random, the model's price coefficient estimates a genuine
        causal effect — not a spurious correlation between price and customer quality.
        This is the same identification strategy used in randomised controlled trials.

        **Optimisation objective**

        For each session with features X, evaluate every shipping option at price p:

        `E[margin | X, p] = P(convert | X, p) × (cart_value × 0.35 + p − 4.50)`

        Select the option that maximises this quantity. No separate conversion-rate constraint
        is needed — the objective already penalises prices that lose conversions, because a
        session that doesn't convert contributes zero margin at any price.

        **Connection to Project 3 — Checkout Uplift**

        The segment decomposition from the CATE model maps directly onto shipping pricing:
        - **Persuadables** (CATE = +12.3%): free shipping is the intervention that converts them
        - **Sure-things** (CATE = +1.0%): already converting — recover the shipping cost
        - **Sleeping dogs**: intervention (discount) slightly hurts conversion — charge full rate

        Uplift-aware shipping pricing is the same architecture as a CATE-informed discount
        policy, applied to a different lever.
        """
    )
