"""P2 — Returns Intelligence."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils import EXCHANGE_RULES, ROOT, load_returns_data, score_return

st.title("🔁 Returns Intelligence")
st.markdown(
    "Three questions answered from one dataset: **Will this order be returned?** "
    "**Is this return fraudulent?** **What should we offer in exchange?**"
)

customers, orders, returns = load_returns_data()

if customers.empty or orders.empty or returns.empty:
    st.warning("Returns data not found. Run `make train-returns` to generate it.")
    st.stop()

# ── Interactive return evaluator ───────────────────────────────────────────────
st.subheader("Return Evaluator")
st.caption(
    "Enter return details to get an instant fraud risk assessment and exchange recommendation. "
    "Uses the same two-stage heuristic + feature-scoring logic as the production model."
)

cats = sorted(orders["category"].unique()) if "category" in orders.columns else ["electronics", "clothing", "shoes"]
reasons = sorted(returns["reason_code"].unique()) if "reason_code" in returns.columns else list(EXCHANGE_RULES.keys())
conditions_list = ["new", "used", "damaged"]

c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
with c1:
    sel_cat = st.selectbox("Category", cats, key="ri_cat")
with c2:
    sel_reason = st.selectbox("Return reason", reasons, key="ri_reason")
with c3:
    sel_cond = st.selectbox("Returned condition", conditions_list, key="ri_cond")
with c4:
    sel_days = st.number_input("Days since purchase", min_value=1, max_value=90, value=14, step=1)
with c5:
    sel_price = st.number_input("Item price ($)", min_value=5, max_value=2000, value=89, step=5)

risk_score, risk_label, exchange_rec = score_return(
    sel_cat, sel_cond, sel_reason, int(sel_days), float(sel_price)
)

res_left, res_right = st.columns(2)
with res_left:
    emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk_label]
    action = {
        "Low": "Approve — process refund or exchange automatically.",
        "Medium": "Flag for manual review before issuing refund.",
        "High": "Hold — investigate before processing. Do not auto-refund.",
    }[risk_label]
    caller = {"Low": st.success, "Medium": st.warning, "High": st.error}[risk_label]
    caller(f"### {emoji} Fraud Risk: **{risk_label}** ({risk_score:.0%})\n\n{action}")

with res_right:
    st.info(
        f"### 📦 Exchange Recommendation\n\n"
        f"**Reason:** {sel_reason.replace('_', ' ').title()}\n\n"
        f"{exchange_rec}"
    )

with st.expander("How the scoring works"):
    st.markdown(
        """
        The two-stage decision pipeline handles returns in order of certainty:

        | Stage | Covers | Speed | Transparency |
        |-------|--------|-------|-------------|
        | **Heuristic rules** | Clear-cut cases — size swaps, color exchanges, warranty claims | Instant | Fully explainable |
        | **Feature scoring** | Ambiguous cases — `changed_mind`, `not_as_described`, late returns in poor condition | <5 ms | SHAP reason codes |

        ~70% of returns are resolved by heuristics. Key fraud signals in the feature model:
        - **Condition mismatch** — item returned as "defective" but in damaged state
        - **Timing patterns** — velocity fraud returns within 3 days; wardrobers return after 20–30 days
        - **High-value categories** — electronics above $150 with condition issues
        - **Graph features** — shared addresses or payment hashes across accounts (ring detection)
        """
    )

st.divider()

# ── Dataset snapshot ───────────────────────────────────────────────────────────
st.subheader("Dataset Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(customers):,}")
c2.metric("Orders", f"{len(orders):,}")
c3.metric("Returns", f"{len(returns):,}")
if "is_fraud" in returns.columns:
    c4.metric("Fraud rate in returns", f"{returns['is_fraud'].mean():.1%}")

st.divider()

# ── Signal charts ──────────────────────────────────────────────────────────────
st.subheader("Fraud Signals")

row1_l, row1_r = st.columns(2)

with row1_l:
    if "category" in orders.columns and "was_returned" in orders.columns:
        cat_rr = orders.groupby("category")["was_returned"].mean().sort_values(ascending=False)
        fig1 = go.Figure(go.Bar(
            x=cat_rr.index, y=cat_rr.values,
            text=[f"{v:.0%}" for v in cat_rr.values], textposition="outside",
            marker_color="#1976D2",
        ))
        fig1.update_layout(
            title="Return rate by category", yaxis_tickformat=".0%",
            yaxis_title="Return rate", height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig1, use_container_width=True)

with row1_r:
    if "archetype" in customers.columns and "was_returned" in orders.columns:
        arch_df = orders.merge(customers[["customer_id", "archetype"]], on="customer_id", how="left")
        arch_rr = arch_df.groupby("archetype")["was_returned"].mean().sort_values(ascending=False)
        arch_colors = {"wardrober": "#EF5350", "velocity": "#FF7043",
                       "ring": "#AB47BC", "normal": "#66BB6A"}
        fig2 = go.Figure(go.Bar(
            x=arch_rr.index, y=arch_rr.values,
            text=[f"{v:.0%}" for v in arch_rr.values], textposition="outside",
            marker_color=[arch_colors.get(k, "#888") for k in arch_rr.index],
        ))
        fig2.update_layout(
            title="Return rate by fraud archetype", yaxis_tickformat=".0%",
            yaxis_title="Return rate", height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)

row2_l, row2_r = st.columns(2)
returns_rich = returns.merge(orders[["order_id", "item_price", "category"]], on="order_id", how="left")

with row2_l:
    if "days_to_return" in returns_rich.columns and "is_fraud" in returns_rich.columns:
        fraud_days = returns_rich[returns_rich["is_fraud"]]["days_to_return"].clip(upper=60)
        legit_days = returns_rich[~returns_rich["is_fraud"]]["days_to_return"].clip(upper=60)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=legit_days, name="Legitimate", opacity=0.65,
                                    marker_color="#43A047", nbinsx=30))
        fig3.add_trace(go.Histogram(x=fraud_days, name="Fraud", opacity=0.65,
                                    marker_color="#E53935", nbinsx=30))
        fig3.update_layout(
            barmode="overlay", title="Days to return: fraud vs legitimate",
            xaxis_title="Days to return", height=280, margin=dict(t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig3, use_container_width=True)

with row2_r:
    if "condition" in returns_rich.columns and "is_fraud" in returns_rich.columns:
        cond_fraud = returns_rich.groupby("condition")["is_fraud"].mean().sort_values(ascending=False)
        cond_colors = {"damaged": "#E53935", "used": "#FF7043", "new": "#43A047"}
        fig4 = go.Figure(go.Bar(
            x=cond_fraud.index, y=cond_fraud.values,
            text=[f"{v:.1%}" for v in cond_fraud.values], textposition="outside",
            marker_color=[cond_colors.get(k, "#888") for k in cond_fraud.index],
        ))
        fig4.update_layout(
            title="Fraud rate by return condition", yaxis_tickformat=".0%",
            yaxis_title="Fraud rate", height=280, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Ring fraud ─────────────────────────────────────────────────────────────────
st.subheader("Why Graph Features Matter")

if "archetype" in customers.columns and "customer_id" in returns.columns:
    ring_mask = customers["archetype"] == "ring"
    normal_mask = customers["archetype"] == "normal"
    ring_ids = customers.loc[ring_mask, "customer_id"]
    normal_ids = customers.loc[normal_mask, "customer_id"]

    ring_fraud = returns[returns["customer_id"].isin(ring_ids)]["is_fraud"].mean()
    normal_fraud = returns[returns["customer_id"].isin(normal_ids)]["is_fraud"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Ring member fraud rate", f"{ring_fraud:.0%}",
              delta=f"+{ring_fraud - normal_fraud:.0%} vs normal")
    m2.metric("Normal customer fraud rate", f"{normal_fraud:.1%}")
    if "address_id" in customers.columns:
        max_per_addr = customers.loc[ring_mask, "address_id"].value_counts().max()
        m3.metric("Max ring members per address", str(max_per_addr))

st.info(
    "Ring members look individually normal — their return rates and timings are plausible in isolation. "
    "The fraud signal lives on the **graph**: multiple accounts sharing a delivery address or payment hash. "
    "`component_size` and `shared_address_count` contribute more lift to the fraud model than all "
    "behavioural features combined."
)

with st.expander("📊 Exchange recommendation model"):
    if "reason_code" in returns.columns:
        reason_counts = returns["reason_code"].value_counts()
        fig5 = go.Figure(go.Bar(
            x=reason_counts.index, y=reason_counts.values,
            marker_color="#1976D2",
            text=reason_counts.values, textposition="outside",
        ))
        fig5.update_layout(
            title="Return volume by reason code",
            height=260, margin=dict(t=50, b=30), yaxis_title="Return count",
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        "| Reason | Heuristic rule | Coverage |\n"
        "|--------|---------------|----------|\n"
        "| `too_small` / `too_large` | Next size in same style | ~30% of returns |\n"
        "| `wrong_color` | Any color variant | ~15% |\n"
        "| `defective` | Exact replacement, expedited | ~20% |\n"
        "| `changed_mind` / `not_as_described` | Feature-scored ranker | ~35% (the long tail) |\n\n"
        "~65–70% of returns hit a heuristic rule directly, returning an instant, "
        "fully explainable recommendation. The ML ranker handles the ambiguous tail."
    )
