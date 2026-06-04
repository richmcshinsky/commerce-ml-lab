"""P1 — Demand Forecasting & Inventory Optimization."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils import ROOT, load_forecast_metrics, load_inventory_policies, sidebar_brand

st.set_page_config(
    page_title="Demand Forecasting · Commerce ML Lab",
    page_icon="📦",
    layout="wide",
)
sidebar_brand()

st.title("📦 Demand Forecasting & Inventory Optimization")
st.markdown(
    "A single global LightGBM model across **3,049 SKUs** translates sales history into "
    "inventory reorder decisions — no per-SKU models, no cold-start problem."
)

policies = load_inventory_policies()
metrics = load_forecast_metrics()

if policies.empty:
    st.warning("Inventory policies not found. Run `make train-forecast` to generate them.")
    st.stop()

# ── Live reorder calculator ────────────────────────────────────────────────────
st.subheader("Live Reorder Calculator")
st.caption("Select a SKU and enter current stock level to get an instant reorder decision.")

policies["_cat"] = policies["id"].str.extract(r"^([A-Z]+)")
categories = sorted(policies["_cat"].dropna().unique())

col_cat, col_sku, col_inv = st.columns([1, 2, 1])
with col_cat:
    sel_cat = st.selectbox("Category", ["All"] + categories)
with col_sku:
    filtered = policies if sel_cat == "All" else policies[policies["_cat"] == sel_cat]
    selected_id = st.selectbox("SKU", filtered["id"].tolist())
with col_inv:
    sku_row = policies[policies["id"] == selected_id].iloc[0]
    s = float(sku_row["reorder_point"])
    S = float(sku_row["order_up_to"])
    current_inv = st.number_input(
        "Units on hand",
        min_value=0,
        max_value=int(S * 3),
        value=int(s * 0.7),
        step=1,
    )

should_reorder = current_inv <= s
qty = max(0, math.ceil(S - current_inv)) if should_reorder else 0

if should_reorder:
    st.error(
        f"### ⚠️  Reorder — place an order for **{qty:,} units**\n\n"
        f"Stock ({current_inv:,}) is at or below the reorder point ({s:,.0f}). "
        f"Order {qty:,} units to restore to the order-up-to level ({S:,.0f})."
    )
else:
    headroom = current_inv - s
    st.success(
        f"### ✅  Hold — no action needed\n\n"
        f"Stock ({current_inv:,}) is {headroom:,.0f} units above the reorder trigger ({s:,.0f}). "
        f"No order required."
    )

# Policy parameters
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Mean daily demand", f"{sku_row['mean_daily_demand']:.2f} units")
c2.metric("Safety stock", f"{sku_row['safety_stock']:.0f} units")
c3.metric("Reorder point (s)", f"{s:,.0f}")
c4.metric("Order-up-to (S)", f"{S:,.0f}")
c5.metric("Lead time", f"{int(sku_row['lead_time_days'])} days")
c6.metric("Service level", f"{sku_row['service_level']:.0%}")

# ── Inventory gauge ────────────────────────────────────────────────────────────
max_val = max(S * 1.4, current_inv * 1.1, 1.0)
to_pct = lambda v: min(100.0 * v / max_val, 100.0)  # noqa: E731
s_pct, S_pct, curr_pct = to_pct(s), to_pct(S), to_pct(current_inv)
marker_color = "#C62828" if should_reorder else "#1565C0"

fig_g = go.Figure()
for x0, x1, fill in [
    (0, s_pct, "rgba(239,83,80,0.12)"),
    (s_pct, S_pct, "rgba(67,160,71,0.12)"),
    (S_pct, 100, "rgba(33,150,243,0.07)"),
]:
    fig_g.add_shape(type="rect", x0=x0, x1=x1, y0=0.2, y1=0.8,
                    fillcolor=fill, line_width=0)

fig_g.add_shape(type="line", x0=s_pct, x1=s_pct, y0=0.1, y1=0.9,
                line=dict(color="#E65100", width=2, dash="dash"))
fig_g.add_shape(type="line", x0=S_pct, x1=S_pct, y0=0.1, y1=0.9,
                line=dict(color="#2E7D32", width=2, dash="dot"))
fig_g.add_shape(type="line", x0=curr_pct, x1=curr_pct, y0=0.05, y1=0.95,
                line=dict(color=marker_color, width=5))

fig_g.add_annotation(x=s_pct, y=0.95, text=f"s={s:,.0f}", showarrow=False,
                     font=dict(color="#E65100", size=11, family="monospace"), xanchor="center")
fig_g.add_annotation(x=S_pct, y=0.95, text=f"S={S:,.0f}", showarrow=False,
                     font=dict(color="#2E7D32", size=11, family="monospace"), xanchor="center")
fig_g.add_annotation(x=curr_pct, y=0.05, text=f"Now: {current_inv:,}", showarrow=False,
                     font=dict(color=marker_color, size=12, family="monospace"), xanchor="center")
if s_pct > 10:
    fig_g.add_annotation(x=s_pct / 2, y=0.5, text="⚠ Reorder", showarrow=False,
                         font=dict(size=10, color="#B71C1C"), xanchor="center")
if S_pct - s_pct > 10:
    fig_g.add_annotation(x=(s_pct + S_pct) / 2, y=0.5, text="Safe zone", showarrow=False,
                         font=dict(size=10, color="#1B5E20"), xanchor="center")

fig_g.update_layout(
    height=110,
    margin=dict(t=5, b=5, l=10, r=10),
    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.plotly_chart(fig_g, use_container_width=True)

st.divider()

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("📊 Model performance & methodology", expanded=False):
    st.subheader("28-day test horizon — all models vs baselines")
    if not metrics.empty:
        display = metrics.copy()
        for col in ("wmape", "mase", "rmsse"):
            if col in display.columns:
                display[col] = display[col].map("{:.3f}".format)
        st.dataframe(
            display.rename(columns={"model": "Model", "wmape": "WMAPE ↓",
                                    "mase": "MASE ↓", "rmsse": "RMSSE ↓"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "MASE < 1.0 means the model beats seasonal naive on its own scale. "
            "WMAPE 0.058 vs 0.072 = 19% improvement. "
            "One global model across all 3,049 SKUs outperforms per-SKU approaches at a fraction of the compute."
        )

    chart_specs = [
        ("Model comparison vs baselines",
         ROOT / "projects/01_demand_forecasting/results/lgbm_model_comparison.png"),
        ("Forecast with 80% prediction interval",
         ROOT / "projects/01_demand_forecasting/results/lgbm_forecast_with_intervals.png"),
        ("SHAP feature importance",
         ROOT / "projects/01_demand_forecasting/results/lgbm_shap_summary.png"),
    ]
    chart_cols = st.columns(3)
    for (label, path), col in zip(chart_specs, chart_cols):
        with col:
            if path.exists():
                st.image(str(path), caption=label, use_container_width=True)
            else:
                st.caption(f"_{label} — chart not found_")

    st.markdown(
        "**Key design decisions:**\n"
        "- Lag features at multiples of 7 preserve weekday alignment across the full SKU range.\n"
        "- 80% PI from quantile regression feeds the safety-stock formula directly: σ̂ = half-width ÷ 1.282.\n"
        "- Global model eliminates cold-start on low-volume SKUs and reduces training overhead 3,049×.\n"
        "- (s, S) policy chosen over newsvendor-only because it handles variable lead times gracefully."
    )
