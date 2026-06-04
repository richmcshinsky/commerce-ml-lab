"""Commerce ML Lab — Interactive Demo
=====================================
Three ML systems, live in your browser. No training required.

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
    Point the app at this file from https://share.streamlit.io
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Commerce ML Lab",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛒 Commerce ML Lab")
    st.caption("Richard McShinsky · [GitHub](https://github.com/richmcshinsky/commerce-ml-lab)")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📦 P1 — Demand Forecasting",
            "🔁 P2 — Returns Intelligence",
            "🎯 P3 — Checkout Uplift",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Built to mirror Redo's ML problem set: inventory forecasting, "
        "returns intelligence, and intervention targeting."
    )


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_inventory_policies() -> pd.DataFrame:
    p = ROOT / "projects/01_demand_forecasting/results/inventory_policies.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_forecast_metrics() -> pd.DataFrame:
    rows = []
    for f in [
        ROOT / "projects/01_demand_forecasting/results/baselines_metrics.csv",
        ROOT / "projects/01_demand_forecasting/results/lgbm_metrics.csv",
    ]:
        if f.exists():
            rows.append(pd.read_csv(f))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data
def load_returns_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ROOT / "projects/02_returns_intelligence/results"
    def _load(name: str) -> pd.DataFrame:
        p = base / name
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _load("customers.parquet"), _load("orders.parquet"), _load("returns.parquet")


@st.cache_data
def load_uplift_data(max_rows: int = 80_000) -> pd.DataFrame:
    p = ROOT / "projects/03_checkout_intent/results/uplift_data.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _qini_curve(
    y: np.ndarray,
    w: np.ndarray,
    score: np.ndarray,
    n_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraction_targeted, incremental_conversions) for a Qini curve."""
    order = np.argsort(score)[::-1]
    y_s, w_s = y[order], w[order]
    n = len(y_s)
    n_treated = w_s.sum()
    n_control = n - n_treated

    bins = np.unique(np.linspace(0, n, n_bins + 1, dtype=int))
    fracs, incr = [0.0], [0.0]
    for k in bins[1:]:
        top_y, top_w = y_s[:k], w_s[:k]
        nt = top_w.sum()
        nc = k - nt
        treated_conv = (top_y * top_w).sum()
        control_conv = (top_y * (1 - top_w)).sum()
        inc = treated_conv - control_conv * (nt / max(nc, 1))
        fracs.append(k / n)
        incr.append(float(inc))
    return np.array(fracs), np.array(incr)


def _qini_coeff(fracs: np.ndarray, incr: np.ndarray) -> float:
    random_line = incr[-1] * fracs
    return float(np.trapz(incr - random_line, fracs))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("Commerce ML Lab")
    st.subheader("An end-to-end ML portfolio focused on e-commerce operations")

    st.markdown(
        """
        This portfolio builds the four ML problem areas that map directly onto
        how a returns/inventory platform like Redo thinks about its data.

        | Project | Question | Key technique |
        |---------|----------|---------------|
        | **P1 · Demand Forecasting** | Given sales history and lead times, when and how much should you reorder? | Global LightGBM + (s, S) inventory policy |
        | **P2 · Returns Intelligence** | Which returns are fraudulent? What exchange should we offer? | Graph features + cost-aware threshold |
        | **P3 · Checkout Uplift** | Who actually *changes behaviour* because of an intervention? | T/S-learner CATE · Qini evaluation |

        Use the sidebar to explore each project interactively.
        """
    )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecast improvement over seasonal naive", "19%", delta_color="normal")
        st.caption("WMAPE 0.058 vs 0.072 — M5 CA_1, 3,049 SKUs, 28-day horizon")
    with col2:
        st.metric("Fraud Precision@50", "~40–60%", delta_color="normal")
        st.caption("Graph features detect address-sharing rings that behavioural features miss")
    with col3:
        st.metric("Uplift > Propensity targeting", "✓", delta_color="normal")
        st.caption("T-learner correctly deprioritises sure-things; propensity model wastes budget on them")

    st.divider()
    st.markdown(
        "**Philosophy:** baselines before models · connect forecasts to decisions · "
        "honest limitations · cost-aware thresholds · right metric for the problem."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: P1 — DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 P1 — Demand Forecasting":
    st.title("📦 Demand Forecasting & Inventory Optimisation")
    st.caption(
        "Global LightGBM trained on M5 Walmart CA_1 (3,049 SKUs). "
        "Forecasts feed a newsvendor → (s, S) inventory policy layer."
    )

    policies = load_inventory_policies()
    metrics = load_forecast_metrics()

    if policies.empty:
        st.warning("Inventory policies CSV not found. Run `make train-forecast` to generate it.")
        st.stop()

    # ── Model comparison ──────────────────────────────────────────────────────
    st.subheader("Model comparison — 28-day test horizon")
    if not metrics.empty:
        display = metrics.copy()
        display["wmape"] = display["wmape"].map("{:.3f}".format)
        display["mase"] = display["mase"].map("{:.3f}".format)
        display["rmsse"] = display["rmsse"].map("{:.3f}".format)
        st.dataframe(
            display.rename(columns={"model": "Model", "wmape": "WMAPE ↓", "mase": "MASE ↓", "rmsse": "RMSSE ↓"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "MASE < 1.0 means the model beats the seasonal naive baseline on its own scale. "
            "WMAPE 0.058 vs 0.072 = 19% improvement."
        )

    st.divider()

    # ── Inventory reorder tool ────────────────────────────────────────────────
    st.subheader("Live inventory reorder calculator")
    st.markdown(
        "Select a SKU and enter its current on-hand inventory to get a real-time "
        "reorder recommendation derived from the LightGBM 80% prediction interval."
    )

    # SKU selector — extract category prefix for grouping
    policies["_cat"] = policies["id"].str.extract(r"^([A-Z]+)")
    categories = sorted(policies["_cat"].dropna().unique())
    sel_cat = st.selectbox("Category filter", ["All"] + categories)

    filtered = policies if sel_cat == "All" else policies[policies["_cat"] == sel_cat]
    sku_ids = filtered["id"].tolist()

    col_sku, col_inv = st.columns([2, 1])
    with col_sku:
        selected_id = st.selectbox("SKU", sku_ids)
    with col_inv:
        sku_row = policies[policies["id"] == selected_id].iloc[0]
        default_inv = int(sku_row["reorder_point"] * 0.7)
        current_inv = st.number_input(
            "Current on-hand inventory (units)",
            min_value=0,
            max_value=int(sku_row["order_up_to"] * 3),
            value=default_inv,
            step=1,
        )

    # Reorder decision
    s = float(sku_row["reorder_point"])
    S = float(sku_row["order_up_to"])
    should_reorder = current_inv <= s
    qty = max(0, math.ceil(S - current_inv)) if should_reorder else 0

    st.divider()

    # Decision card
    if should_reorder:
        st.error(
            f"## ⚠️ REORDER NOW — order **{qty:,} units**\n\n"
            f"Current inventory ({current_inv:,}) ≤ reorder point ({s:,.0f}).  \n"
            f"Order {qty:,} units to reach order-up-to level {S:,.0f}.",
        )
    else:
        headroom = current_inv - s
        st.success(
            f"## ✅ HOLD — no action needed\n\n"
            f"Current inventory ({current_inv:,}) > reorder point ({s:,.0f}).  \n"
            f"You have {headroom:,.0f} units of headroom before the next reorder trigger.",
        )

    # Policy parameters
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Mean daily demand", f"{sku_row['mean_daily_demand']:.2f} units")
    col_b.metric("Safety stock", f"{sku_row['safety_stock']:.0f} units")
    col_c.metric("Reorder point (s)", f"{s:,.0f} units")
    col_d.metric("Order-up-to (S)", f"{S:,.0f} units")

    col_e, col_f = st.columns(2)
    col_e.metric("Lead time", f"{int(sku_row['lead_time_days'])} days")
    col_f.metric("Target service level", f"{sku_row['service_level']:.0%}")

    st.divider()

    # Inventory level chart
    st.subheader("Inventory position visualisation")
    fig = go.Figure()

    x_range = np.linspace(0, S * 1.5, 300)
    fig.add_vline(x=s, line_dash="dash", line_color="orange",
                  annotation_text=f"Reorder point s = {s:,.0f}", annotation_position="top right")
    fig.add_vline(x=S, line_dash="dot", line_color="green",
                  annotation_text=f"Order-up-to S = {S:,.0f}", annotation_position="top left")
    fig.add_vline(x=current_inv, line_color="red" if should_reorder else "blue", line_width=3,
                  annotation_text=f"Current = {current_inv:,}", annotation_position="top right")
    fig.add_vrect(x0=0, x1=s, fillcolor="red", opacity=0.06, line_width=0, annotation_text="Reorder zone")
    fig.add_vrect(x0=s, x1=S, fillcolor="green", opacity=0.06, line_width=0, annotation_text="Safe zone")

    fig.update_layout(
        height=160,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title="Inventory level (units)",
        showlegend=False,
        yaxis_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Saved result charts
    chart_cols = st.columns(2)
    for i, (label, path) in enumerate([
        ("LightGBM vs baselines", ROOT / "projects/01_demand_forecasting/results/lgbm_model_comparison.png"),
        ("SHAP feature importance", ROOT / "projects/01_demand_forecasting/results/lgbm_shap_summary.png"),
    ]):
        with chart_cols[i % 2]:
            if path.exists():
                st.image(str(path), caption=label, use_container_width=True)

    st.caption(
        "**Key design:** lag features at multiples of 7 preserve weekday alignment. "
        "80% PI from quantile regression feeds directly into safety-stock formula (σ̂ = half-width / 1.282). "
        "One model across all 3,049 SKUs beats 3,049 individual models at a fraction of the compute."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: P2 — RETURNS INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔁 P2 — Returns Intelligence":
    st.title("🔁 Returns Intelligence Suite")
    st.caption(
        "Three models, one data schema: return likelihood scoring, fraud/abuse detection, "
        "and exchange recommendations — all from the same synthetic dataset."
    )

    customers, orders, returns = load_returns_data()

    if customers.empty or orders.empty or returns.empty:
        st.warning("Returns parquet files not found. Run `make train-returns` to generate them.")
        st.stop()

    # ── Dataset overview ──────────────────────────────────────────────────────
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(customers):,}")
    c2.metric("Orders", f"{len(orders):,}")
    c3.metric("Returns", f"{len(returns):,}")
    fraud_rate = returns["is_fraud"].mean()
    c4.metric("Fraud rate in returns", f"{fraud_rate:.1%}")

    st.divider()

    # ── Return rates ──────────────────────────────────────────────────────────
    st.subheader("Return rates by category and fraud archetype")
    col_l, col_r = st.columns(2)

    with col_l:
        cat_rr = orders.groupby("category")["was_returned"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(
            x=cat_rr.index, y=cat_rr.values,
            text=[f"{v:.0%}" for v in cat_rr.values], textposition="outside",
            marker_color=["#2196F3"] * len(cat_rr),
        ))
        fig.update_layout(
            title="Return rate by product category",
            yaxis_tickformat=".0%",
            height=350, margin=dict(t=50, b=30),
            yaxis_title="Return rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        orders_with_arch = orders.merge(customers[["customer_id", "archetype"]], on="customer_id")
        arch_rr = orders_with_arch.groupby("archetype")["was_returned"].mean().sort_values(ascending=False)
        colors = {"wardrober": "#F44336", "velocity": "#FF9800", "ring": "#9C27B0", "normal": "#4CAF50"}
        fig2 = go.Figure(go.Bar(
            x=arch_rr.index,
            y=arch_rr.values,
            text=[f"{v:.0%}" for v in arch_rr.values], textposition="outside",
            marker_color=[colors.get(k, "#888") for k in arch_rr.index],
        ))
        fig2.update_layout(
            title="Return rate by fraud archetype",
            yaxis_tickformat=".0%",
            height=350, margin=dict(t=50, b=30),
            yaxis_title="Return rate",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Fraud signals ─────────────────────────────────────────────────────────
    st.subheader("Fraud signal analysis")
    returns_rich = returns.merge(orders[["order_id", "item_price", "category"]], on="order_id")

    col_ll, col_rr = st.columns(2)

    with col_ll:
        # Days to return by fraud label
        fraud_ret = returns_rich[returns_rich["is_fraud"]]["days_to_return"].clip(upper=60)
        legit_ret = returns_rich[~returns_rich["is_fraud"]]["days_to_return"].clip(upper=60)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=legit_ret, name="Legitimate", opacity=0.6,
                                    marker_color="#4CAF50", nbinsx=30))
        fig3.add_trace(go.Histogram(x=fraud_ret, name="Fraud", opacity=0.6,
                                    marker_color="#F44336", nbinsx=30))
        fig3.update_layout(
            barmode="overlay", title="Days to return: fraud vs legitimate",
            xaxis_title="Days to return", height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_rr:
        # Return condition breakdown
        cond_fraud = returns_rich.groupby("condition")["is_fraud"].mean().sort_values(ascending=False)
        fig4 = go.Figure(go.Bar(
            x=cond_fraud.index, y=cond_fraud.values,
            text=[f"{v:.1%}" for v in cond_fraud.values], textposition="outside",
            marker_color=["#F44336", "#FF9800", "#4CAF50"][:len(cond_fraud)],
        ))
        fig4.update_layout(
            title="Fraud rate by return condition",
            yaxis_tickformat=".0%",
            height=300, margin=dict(t=50, b=30),
            yaxis_title="Fraud rate",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Graph feature insight ─────────────────────────────────────────────────
    st.subheader("Ring fraud: why graph features matter")
    ring_cust = customers[customers["archetype"] == "ring"]
    ring_fraud_rate = returns[returns["customer_id"].isin(ring_cust["customer_id"])]["is_fraud"].mean()
    normal_fraud_rate = returns[returns["customer_id"].isin(
        customers[customers["archetype"] == "normal"]["customer_id"])]["is_fraud"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Ring member fraud rate", f"{ring_fraud_rate:.0%}", delta=f"+{ring_fraud_rate - normal_fraud_rate:.0%} vs normal")
    col2.metric("Normal customer fraud rate", f"{normal_fraud_rate:.1%}")
    ring_addr_sharing = ring_cust["address_id"].value_counts()
    col3.metric("Ring members sharing one address", f"{ring_addr_sharing.max():.0f} max")

    st.info(
        "Ring members look individually normal — their return rates and conditions are plausible. "
        "The fraud signal lives on the **graph**: multiple accounts sharing a delivery address or "
        "payment hash. `component_size` and `shared_address_count` are the highest-gain features "
        "in the fraud model, lifting PR-AUC more than all behavioural features combined."
    )

    st.divider()

    # ── Exchange recommendation examples ─────────────────────────────────────
    st.subheader("Exchange recommendation logic")
    reason_counts = returns["reason_code"].value_counts()
    fig5 = go.Figure(go.Bar(
        x=reason_counts.index, y=reason_counts.values,
        marker_color="#2196F3",
        text=reason_counts.values, textposition="outside",
    ))
    fig5.update_layout(
        title="Return volume by reason code",
        height=280, margin=dict(t=50, b=30),
        yaxis_title="Return count",
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        """
        **Two-stage exchange recommendation:**

        | Stage | Covers | Speed | Explainability |
        |-------|--------|-------|---------------|
        | Heuristic rules | `too_small → next size up`, `defective → exact replacement`, `wrong_color → all colours` | Instant | Perfect |
        | Feature scoring | `changed_mind`, `not_as_described` (ambiguous cases) | < 5 ms | SHAP reason codes |

        ~70% of returns hit a heuristic rule directly. The ranker handles the long tail.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: P3 — CHECKOUT UPLIFT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 P3 — Checkout Uplift":
    st.title("🎯 Checkout Intent & Uplift Modelling")
    st.caption(
        "Propensity scores tell you who is *likely* to convert. "
        "Uplift scores tell you who you can actually *influence*. These are not the same thing."
    )

    df = load_uplift_data()
    if df.empty:
        st.warning("Uplift data parquet not found. Run `make data-criteo` then `make train-intent`.")
        st.stop()

    # ── The core insight ──────────────────────────────────────────────────────
    st.subheader("The propensity trap")
    st.markdown(
        """
        A standard intent model learns **P(convert | X)** — probability of converting regardless
        of whether you intervene. Targeting high-propensity users with a discount wastes budget on:

        - **Sure-things**: they convert anyway. You just gave away margin.
        - **Lost causes**: they won't convert regardless. Wasted spend.

        The right target is **persuadables** — users who convert *because of* the intervention
        but wouldn't without it. Uplift modelling estimates this **CATE** (Conditional Average
        Treatment Effect) directly.
        """
    )

    # ── Segment overview ──────────────────────────────────────────────────────
    if "segment" in df.columns:
        st.subheader("Segment composition and true effect sizes")
        seg_stats = (
            df.groupby("segment")
            .agg(
                count=("conversion", "size"),
                propensity=("conversion", "mean"),
                true_cate=(
                    "conversion",
                    lambda x: (
                        x[df.loc[x.index, "treatment"] == 1].mean()
                        - x[df.loc[x.index, "treatment"] == 0].mean()
                    ),
                ),
            )
            .reset_index()
        )

        col_l, col_r = st.columns(2)
        with col_l:
            fig = go.Figure()
            segs = seg_stats["segment"].tolist()
            colors_map = {
                "persuadable": "#2196F3",
                "sure_thing": "#4CAF50",
                "lost_cause": "#9E9E9E",
                "sleeping_dog": "#FF9800",
            }
            bar_colors = [colors_map.get(s, "#888") for s in segs]
            x_pos = list(range(len(segs)))
            fig.add_trace(go.Bar(
                x=segs, y=seg_stats["propensity"].tolist(),
                name="Propensity P(Y=1)", marker_color=bar_colors, opacity=0.7,
                text=[f"{v:.2f}" for v in seg_stats["propensity"]], textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=segs, y=seg_stats["true_cate"].tolist(),
                name="True CATE (uplift)", marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in seg_stats["true_cate"]], textposition="outside",
            ))
            fig.update_layout(
                barmode="group",
                title="Propensity vs true CATE by segment",
                height=360, margin=dict(t=50, b=30),
                yaxis_title="Rate / effect size",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "**Sure-things** have the highest propensity but near-zero CATE. "
                "Targeting by propensity fills the top bucket with sure-things — wasted budget."
            )

        with col_r:
            st.dataframe(
                seg_stats.rename(columns={
                    "segment": "Segment",
                    "count": "N",
                    "propensity": "P(convert) ↓",
                    "true_cate": "True CATE",
                })
                .assign(**{
                    "P(convert) ↓": seg_stats["propensity"].map("{:.3f}".format),
                    "True CATE": seg_stats["true_cate"].map("{:+.4f}".format),
                }),
                hide_index=True,
                use_container_width=True,
                height=200,
            )
            st.divider()
            st.markdown("**What each segment means for targeting:**")
            st.markdown(
                "- 🔵 **Persuadable** — target these. High CATE, medium propensity.\n"
                "- 🟢 **Sure-thing** — skip. They'll convert anyway; discounting costs margin.\n"
                "- ⚫ **Lost cause** — skip. No amount of incentive will move them.\n"
                "- 🟠 **Sleeping dog** — avoid. Intervention *reduces* conversion probability."
            )

    st.divider()

    # ── Qini curves ──────────────────────────────────────────────────────────
    st.subheader("Qini curve comparison: targeting policies")

    y = df["conversion"].values.astype(float)
    w = df["treatment"].values.astype(float)
    feature_cols = [f"f{i}" for i in range(12)]

    # Compute three scoring policies
    # Oracle: use ground-truth persuadable label
    if "segment" in df.columns:
        oracle_score = (df["segment"] == "persuadable").astype(float).values
    else:
        oracle_score = None

    # Propensity proxy: features correlated with high base conversion (sure-things have high f0)
    propensity_score = (df["f0"] + 0.6 * df["f1"] + 0.3 * df["f2"]).values
    # Uplift proxy: features correlated with treatment response (persuadables have high f4)
    uplift_score = (df["f4"] + 0.7 * df["f5"] - 0.5 * df["f0"]).values
    # Random baseline
    random_score = np.random.default_rng(42).random(len(df))

    with st.spinner("Computing Qini curves…"):
        policies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if oracle_score is not None:
            policies["Oracle (persuadable ground truth)"] = _qini_curve(y, w, oracle_score)
        policies["Uplift proxy (f4+f5−f0)"] = _qini_curve(y, w, uplift_score)
        policies["Propensity proxy (f0+f1+f2)"] = _qini_curve(y, w, propensity_score)
        policies["Random targeting"] = _qini_curve(y, w, random_score)

    fig_qini = go.Figure()
    style_map = {
        "Oracle (persuadable ground truth)": ("#2196F3", "solid", 3),
        "Uplift proxy (f4+f5−f0)": ("#4CAF50", "solid", 2),
        "Propensity proxy (f0+f1+f2)": ("#F44336", "dash", 2),
        "Random targeting": ("#9E9E9E", "dot", 1),
    }
    for name, (fracs, incr) in policies.items():
        color, dash, width = style_map.get(name, ("#888", "solid", 1))
        q = _qini_coeff(fracs, incr)
        fig_qini.add_trace(go.Scatter(
            x=fracs * 100, y=incr,
            name=f"{name}  (Qini={q:+.1f})",
            line=dict(color=color, dash=dash, width=width),
        ))

    # Random baseline line
    total_incr = list(policies.values())[0][1][-1]
    fig_qini.add_trace(go.Scatter(
        x=[0, 100], y=[0, total_incr],
        name="Random (reference)", line=dict(color="#9E9E9E", dash="dot", width=1),
        showlegend=False,
    ))

    fig_qini.update_layout(
        title="Qini curves — who gets targeted matters",
        xaxis_title="% of users targeted (budget)",
        yaxis_title="Incremental conversions above random",
        height=420, margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_qini, use_container_width=True)

    st.caption(
        "Positive Qini = better than random. The uplift proxy targets persuadables and outperforms "
        "the propensity proxy, which fills its top bucket with sure-things. "
        "In production, T/S-learner models replace these linear proxies — "
        "see `report.ipynb` for the full comparison."
    )

    st.divider()

    # ── Budget targeting tool ─────────────────────────────────────────────────
    st.subheader("Interactive targeting budget")
    budget_pct = st.slider("Targeting budget (% of users)", 5, 50, 20, step=5)

    if "segment" in df.columns:
        n_target = int(len(df) * budget_pct / 100)
        budget_rows = []
        for name, score in [
            ("Uplift proxy", uplift_score),
            ("Propensity proxy", propensity_score),
            ("Random", random_score),
        ]:
            top_idx = np.argsort(score)[::-1][:n_target]
            top_segs = df["segment"].values[top_idx]
            seg_counts = pd.Series(top_segs).value_counts(normalize=True)
            treated_conv = (y * w)[top_idx].sum()
            control_conv = (y * (1 - w))[top_idx].sum()
            nt = w[top_idx].sum()
            nc = n_target - nt
            incr = treated_conv - control_conv * (nt / max(nc, 1))
            budget_rows.append({
                "Policy": name,
                f"Persuadables in top {budget_pct}%": f"{seg_counts.get('persuadable', 0):.0%}",
                f"Sure-things in top {budget_pct}%": f"{seg_counts.get('sure_thing', 0):.0%}",
                "Est. incremental conversions": f"{incr:,.0f}",
            })

        st.dataframe(pd.DataFrame(budget_rows), hide_index=True, use_container_width=True)
        st.caption(
            "At a fixed budget, the uplift proxy fills the top bucket with persuadables. "
            "The propensity proxy fills it with sure-things — who would have converted anyway."
        )
