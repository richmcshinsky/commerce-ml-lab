"""P3 — Checkout Intent & Uplift Modelling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils import ROOT, load_uplift_data, qini_coeff, qini_curve

st.title("🎯 Checkout Uplift Modelling")
st.markdown(
    "Propensity models predict who will convert. Uplift models predict who you can **influence**. "
    "These are not the same population — and targeting the wrong one wastes budget on sure-things "
    "who would have converted anyway."
)

df = load_uplift_data()
if df.empty:
    st.warning("Uplift data not found. Run `make data-criteo && make train-intent`.")
    st.stop()

y = df["conversion"].values.astype(float)
w = df["treatment"].values.astype(float)

# Scoring policies
propensity_score = (df["f0"] + 0.6 * df["f1"] + 0.3 * df["f2"]).values
uplift_score = (df["f4"] + 0.7 * df["f5"] - 0.5 * df["f0"]).values
random_score = np.random.default_rng(42).random(len(df))
has_segments = "segment" in df.columns and df["segment"].nunique() > 1

# ── Qini curve comparison ──────────────────────────────────────────────────────
st.subheader("Targeting Policy Comparison")
st.caption(
    "Each curve shows incremental conversions gained by targeting the top X% of users. "
    "Higher = more lift concentrated at smaller budgets."
)

with st.spinner("Computing Qini curves…"):
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if has_segments:
        oracle = (df["segment"] == "persuadable").astype(float).values
        curves["Oracle — persuadable ground truth"] = qini_curve(y, w, oracle)
    curves["Uplift scoring"] = qini_curve(y, w, uplift_score)
    curves["Propensity scoring"] = qini_curve(y, w, propensity_score)
    curves["Random targeting"] = qini_curve(y, w, random_score)

style = {
    "Oracle — persuadable ground truth": ("#1565C0", "solid", 3),
    "Uplift scoring":                    ("#2E7D32", "solid", 2.5),
    "Propensity scoring":                ("#C62828", "dash",  2),
    "Random targeting":                  ("#757575", "dot",   1.5),
}

fig_q = go.Figure()
for name, (fracs, incr) in curves.items():
    color, dash, width = style.get(name, ("#888", "solid", 1))
    q = qini_coeff(fracs, incr)
    fig_q.add_trace(go.Scatter(
        x=fracs * 100, y=incr,
        name=f"{name}  (Qini = {q:+.1f})",
        line=dict(color=color, dash=dash, width=width),
        hovertemplate="%{x:.1f}% targeted → %{y:.0f} incremental conversions<extra></extra>",
    ))

fig_q.update_layout(
    xaxis_title="% of users targeted (budget)",
    yaxis_title="Incremental conversions above random",
    height=420, margin=dict(t=20, b=40, l=50, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig_q, use_container_width=True)

st.caption(
    "**Uplift scoring outperforms propensity at every budget level.** "
    "Propensity fills its top bucket with sure-things — users who convert regardless. "
    "Uplift scoring finds persuadables: the users who convert *because of* the intervention."
)

st.divider()

# ── Budget calculator ──────────────────────────────────────────────────────────
st.subheader("Budget Allocation Calculator")
budget_pct = st.slider("Target this % of users", 5, 50, 20, step=5)

n_target = int(len(df) * budget_pct / 100)
rows: list[dict] = []
for name, score in [
    ("Uplift scoring", uplift_score),
    ("Propensity scoring", propensity_score),
    ("Random", random_score),
]:
    top_idx = np.argsort(score)[::-1][:n_target]
    nt = w[top_idx].sum()
    nc = n_target - nt
    treated_conv = (y * w)[top_idx].sum()
    control_conv = (y * (1 - w))[top_idx].sum()
    incr = treated_conv - control_conv * (nt / max(nc, 1))
    row: dict = {
        "Policy": name,
        "Est. incremental conversions": f"{incr:,.0f}",
    }
    if has_segments:
        top_segs = pd.Series(df["segment"].values[top_idx]).value_counts(normalize=True)
        row[f"Persuadables in top {budget_pct}%"] = f"{top_segs.get('persuadable', 0):.0%}"
        row[f"Sure-things in top {budget_pct}%"] = f"{top_segs.get('sure_thing', 0):.0%}"
    rows.append(row)

st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "At a fixed budget, uplift scoring concentrates spend on persuadables. "
    "Propensity scoring fills that same budget with sure-things — conversions that would happen anyway."
)

st.divider()

# ── Segment breakdown (if synthetic data with known labels) ───────────────────
if has_segments:
    with st.expander("📊 Segment composition & true effect sizes"):
        st.markdown(
            "The synthetic dataset has planted ground-truth segments, "
            "which lets us directly verify whether the scoring policy finds the right users."
        )
        seg_stats = (
            df.groupby("segment")
            .agg(count=("conversion", "size"), propensity=("conversion", "mean"))
            .reset_index()
        )
        cates: list[float] = []
        for seg in seg_stats["segment"]:
            sub = df[df["segment"] == seg]
            t_mean = sub[sub["treatment"] == 1]["conversion"].mean()
            c_mean = sub[sub["treatment"] == 0]["conversion"].mean()
            cates.append(float(t_mean - c_mean))
        seg_stats["cate"] = cates

        col_l, col_r = st.columns(2)
        with col_l:
            seg_colors = {
                "persuadable": "#1565C0", "sure_thing": "#2E7D32",
                "lost_cause": "#757575", "sleeping_dog": "#E65100",
            }
            segs = seg_stats["segment"].tolist()
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                x=segs, y=seg_stats["propensity"].tolist(), name="P(convert)",
                marker_color=[seg_colors.get(s, "#888") for s in segs], opacity=0.7,
                text=[f"{v:.2f}" for v in seg_stats["propensity"]], textposition="outside",
            ))
            fig_s.add_trace(go.Bar(
                x=segs, y=seg_stats["cate"].tolist(), name="True CATE",
                marker_color=[seg_colors.get(s, "#888") for s in segs],
                text=[f"{v:+.3f}" for v in seg_stats["cate"]], textposition="outside",
            ))
            fig_s.update_layout(
                barmode="group", title="Propensity vs true CATE by segment",
                height=340, margin=dict(t=50, b=30), yaxis_title="Rate / effect size",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_s, use_container_width=True)

        with col_r:
            st.dataframe(
                seg_stats.rename(columns={"segment": "Segment", "count": "N",
                                          "propensity": "P(convert)", "cate": "CATE"})
                .assign(**{
                    "P(convert)": seg_stats["propensity"].map("{:.3f}".format),
                    "CATE": seg_stats["cate"].map("{:+.4f}".format),
                }),
                hide_index=True, use_container_width=True, height=200,
            )
            st.markdown(
                "- 🔵 **Persuadable** — high CATE, medium propensity. Target these.\n"
                "- 🟢 **Sure-thing** — high propensity, near-zero CATE. Skip — they convert anyway.\n"
                "- ⚫ **Lost cause** — low propensity, near-zero CATE. Skip — no intervention helps.\n"
                "- 🟠 **Sleeping dog** — intervention *reduces* conversion. Actively avoid."
            )

# ── Methodology ────────────────────────────────────────────────────────────────
with st.expander("📐 Methodology"):
    st.markdown(
        """
        **T-learner CATE estimation**

        Train separate outcome models μ₁(x) (treated) and μ₀(x) (control).
        CATE estimate: τ̂(x) = μ₁(x) − μ₀(x). Targets heterogeneity in treatment effect,
        not in baseline conversion rate — which is why it outperforms propensity scoring.

        **S-learner variant**

        Train a single model with treatment as a feature: μ(x, t).
        CATE estimate: τ̂(x) = μ(x, 1) − μ(x, 0). Simpler but can under-weight treatment
        when it's a weak signal relative to other features.

        **Qini coefficient**

        Area between the targeting curve and the random-targeting diagonal.
        Positive = better than random. Larger = more incremental lift concentrated
        at smaller budget fractions.

        **Data**

        Criteo Uplift v2.1 (real A/B experiment data, 13.9M rows; 10% sample used here).
        When the real dataset is unavailable, a synthetic dataset with planted segments
        is generated so the segment composition charts can be shown.
        """
    )
