# commerce-ml-lab

[![CI](https://github.com/richmcshinsky/commerce-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/richmcshinsky/commerce-ml-lab/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://richmcshinsky-commerce-ml-lab.streamlit.app)

A focused ML portfolio covering four problem areas at the intersection of
e-commerce operations and machine learning: demand forecasting, returns
intelligence, checkout intent optimisation, and shipping price optimisation.

Built as a working demonstration of how I think about commerce ML problems
end-to-end — from framing and baselines through to production-ready code and
inventory/business decision layers.

---

## Projects

| # | Project | Problem | Data | Report | Status |
|---|---------|---------|------|--------|--------|
| 01 | [Demand Forecasting](projects/01_demand_forecasting/) | Given sales history and supply chain lead times, when and how much should you reorder? | M5 (Walmart, Kaggle) | [📓 report](projects/01_demand_forecasting/report.ipynb) | ✅ Complete — WMAPE 0.058, 19% vs seasonal naive; (s,S) policy per SKU |
| 02 | [Returns Intelligence](projects/02_returns_intelligence/) | Return likelihood scoring, fraud/abuse detection, and exchange recommendations — three models, one data schema | Synthetic (generated) | [📓 report](projects/02_returns_intelligence/report.ipynb) | ✅ Complete — Fraud PR-AUC 0.560 (23× random); graph features detect rings; cost-aware threshold |
| 03 | [Checkout Intent & Uplift](projects/03_checkout_intent/) | Session behaviour predicts checkout probability; uplift modelling identifies who actually benefits from an intervention | Criteo Uplift v2 | [📓 report](projects/03_checkout_intent/report.ipynb) | ✅ Complete — T/S-learner CATE; segment decomposition shows sure-things (15.9% propensity, +1.0% CATE) vs persuadables (+12.3% CATE) |
| 04 | [Shipping Price Optimisation](projects/04_shipping_optimization/) | Which shipping option to show and at what price to maximise margin without hurting conversion — per checkout session | Synthetic A/B test (generated) | — | ✅ Complete — causal elasticity model; per-session optimisation; +33% expected margin vs flat rate |

---

## Philosophy

**Baselines first.** Every project establishes a simple baseline before fitting
any ML model. A seasonal naive forecast or a rules-based fraud flag that takes
30 minutes to build is the benchmark everything else must beat. Most "ML
improvements" in industry are improvements over a missing baseline.

**Connect forecasts to decisions.** A demand forecast that doesn't tell you
whether to reorder is incomplete. Each project here has a decision layer: the
newsvendor reorder policy sits on top of the demand forecast; the cost-aware
fraud threshold sits on top of the fraud score; the uplift policy sits on top
of the propensity model.

**Honest about limitations.** Every project README has a limitations section.
Models trained on synthetic or public data behave differently on production
data with its own distribution shifts, label noise, and adversarial patterns.

**Right metric for the problem.** WMAPE (not MAPE) for forecasting. PR-AUC
(not ROC-AUC) for imbalanced fraud detection. Qini coefficient (not AUC) for
uplift evaluation.

---

## Repository structure

```
commerce-ml-lab/
├── src/commerce_ml/          # Shared library: data, features, metrics, viz
├── projects/
│   ├── 01_demand_forecasting/
│   │   ├── report.ipynb      # Polished end-to-end narrative
│   │   ├── notebooks/        # Exploration notebooks (run by hand)
│   │   ├── src/forecasting/  # Production Python code
│   │   └── api/              # FastAPI /forecast and /reorder endpoints
│   ├── 02_returns_intelligence/
│   │   ├── report.ipynb
│   │   ├── notebooks/
│   │   ├── src/returns/      # likelihood.py, fraud.py, exchange.py
│   │   └── api/              # FastAPI /returns/{score,fraud,exchange}
│   └── 03_checkout_intent/
│       ├── report.ipynb
│       ├── notebooks/
│       └── src/intent/
├── data/                     # Gitignored — download via `make data`
├── docs/                     # Architecture decisions and future work
└── tests/                    # Shared library tests
```

---

## Interactive demo

The portfolio app is live on Streamlit Cloud — no setup required:

**[richmcshinsky-commerce-ml-lab.streamlit.app](https://richmcshinsky-commerce-ml-lab.streamlit.app)**

To run locally:

```bash
streamlit run streamlit_app.py
```

All result data (parquets, CSVs, charts) is committed to the repo, so every page loads immediately without running training first.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/richmcshinsky/commerce-ml-lab
cd commerce-ml-lab

# 2. Install (requires uv — https://docs.astral.sh/uv/)
uv sync --all-extras

# 3. Download datasets
make data-criteo          # ~700 MB, no auth required
make data-m5              # requires Kaggle account + M5 terms accepted

# 4. Generate synthetic data and train all models
make data-synthetic       # returns intelligence synthetic data
make train-returns        # returns intelligence models
make train-shipping       # shipping price optimisation model (no download required)

# 5. Run tests
make test

# 6. Serve APIs locally
make serve-returns        # http://localhost:8002/docs
make serve-shipping       # http://localhost:8003/docs
```

---

## Key design decisions

See [`docs/architecture.md`](docs/architecture.md) for full rationale. Brief
summary:

- **One shared library** (`src/commerce_ml/`) for data loading, features,
  metrics, and visualisation. Three projects share it rather than duplicating.
- **LightGBM as the primary ML tool** for all tabular models. It consistently
  outperforms deep models on structured retail data; choosing it over a
  transformer is a deliberate signal about practical ML judgment.
- **Synthetic data for returns** because no clean public returns fraud dataset
  exists. Building good synthetic data with known archetypes is itself a skill,
  and it makes evaluation honest (we know the ground truth).
- **FastAPI for production code** (Projects 01 and 02) because a `/predict`
  endpoint with Pydantic schemas and auto-generated Swagger UI reads cleanly
  to engineering teams.

---

## What I'd build with more time

See [`docs/future-work.md`](docs/future-work.md).
