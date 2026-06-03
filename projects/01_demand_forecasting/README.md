# Project 01: Demand Forecasting

> Given sales history and supply chain lead times, when and how much should you reorder?

This project models retail demand at the SKU × store level using the M5
Forecasting dataset (Walmart CA_1), then translates probabilistic forecasts into
actionable inventory reorder decisions using the newsvendor model and (s, S) policy.

The key framing: **a demand forecast is only useful if it drives a decision.**
This project builds the full bridge from raw sales history → ML forecast →
inventory policy → simulated cost savings.

---

## Results

### Forecast accuracy — M5 CA_1, 28-day horizon

| Model | WMAPE | MASE | RMSSE | Notes |
|-------|------:|-----:|------:|-------|
| **LightGBM (global)** | **0.058** | 0.807 | 0.615 | One model, all 3,049 SKUs |
| Seasonal naive (7-day) | 0.072 | 1.000 | 0.681 | Baseline — same weekday, prior week |
| Moving average (28-day) | 0.151 | 2.107 | 1.557 | |
| Naive (last value) | 0.295 | 4.104 | 2.676 | |

LightGBM improves on the seasonal naive baseline by **19%** (WMAPE 0.058 vs 0.072).
MASE < 1.0 confirms the model beats the seasonal naive on its own scale.

### Prediction interval calibration

- **SKU-level 80% interval coverage: 86%** (target: 80%) — slightly conservative, safe for inventory buffers
- Store-aggregate interval calibrated from empirical residuals; correctly accounts for positive cross-SKU correlation

### Inventory simulation — 90-day, 7-day lead time, 95% SL target

| SKU type | Policy | Fill rate | Total cost vs. naive |
|----------|--------|----------:|---------------------:|
| Fast-moving | Optimised (s,S) | 100% | ~13% lower |
| Fast-moving | Naive fixed | 100% | baseline |
| Medium-moving | Optimised (s,S) | 99% | lower |
| Slow-moving | Optimised (s,S) | 94%* | comparable |

\* Normal approximation for discrete intermittent demand cannot always achieve the
exact target SL. See Limitations.

---

## Data

M5 Forecasting dataset, Store CA_1 (~3,049 SKUs × ~1,913 days ≈ 5.8M rows).

```bash
make data-m5   # requires Kaggle account + M5 competition terms accepted
```

See [docs/data-sources.md](../../docs/data-sources.md) for full instructions.

---

## Notebooks

Run in order. Each is self-contained but builds on prior results.

| Notebook | Contents |
|----------|----------|
| `notebooks/01_eda.ipynb` | Sales distributions, zero-inflation, weekly seasonality, price variation |
| `notebooks/02_baselines.ipynb` | Naive, seasonal naive, and 28-day moving average baselines |
| `notebooks/03_lightgbm.ipynb` | Global LightGBM — feature engineering, training, walk-forward backtest, SHAP importance |
| `notebooks/04_inventory_optimization.ipynb` | (s,S) policy per SKU, service-level frontier, newsvendor Q*, simulation |
| `report.ipynb` | **Polished end-to-end narrative** — the full story in one clean document |

---

## Production code

```
src/forecasting/
├── data.py       # M5 loading: melt wide→long, join calendar/prices, train/test split
├── features.py   # Lag (7/14/28/91/364d), rolling mean/std, calendar, Fourier, price features
├── lgbm_model.py # LGBMForecaster — fit, batched recursive predict, 80% PI, save/load
├── models.py     # BaseForecaster, Naive, SeasonalNaive, MovingAverage baselines
├── inventory.py  # Newsvendor Q*, safety stock, reorder point, (s,S) simulation
├── train.py      # CLI entry point: make train-forecast
└── evaluate.py   # Walk-forward splits, WMAPE/MASE/RMSSE metrics
```

---

## API

```bash
make train-forecast   # ~3 min on a modern laptop; saves results/lgbm_model.pkl
make serve-forecast   # starts uvicorn on :8001
# Open http://localhost:8001/docs
```

Endpoints:
- `POST /forecast` — 28-day point forecasts + 80% prediction intervals for any SKU
- `POST /reorder` — reorder recommendation (reorder_now, recommended_quantity, reorder_point) given current inventory, lead time, and cost parameters

---

## Key design decisions

**Baselines before models.** Seasonal naive establishes the floor. Every ML model is
evaluated against it, not against weaker alternatives. MASE < 1 is the minimum bar.

**Global model.** One LightGBM trained across all 3,049 series simultaneously.
Categorical features (item_id, dept_id, cat_id) let it specialise per series while
learning shared seasonality patterns once. Global models consistently outperform
per-series models on M5-style retail data at a fraction of the compute cost.

**Lag alignment at multiples of 7.** Lag features at 7, 14, 28, 91, 364 days
preserve day-of-week alignment — "lag_7 on a Monday" is always "last Monday's sales."
This is the single most important feature engineering decision for weekly-seasonal data.

**Batched recursive prediction.** Test forecasts are generated 7 days at a time,
feeding predictions back as lag inputs for the next batch. This avoids the NaN
collapse that occurs when all test lags point to unobserved future values.

**Interval → σ → policy.** The 80% quantile interval from the model feeds directly
into the (s, S) inventory formulas via σ̂ = half-width / 1.282. This closes the
loop from ML output to operational decision without a separate uncertainty model.

---

## Limitations

- **Single store.** Trained on CA_1 only. Cross-store or cross-retailer transfer is
  untested; different store formats have materially different demand patterns.
- **Independent SKU demand.** No cross-SKU substitution or cannibalism. A stockout
  on one SKU in practice lifts demand on its neighbours.
- **Fixed lead time.** Real supply chains have stochastic lead times. A 1-day lead
  time variance requires ~40% more safety stock at 95% SL.
- **Normal distribution for slow movers.** The safety-stock formula over-estimates
  buffer stock for SKUs with mean demand < 1 unit/day (~60% of M5). A Poisson or
  negative-binomial model would be more appropriate for the intermittent long tail.
- **No promotional uplift.** Markdowns affect both sell-through and demand volume.
  The model treats price as an exogenous input, not a decision variable.

---

## What I'd do with more time

- **Discrete demand model for slow movers.** Replace the normal safety-stock formula
  with a Poisson or negative-binomial distribution for SKUs with mean < 1 unit/day.
- **Hierarchical reconciliation.** Reconcile bottom-up forecasts at category / store /
  total levels using MinT to enforce aggregate consistency.
- **Price elasticity.** Estimate demand-price curves per category; couple into an
  optimal markdown policy alongside the inventory layer.
- **Multi-echelon inventory.** Extend from single-store to DC → store with correlated
  demand and safety stock pooling across locations.
- **Online learning.** Retrain incrementally with new data via LightGBM's `init_model`
  rather than full retraining nightly.
- **Conformal prediction.** Replace quantile regression with split conformal prediction
  for distribution-free, coverage-guaranteed prediction intervals.
