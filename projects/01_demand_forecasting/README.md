# Project 01: Demand Forecasting

> Given sales history and supply chain lead times, when and how much should you reorder?

This project models retail demand at the SKU × store level using the M5
Forecasting dataset (Walmart), then translates probabilistic forecasts into
actionable inventory reorder decisions using the newsvendor model.

The key framing: a demand forecast is only useful if it drives a decision.
This project builds the bridge from "here is what we expect to sell" to
"here is whether and how much to reorder given your lead time and cost structure."

---

## Results

*(To be filled in after training)*

| Model | WMAPE | MASE | Notes |
|-------|-------|------|-------|
| Naive baseline | — | 1.00 (reference) | Last observed value |
| Seasonal naive (7-day) | — | — | Same day, prior week |
| Moving average (28-day) | — | — | |
| ETS / Holt-Winters | — | — | |
| LightGBM global model | — | — | One model, all SKUs |

---

## Data

M5 Forecasting dataset, Store CA_1 (~3,049 SKUs × ~1,913 days).
See [docs/data-sources.md](../../docs/data-sources.md) for download instructions.

```bash
make data-m5
```

---

## Notebooks (run by hand)

| Notebook | Contents |
|----------|----------|
| `notebooks/01_eda.ipynb` | Dataset exploration — sales distributions, zero rates, seasonality |
| `notebooks/02_baselines.ipynb` | Naive, seasonal naive, and moving average baselines |
| `notebooks/03_lightgbm.ipynb` | Global LightGBM model — feature engineering, training, evaluation |
| `notebooks/04_inventory_optimization.ipynb` | Newsvendor model, safety stock, and (s, S) policy simulation |

---

## Production code

```
src/forecasting/
├── data.py       # M5 loading, melt wide→long, train/test split
├── features.py   # Lag, rolling, calendar, and price features
├── models.py     # Baseline, ETS, and LightGBM model classes
├── inventory.py  # Newsvendor, safety stock, (s, S) reorder policy
└── evaluate.py   # Walk-forward backtesting, metric summary
```

---

## API

```bash
make serve-forecast
# Open http://localhost:8001/docs
```

Endpoints:
- `POST /forecast` — point forecasts + 80% prediction intervals for a SKU
- `POST /reorder` — reorder recommendation given current inventory + lead time

---

## Key design decisions

- **Baselines first.** Seasonal naive establishes the floor. All ML models are
  compared against it, not against each other in isolation.
- **Global model.** One LightGBM model trained across all 3,049 series with
  SKU and category as categorical features. This is modern best practice:
  global models consistently beat per-series models on M5.
- **Inventory decision layer.** The newsvendor model translates forecast mean
  and variance into an order quantity that minimises expected cost given
  stated overage and underage costs. Most ML forecasting projects stop before
  this step.

---

## Limitations

- Trained on one Walmart store (CA_1), not all stores. Generalisation to other
  retail formats (fast fashion, electronics) is untested.
- Demand is assumed independent across SKUs. In practice, promotions and
  stockouts create cross-SKU effects.
- Lead time is treated as deterministic. Real supply chains have uncertain
  lead times, especially for imported goods.
- No demand-price interaction. Markdowns affect both clearance and demand;
  this model treats price as an exogenous feature.
