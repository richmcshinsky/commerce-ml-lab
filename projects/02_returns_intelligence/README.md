# Project 02: Returns Intelligence

> Three models, one data schema — return likelihood scoring, fraud detection, and exchange recommendations.

This project builds a unified returns intelligence system: the same synthetic
customer/order/return dataset feeds three distinct models serving three
different business decisions.

The design mirrors how a production ML system at a returns platform would work:
one data pipeline, multiple model consumers.

---

## The three models

### Model A: Return Likelihood
`P(return | order features)` — scored at order fulfillment time.

Used for: risk-based fulfilment routing, aggregate return forecasting by store/week,
and as a feature in the fraud model.

### Model B: Fraud & Abuse Detection
`P(fraud | return + customer history + graph features)` — scored at return initiation.

Fraud archetypes detected:
- **Wardrober**: buys high-price apparel, returns worn items
- **Velocity returner**: places many orders in short windows, returns most
- **Address-sharing ring**: multiple accounts sharing address or payment method

Design: LightGBM on behavioural features + graph features from networkx →
isotonic calibration → cost-aware threshold selection.

### Model C: Exchange Recommendation
Given return + reason code, rank exchange candidates.

Design: heuristic filter by reason code (covers ~70% of cases, instantly
explainable) + LightGBM ranker (LambdaRank) on top for fine-grained scoring.

---

## Results

*(To be filled in after training)*

### Return likelihood
| Model | PR-AUC | Notes |
|-------|--------|-------|
| Rules baseline (return_rate > 0.3) | — | |
| LightGBM | — | |

### Fraud detection
| Model | PR-AUC | Precision@1% | Recall@1%FPR | Notes |
|-------|--------|--------------|-------------|-------|
| Rules baseline | — | — | — | |
| LightGBM (no graph) | — | — | — | |
| LightGBM + graph features | — | — | — | Graph lift |

---

## Data

Synthetic dataset generated locally (100k customers, ~350k orders, ~42k returns,
~2% fraud rate in returns). See [docs/data-sources.md](../../docs/data-sources.md).

```bash
make data-synthetic
```

---

## Notebooks (run by hand)

| Notebook | Contents |
|----------|----------|
| `notebooks/01_synthetic_data.ipynb` | Data generation, archetype inspection, EDA |
| `notebooks/02_return_likelihood.ipynb` | Return likelihood model — features, training, calibration |
| `notebooks/03_fraud_detection.ipynb` | Fraud model — graph features, threshold selection, SHAP |
| `notebooks/04_exchange_recs.ipynb` | Exchange recommender — heuristics, ranker, hit-rate evaluation |

---

## Production code

```
src/returns/
├── synthetic.py   # Re-exports from shared data generator
├── likelihood.py  # Return likelihood model
├── fraud.py       # Fraud detection with graph features + SHAP
├── exchange.py    # Heuristic filter + LightGBM ranker
└── evaluate.py    # Metrics: PR-AUC, Precision@K, cost-aware threshold
```

---

## API

```bash
make serve-returns
# Open http://localhost:8002/docs
```

Endpoints:
- `POST /returns/score` — return likelihood for an order
- `POST /returns/fraud` — fraud score for a return event
- `POST /returns/exchange` — exchange candidates given return + reason

---

## Key design decisions

- **One synthetic dataset, three models.** Mirrors production architecture
  where returns data is shared infrastructure, not siloed per model.
- **Graph features.** Shared address and payment edges between accounts are
  the primary signal for ring fraud. Graph degree and connected-component size
  lift PR-AUC more than any single behavioural feature.
- **Cost-aware threshold.** The fraud threshold is chosen to minimise
  `cost_fp × FP + cost_fn × FN` for a stated business cost ratio, not set
  at 0.5. The ratio of FP cost (annoyed customer) to FN cost (missed fraud
  value) is a business input, explicitly documented.
- **Heuristics first for exchange.** The reason-code heuristic filter handles
  the obvious cases (too small → next size up) instantly and with perfect
  explainability. The ranker adds value on ambiguous cases ("changed_mind",
  "didn't like fit") where heuristics don't have a clear answer.

---

## Limitations

- Synthetic data has cleaner signals than production data. Real wardrobers
  adapt their patterns over time; the model would degrade without regular
  retraining.
- Label noise in production. "Fraud" is often discovered only after
  investigation; real datasets have both unknown positives and disputed labels.
- Graph features require shared infrastructure at inference time. Computing
  connected components on a growing customer graph in low latency requires
  either pre-computation or a dedicated graph database.
