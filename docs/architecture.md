# Architecture decisions

Design decisions made in this repository, and the reasoning behind them.

---

## Monorepo over separate repos

One monorepo rather than three independent projects. The shared infrastructure
(`src/commerce_ml/`) — data loaders, feature engineering, metrics, plotting —
is used by all three projects. 

## LightGBM as the primary model

All tabular models in this portfolio use LightGBM, not deep neural networks.
This is a deliberate choice, not a limitation:

- Gradient-boosted trees consistently match or beat deep learning on structured
  tabular data in every major Kaggle retail competition.
- Training is fast (minutes vs. hours), reproducible, and interpretable via SHAP.
- The portfolio includes honest comparisons where applicable ("what would a
  deep model add here, and is it worth the complexity?").

Deep learning appears in Project 03 as an explicitly-considered alternative,
not as the default.

## Synthetic data for returns

No clean public returns fraud dataset exists. Options were:

1. Use a generic fraud dataset (credit card, insurance) — wrong domain.
2. Find a returns-adjacent dataset and reframe — misleading.
3. Generate synthetic data with documented archetypes — correct choice.

Building synthetic data with planted archetypes (wardrobers, velocity returners,
address-sharing rings) and a known fraud rate allows honest evaluation: we know
the ground truth. The data generator is documented and reproducible.

## FastAPI for production services

Projects 01 and 02 each have a FastAPI service with Pydantic schemas and
auto-generated Swagger documentation at `/docs`. This choice over alternatives:

- **vs. Streamlit**: engineers don't demo products with Streamlit; they make
  API calls. A `/predict` endpoint is closer to how these models would actually
  be consumed.
- **vs. bare Python scripts**: scripts don't have schemas, error handling, or
  a way to be called from other services. FastAPI shows production intent.
- **vs. Flask**: Pydantic schemas and async support make FastAPI the current
  standard for new Python ML APIs.

## Walk-forward backtesting (not k-fold)

Forecasting models are evaluated with walk-forward (expanding window)
cross-validation, not k-fold. Shuffling time-series data causes leakage:
a model trained on a shuffled split can see future data in training, producing
unrealistically low validation error. Walk-forward evaluation simulates
production conditions: train on the past, predict the future.

## WMAPE over MAPE

The M5 competition and most retail forecasting literature uses WMAPE (Weighted
Mean Absolute Percentage Error) rather than MAPE. MAPE is undefined when actual
sales are zero (common in retail for slow-moving SKUs) and treats a miss on
a high-volume product the same as a miss on a low-volume product. WMAPE weights
errors proportionally to actual volume and avoids division by zero.

## Cost-aware fraud threshold

The fraud model uses a threshold selected to minimise expected business cost
given a stated FP/FN cost matrix, not a default threshold of 0.5. This matters
because false positives (flagging a legitimate return) have a real customer
goodwill cost, while false negatives (missing fraud) have a direct financial
cost. The optimal threshold depends on the ratio of these costs, which is a
business input, not a model hyperparameter.

## Propensity vs. uplift for intent

The checkout intent project explicitly demonstrates that propensity scores are
the wrong trigger for intervention. A high-propensity shopper is likely to
convert anyway — discounting them wastes margin. A low-propensity shopper is
unlikely to convert regardless — discounting them rarely changes behaviour.
Uplift modelling estimates the *incremental* effect of the intervention,
identifying the "persuadables" who are neither sure-things nor lost-causes.
This distinction is routinely missed in practice.
