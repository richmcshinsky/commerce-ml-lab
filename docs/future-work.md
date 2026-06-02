# Future work

What I would build with more time, in rough priority order.

---

## Project 01: Demand Forecasting

**Foundation model fine-tuning.** TimesFM (Google), Chronos (Amazon), and
Moirai (Salesforce) are pre-trained on large time-series corpora and can
generate decent zero-shot forecasts on new series. The interesting question
is not "does the foundation model beat LightGBM?" (it often doesn't on a
well-featurised tabular model) but "when does it beat LightGBM, and for
which series types?" Cold-start series with limited history are the clear
win case.

**Demand shaping with price elasticity.** This portfolio treats price as a
feature (exogenous), not a decision variable. A complete inventory system
accounts for price-demand interaction: a markdown affects both the clearance
decision and the forecast. Estimating price elasticity and coupling it with
the reorder policy is non-trivial but high-value.

**Cross-store transfer.** Some SKUs are new to a store but have history at
other stores. A cross-store transfer model (or a global model with store
embeddings) handles cold-start better than per-store models.

**Hierarchical reconciliation.** The M5 competition winning approaches use
MinT (Minimum Trace) reconciliation to ensure SKU-level, department-level,
and store-level forecasts are internally consistent. This is a real
operational requirement: if the SKU forecasts sum to more than the store
forecast allows for, procurement plans are infeasible. Implementing MinT
was cut for time.

---

## Project 02: Returns Intelligence

**GNN for ring fraud detection.** This portfolio uses hand-engineered graph
features (degree, connected-component size, fraction of fraud-labelled
neighbours). A Graph Neural Network (GraphSAGE or GAT) would learn these
representations automatically and generalise better to novel ring structures.
The tradeoff: harder to explain, harder to debug, longer training time.
Worth building if graph features show strong lift.

**Online learning / drift monitoring.** Fraud patterns evolve adversarially.
A fraudster who gets flagged changes behaviour. A production system needs:
- Distribution shift detection (PSI or KS test on feature distributions)
- Champion-challenger setup for model comparison
- Triggered retraining when drift is detected
This is architecture, not ML, but it's the part that determines whether a
fraud model stays useful after deployment.

**NLP on open-text return reasons.** Redo's analytics product already
extracts structure from free-text return reasons ("28% cite color variance
between photos and shipped item"). A fine-tuned small LLM or a sentence
embedding model on return reason text would improve the exchange recommender
and may add signal to the fraud model (e.g. suspiciously generic return
reasons vs. specific ones).

---

## Project 03: Checkout Intent & Uplift

**X-learner and DR-learner.** This portfolio implements T-learner and
S-learner. The X-learner (Künzel et al. 2019) and Doubly Robust learner
are more statistically efficient, especially in the common case where
treatment and control groups are unequally sized.

**Real-time scoring pipeline.** The current model is batch-scored. In
production, intent scoring needs to happen within a page load: a user is
browsing the product page, and the model needs to decide within 50–100 ms
whether to show an incentive. This requires a feature store (pre-computed
session features updated in near-real-time) and a low-latency serving layer.

**Long-term outcome metrics.** Incremental conversion is the proximate
metric but not the right long-term metric. An intervention that converts
a borderline customer who later returns the item at high cost may be
net-negative. Customer LTV impact and second-return rate are harder to
measure (requires 30–90 day follow-up windows) but are the right targets.

---

## Cross-cutting infrastructure

**Feature store.** All three projects independently compute feature
pipelines. A production system would share a feature store: SKU-level
historical statistics computed once and served to all downstream models.
This eliminates duplicate computation and ensures training-serving skew
is impossible (the same feature computation runs at training and inference).

**A/B testing framework.** Each project has a concept for how it would
be A/B tested, but building the infrastructure — experiment assignment,
holdout management, power analysis, metric guardrails — is its own
substantial engineering project. The hardest part is not the statistics
but the plumbing: ensuring users stay in their assigned variant across
sessions and that metric computation is unbiased.

**Model registry and versioning.** As the number of models grows, you
need a way to track which model version is in production, what data it
was trained on, and how it performed at training time vs. in production.
MLflow or a simple versioned object store solves this.
