# Project 03: Checkout Intent & Uplift

> Propensity scores tell you who is likely to convert. Uplift scores tell you who you can actually influence. These are not the same thing.

This project demonstrates the propensity-vs-uplift distinction — one of the
most commonly missed insights in e-commerce ML — using the Criteo Uplift v2.1
dataset.

---

## The core point

A standard intent model learns `P(convert | session features)`. It's tempting
to use this to target discounts or pop-ups: "show an incentive to users with
score < 0.3."

This is wrong for two reasons:
1. **Sure-things**: high-propensity users would have converted anyway.
   Discounting them wastes margin.
2. **Lost-causes**: very low-propensity users won't convert regardless of the
   intervention. Spending on them is wasted.

The right target is the **persuadable** segment: users who convert *because of*
the intervention but wouldn't without it. Uplift modelling estimates this
individual treatment effect directly.

---

## Results

*(To be filled in after training)*

| Model | Qini Coefficient | Notes |
|-------|-----------------|-------|
| Random targeting | 0.0 (baseline) | |
| Propensity-based (wrong approach) | — | Shown for comparison |
| T-learner uplift | — | |
| S-learner uplift | — | |

Incremental revenue comparison: propensity-targeting vs. uplift-targeting at
top-20% of population. *(To be filled in)*

---

## Data

Criteo Uplift v2.1 — 13.9M rows, 10% sample used (~1.4M rows).
See [docs/data-sources.md](../../docs/data-sources.md).

```bash
make data-criteo
```

---

## Notebooks (run by hand)

| Notebook | Contents |
|----------|----------|
| `notebooks/01_eda.ipynb` | Dataset overview, treatment/control balance, feature distributions |
| `notebooks/02_propensity.ipynb` | Standard propensity model — the "naive" approach |
| `notebooks/03_uplift.ipynb` | T-learner and S-learner uplift models, Qini curves, policy comparison |

---

## Production code

```
src/intent/
├── features.py  # Session feature engineering (shared + Criteo-specific)
├── models.py    # Propensity classifier, T-learner, S-learner
└── evaluate.py  # Qini coefficient, uplift@K, incremental revenue
```

---

## Key design decisions

- **Explicit propensity-as-baseline.** The notebook builds the propensity
  model first and shows exactly where it fails — not as a straw-man but as
  the approach that most teams actually use.
- **T-learner and S-learner.** Both are implemented and compared. T-learner is
  more flexible; S-learner shares information across treatment arms (better
  in small data).
- **Qini curves, not AUC.** Qini measures incremental conversions above random
  targeting, which is the right question for intervention ROI.

---

## Limitations

- Criteo features are anonymised (`f0`…`f11`). In production, session
  features (pages viewed, time-on-site, cart size) are interpretable and
  can be engineered specifically.
- The treatment in Criteo is binary (exposed/not). Real interventions have
  multiple treatment arms (different discount levels, different message
  types) requiring multi-treatment uplift models.
- The dataset measures short-term conversion. The business-relevant metric
  for interventions that increase conversion but also increase returns is
  customer LTV impact, which requires 30–90 day follow-up.
