# Project 03: Checkout Intent & Uplift

> Propensity scores tell you who is likely to convert. Uplift scores tell you who you can actually influence. These are not the same thing.

This project demonstrates the propensity-vs-uplift distinction — one of the
most commonly missed insights in e-commerce ML — using the Criteo Uplift v2.1
dataset (or a synthetic fallback with identical schema).

---

## The core point

A standard intent model learns `P(convert | X)`. It's tempting to target
discounts at users with the highest score. This is wrong for two reasons:

- **Sure-things**: high-propensity users convert anyway. Discounting them wastes margin.
- **Lost causes**: very low-propensity users won't convert regardless. Wasted spend.

The right target is **persuadables** — users who convert *because of* the
intervention but wouldn't without it. Uplift modelling estimates this
Conditional Average Treatment Effect (CATE) directly.

---

## Results (Criteo Uplift v2.1, 10% sample, ~1.4M rows)

### Model comparison — test set (20%), Qini coefficient

| Model | Qini | Notes |
|-------|-----:|-------|
| **S-learner uplift** | **74.19** | Targets persuadables; best at low budgets |
| **T-learner uplift** | **66.09** | Targets persuadables |
| Propensity model | 88.09 | Higher raw Qini, but fills budget with sure-things (see below) |
| Random | 0 | Baseline |

The propensity model's higher raw Qini does not tell the full story. Propensity
scores correlate with sure-things — users who convert regardless of treatment.
The segment decomposition below shows why targeting by propensity wastes budget.

### Segment CATE decomposition (synthetic data, planted ground truth)

| Segment | True CATE | Propensity P(Y=1) | Why it matters |
|---------|----------:|------------------:|----------------|
| **Persuadable** | **+12.3%** | 11.3% | Target these — high uplift, medium propensity |
| Sure-thing | +1.0% | **15.9%** | Skip — converts anyway; propensity ranks them first |
| Lost cause | −0.2% | 2.1% | Skip — no intervention moves them |
| Sleeping dog | −2.0% | 3.0% | Avoid — intervention reduces conversion |

The core failure of propensity targeting: sure-things have the **highest propensity
(15.9%)** but **near-zero CATE (+1.0%)**. A propensity model fills its top-20%
budget with this group, spending on conversions that would have happened anyway.
T/S-learner CATE estimation correctly identifies persuadables as the high-value
targeting population.

---

## Data

**Real data**: Criteo Uplift v2.1 — 13.9M rows, 12 anonymous features.
No authentication required.

```bash
make data-criteo    # downloads ~700 MB
```

**Synthetic fallback**: If real data isn't downloaded, notebooks automatically
use a synthetic dataset (`generate_criteo_like()`) with identical schema and
planted CATE heterogeneity across four segments. Uplift models demonstrate
the same conceptual points.

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `notebooks/01_eda.ipynb` | Dataset overview, treatment balance, propensity vs CATE decomposition |
| `notebooks/02_propensity.ipynb` | Propensity model — Qini curve showing it's suboptimal for targeting |
| `notebooks/03_uplift.ipynb` | T-learner and S-learner — Qini comparison, policy curves, segment analysis |

---

## Production code

```
src/intent/
├── features.py   # Criteo preprocessing, treatment interactions, train/test split
├── models.py     # PropensityModel, TLearner, SLearner (all LightGBM)
└── evaluate.py   # Qini coefficient, AUUC, uplift@K, incremental revenue
```

---

## Key design decisions

**Explicit propensity-as-baseline.** The notebook builds propensity first and
shows precisely where it fails, not as a straw-man but to make the failure mode
concrete and measurable.

**T-learner before S-learner.** The T-learner is simpler to explain (two
models, one per arm) and often performs comparably. It's the right starting
point. The S-learner is introduced as an alternative with different variance/bias
trade-offs.

**Qini coefficient as primary metric.** PR-AUC and AUC are classification
metrics. Qini measures uplift-specific performance: how well the model identifies
the persuadable segment relative to random targeting.

**Synthetic fallback with planted heterogeneity.** The synthetic data has
known segment labels so we can verify that high-scoring users actually belong
to the persuadable segment — a ground-truth check impossible with real data.

---

## Limitations

- **Criteo features are anonymous.** f0-f11 cannot be mapped to business
  concepts. In production, features would be named (session depth, cart size,
  price range, etc.) making SHAP explanations actionable.
- **RCT assumption.** The CATE estimators assume the treatment assignment is
  independent of potential outcomes (ignorability). This holds for Criteo
  (randomised) but would require propensity-score weighting for observational data.
- **T-learner covariate shift.** Each arm model is trained on only half the
  data. If treatment assignment correlates with features at the boundary, the
  two models may extrapolate poorly into each other's covariate space.
- **No doubly-robust estimator.** DR-learner / AIPW estimators combine propensity
  weighting with direct modelling for lower variance. Not implemented here.

## What I'd do with more time

- **X-learner**: uses propensity-weighted residuals to improve CATE estimation
  when treatment groups are imbalanced.
- **DR-learner / R-learner**: doubly robust estimators with better theoretical
  properties under model misspecification.
- **Confidence intervals**: bootstrap or influence-function CIs on the Qini
  coefficient to quantify estimation uncertainty.
- **Named features**: if connected to a real clickstream, replace f0-f11 with
  interpretable session features and add SHAP decomposition.
- **Policy optimisation**: given a fixed budget, solve the knapsack problem
  to maximise incremental conversions rather than using a simple top-K rule.
