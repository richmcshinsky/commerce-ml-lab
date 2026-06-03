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

## Results (synthetic data, 200k rows)

### Model comparison — test set (20%), Qini coefficient

| Model | Qini | Uplift@20% | What it targets |
|-------|-----:|-----------:|-----------------|
| **T-learner uplift** | **highest** | highest | Persuadables (correct) |
| **S-learner uplift** | high | high | Persuadables (correct) |
| Propensity (wrong) | lower | lower | Sure-things (wastes budget) |
| Random | ~0 | ~ATE | Uniform across segments |

T-learner and S-learner both correctly prioritise persuadables in the top-20%
targeting bucket. The propensity model fills its top-20% mostly with sure-things.

### Segment composition of top-20% targeting

| Segment | CATE | Propensity | T-learner top-20% | Propensity top-20% |
|---------|-----:|-----------:|------------------:|-------------------:|
| Persuadable | +0.12 | medium | high ✓ | low ✗ |
| Sure-thing | +0.02 | **highest** | low ✓ | **high ✗** |
| Lost cause | ~0 | low | low ✓ | medium |
| Sleeping dog | -0.02 | low | low ✓ | low |

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
