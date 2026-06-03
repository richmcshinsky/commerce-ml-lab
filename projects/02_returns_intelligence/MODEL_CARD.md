# Model Card — Returns Intelligence Suite

**Version**: 1.0  
**Date**: June 2026  
**Author**: Richard McShinsky  
**Project**: commerce-ml-lab / Project 02

This card covers the three models in the returns intelligence system as a unit.
They share a training dataset and are designed to operate as a pipeline.

---

## System overview

| Model | Class | Scored at | Primary output |
|-------|-------|-----------|----------------|
| A — Return likelihood | `ReturnLikelihoodModel` | Order fulfillment | P(return), risk tier |
| B — Fraud detection | `FraudDetectionModel` | Return initiation | P(fraud), is_flagged, top reasons |
| C — Exchange recommendation | `ExchangeRecommender` | Return approval | Ranked exchange candidates |

---

## Model A: Return Likelihood

### Intended use
Score the probability that a placed order will eventually be returned.
Use for: pre-shipment quality routing, weekly return volume forecasting,
and as an input feature to Model B.

**Out of scope**: not suitable for denying orders or penalising customers.
High return probability is not evidence of fraud — many legitimate customers
return frequently (apparel sizing, gift purchases).

### Training data
Synthetic e-commerce returns dataset: 20,000 customers, ~70,000 orders.
Features available at fulfillment time: category, item price, quantity,
channel (web/mobile/marketplace), account age, customer lifetime return rate.

### Architecture
LightGBM binary classifier (`objective="binary"`) with isotonic probability
calibration on a 20% held-out calibration set.

### Performance (20% temporal test split)
| Metric | Value | Baseline |
|--------|-------|----------|
| PR-AUC | ~0.55–0.70 | ~0.14 (return rate) |
| Calibration | Isotonic regression; max bin deviation < 5pp | Uncalibrated LGB |

### Threshold / operating point
Risk tiers are applied post-hoc. No single classification threshold is used;
downstream consumers apply their own cost-based cutoffs.

| Tier | P(return) | Suggested action |
|------|-----------|-----------------|
| Low | < 10% | Standard fulfillment |
| Medium | 10–25% | Optional quality check |
| High | > 25% | Enhanced inspection, consider hold |

---

## Model B: Fraud Detection

### Intended use
Score return events for potential fraud or abuse at the point of return
initiation. Flag high-risk returns for analyst review before issuing refund.

**Out of scope**: automated refund denial. The output is a flag for human
review, not an automated decision. All flagged returns should be reviewed
by a trained analyst before any customer-facing action is taken.

### Training data
Same synthetic dataset. Features available at return time: days to return,
return condition, reason code, item price and category, customer history
(total orders, total returns, lifetime return rate, account age), plus
graph features derived from the customer-address/payment network.

### Architecture
1. **Graph feature extraction** (networkx): bipartite customer ↔ address
   and customer ↔ payment graphs. Extracted features: `shared_address_count`,
   `shared_payment_count`, `component_size`.
2. **LightGBM classifier** on 14 behavioural + graph features.
   `scale_pos_weight=10` to correct class imbalance (~3–4% fraud rate).
3. **Isotonic calibration** on held-out validation set.
4. **Cost-aware threshold**: auto-selected to minimise
   `FP_cost × FP_rate + FN_cost × FN_rate` with default FP_cost=5, FN_cost=50.

### Performance (20% temporal test split)
| Metric | Value | Baseline |
|--------|-------|----------|
| PR-AUC | ~0.55–0.70 | ~0.03–0.04 (fraud rate) |
| Precision@50 | ~40–60% | ~3–4% |
| Operating threshold | ~0.15–0.25 | Default 0.50 |

### Fraud archetypes detected
| Archetype | Key features | Detection rate |
|-----------|-------------|----------------|
| Wardrober | used/damaged condition, expensive category, late return | High |
| Velocity returner | many recent orders, fast return, high return rate | High |
| Address-sharing ring | component_size > 1, shared_address_count > 0 | Medium–High |

### SHAP reason codes
Every flagged return includes the top-2 features driving the score. Used
for analyst review prioritisation and customer communication.

---

## Model C: Exchange Recommendation

### Intended use
Recommend exchange alternatives to customers initiating a return, with the
goal of retaining the sale. Presented after Model B has cleared the return
as non-fraudulent.

**Out of scope**: do not show exchange recommendations to fraud-flagged returns.

### Architecture
Two-stage pipeline:

**Stage 1 — Heuristic filter** (reason-code rules, ~70% coverage):
- `too_small` → same style, next size up
- `too_large` → same style, next size down
- `wrong_color` → same base SKU, all available colors
- `defective` → exact replacement (same item_id)
- `changed_mind` / `not_as_described` → same category, top-N by popularity

**Stage 2 — Feature scoring** (for all candidates):
Score = 0.5 × popularity_norm + 0.3 × price_proximity + 0.2 × rank_score

### Performance
No ground-truth conversion labels in the training data (synthetic). In production,
acceptance rate (customer accepts exchange offer) would be the primary metric.
Proxy evaluation: top-1 candidate score distribution and rule coverage.

| Metric | Value |
|--------|-------|
| Heuristic rule coverage | ~70% of reason codes |
| Catalog size (20K dataset) | ~1,500 items |
| Mean top-1 score | ~0.85 |

---

## Ethical considerations

### Fairness
The fraud model uses behavioural features that may correlate with demographics.
Specifically:
- **Account age**: newer accounts score higher. New immigrants or people who
  recently switched retailers may be disproportionately flagged.
- **Address sharing**: legitimate co-habitants (roommates, families) may share
  addresses without fraudulent intent.

**Mitigations**: (1) human review required before any customer action; (2) appeals
process for flagged customers; (3) regular bias audits across demographic proxies
if customer demographic data becomes available.

### Transparency
All fraud flags include SHAP reason codes in plain English. Customers who are
denied a refund have the right to know the reason. The system is designed so
analysts can explain every flag to a customer.

### Data governance
- Synthetic data only — no real customer PII used in this portfolio.
- In production: customer graph data (shared addresses) must be handled under
  applicable privacy law (GDPR, CCPA). Address sharing should not be stored
  beyond the fraud detection use case without explicit consent.

---

## Deployment requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| LightGBM | 4.3+ |
| networkx | 3.3+ (for graph features) |
| scikit-learn | 1.5+ (for isotonic calibration) |
| Inference latency | Model A: < 10ms; Model B: 50–200ms (graph build); Model C: < 5ms |
| Graph pre-computation | Customer address/payment graph must be rebuilt on a nightly schedule for Model B |

### Retraining schedule
- **Model A**: monthly, or when return rate shifts > 2pp
- **Model B**: bi-weekly, or when fraud rate shifts > 0.5pp; wardrobers adapt patterns
- **Model C**: quarterly, or when catalog changes significantly

---

## Known limitations

1. **Synthetic training data** has cleaner signal separation than production data.
   Real wardrobers and velocity returners adapt their behaviour over time.
2. **Graph features require pre-computation** at inference time. Computing
   connected components on a live, growing graph requires a dedicated graph
   database (e.g. Neptune) or nightly batch jobs — not implemented here.
3. **No temporal graph features**. The model uses a static snapshot of address/payment
   sharing. New ring members joining an existing cluster after training will not
   be correctly scored until the next retraining.
4. **Exchange recommender has no feedback loop**. Without accepted/rejected exchange
   labels, the scoring weights are heuristic. In production, a bandit or A/B test
   would tune the weights toward actual acceptance rates.
5. **Label noise**. In production, fraud labels are confirmed only after investigation.
   Unconfirmed fraud (likely positives that weren't investigated) are treated as
   negatives during training, biasing the model toward under-detection.
