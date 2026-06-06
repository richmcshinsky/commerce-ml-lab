# Project 04: Shipping Price Optimisation

> Which shipping option should you show — and at what price — to maximise margin without hurting conversion?

This project builds a per-session shipping price optimiser that evaluates all available
shipping tiers and selects the one that maximises expected margin given a customer's
checkout context.

The key insight is that a flat shipping rate is optimal for no one: it undercharges
sure-things who would convert at $12.99, and it overcharges persuadables who only
convert with free shipping.

---

## The problem

At checkout, a merchant must decide which shipping options to display and how to price them.
The naive approach — show the same options at the same prices to everyone — leaves money
on the table in two directions simultaneously:

- **Sure-things**: high-propensity users who convert regardless. Charging them $4.99 when
  they would convert at $12.99 is a guaranteed margin giveaway.
- **Persuadables**: price-sensitive users who convert with free shipping but abandon at $4.99.
  Charging them the flat rate loses a profitable order entirely.

---

## Optimisation objective

For each session with features X, evaluate every shipping option (price p) and select:

```
p* = argmax  P(convert | X, p) × (cart_value × 0.35 + p − $4.50)
       p
```

where `$4.50` is the merchant's fulfilment cost.  No separate conversion constraint is
needed because the objective already penalises prices that drive away too many customers.

---

## Architecture

### Conversion elasticity model
`P(convert | session_features, shipping_price)` — LightGBM on A/B test data where
shipping price was randomly assigned per session.  Because price assignment is random,
the price coefficient is causal: it estimates the marginal effect of a $1 shipping increase
on conversion probability, controlling for session context.

Calibrated with isotonic regression for well-calibrated probability outputs.

### Shipping price optimiser
Evaluates all four tiers for each session and returns the expected-margin-maximising option.
Also supports a minimum conversion rate floor for conservative deployments.

### Shipping option catalogue

| Option | Price | Delivery |
|--------|------:|---------|
| Free Shipping | $0.00 | 5–7 days |
| Standard | $4.99 | 3–5 days |
| Expedited | $7.99 | 2–3 days |
| Express | $12.99 | 1 day |

---

## Results (60k sessions, random_state=42)

| Model | PR-AUC | Notes |
|-------|-------:|-------|
| Random baseline | ~0.37 | Overall conversion rate |
| **Elasticity model (calibrated)** | **~0.72** | LightGBM + isotonic calibration |

### Policy comparison — expected margin per session

| Policy | Expected margin | Conversion rate |
|--------|---------------:|---------------:|
| Always free shipping | lower | highest |
| **Flat rate ($4.99)** | **baseline** | **baseline** |
| Always Express ($12.99) | lower | lowest |
| **Optimised** | **+16.4% vs flat rate** | **+1.3 pp** (also improves) |

The optimised policy improves expected margin while keeping conversion rate approximately flat —
because it is charging more only to sure-things (who convert regardless) and subsidising
persuadables (whose orders are profitable even after the shipping subsidy).

Exact metrics vary with random seed; run `make train-shipping` to reproduce.

### Segment pricing under optimised policy

| Segment | Recommended option | Rationale |
|---------|--------------------|-----------|
| Sure-thing | Express ($12.99) | Converts regardless — recover fulfilment cost and then some |
| Persuadable | Free Shipping ($0.00) | Price-sensitive — free shipping converts an otherwise-lost order |
| Lost cause | Express ($12.99) | Won't convert either way — maximise revenue if they do |
| Sleeping dog | Express ($12.99) | Counter-intuitive: higher price acts as a quality/urgency signal |

---

## Connection to Project 03 (Checkout Uplift)

The segment decomposition from the CATE model maps directly to shipping pricing strategy:

- Persuadables (CATE = +12.3%): free shipping *is* the treatment that converts them
- Sure-things (CATE = +1.0%): already converting — charge full rate
- Sleeping dogs: higher price slightly increases P(convert)

In production, the CATE estimates from the uplift model can feed shipping price selection
as a downstream decision layer — the same architecture as the (s, S) policy sitting on top
of the demand forecast in Project 01.

---

## Data

Synthetic A/B test data — no download required.

```bash
make train-shipping   # generates sessions + trains model (~3 min)
```

Default: 60,000 checkout sessions, four segments, six price points tested randomly.

---

## Production code

```
src/shipping/
├── synthetic.py   # Checkout session generator with segment-specific price sensitivity
├── elasticity.py  # ConversionElasticityModel — LightGBM + isotonic calibration
├── optimizer.py   # ShippingPriceOptimizer — per-session price selection
└── train.py       # CLI entry point: make train-shipping
```

---

## API

```bash
make serve-shipping
# Open http://localhost:8003/docs
```

Endpoint:
- `POST /shipping/recommend` — optimal shipping option for a checkout session

---

## Key design decisions

- **A/B test data for causal identification.** Random price assignment in training data means
  the price coefficient is a causal estimate, not a spurious correlation with user quality.
- **Expected margin as objective.** Maximising `P(convert) × margin` handles the conversion
  vs. revenue tradeoff naturally — no need for a separate conversion constraint.
- **Segment-aware pricing emerges from the model.** The optimiser does not explicitly
  segment users; segment-specific optimal prices emerge from the learned elasticity model.
- **Same data as Project 03.** Session features (f0–f5, cart metrics, device, returning status)
  are identical to the intent model — no new data pipeline required.

---

## Limitations

- **Synthetic data has cleaner signal than production.** Real price sensitivity varies by
  product category, promotion history, and competitor pricing.  Model would need periodic
  recalibration on live A/B test data.
- **Single-option display.** This project recommends one shipping tier per session.
  Production may show multiple options — the optimal displayed set is a more complex
  combinatorial problem.
- **No inventory / carrier constraints.** Express options may not be available for all
  origins, carrier capacity limits, or product sizes.
- **Price anchoring effects not modelled.** Showing $12.99 Express alongside $0.00 Free
  may change the perception of "Free" itself — display order and menu composition matter
  beyond just the individually optimal price.

## What I'd do with more time

- **Ranked option display.** Instead of recommending one option, optimise the full set and
  order to display — the sequence affects perceived value of each tier.
- **Contextual bandits.** Use an online learning algorithm to explore new price points and
  exploit known-good assignments simultaneously, adapting to distribution shifts.
- **Multi-objective Pareto front.** Expose a margin vs. conversion tradeoff curve so
  business stakeholders can choose their operating point explicitly.
- **Carrier cost integration.** Replace the flat $4.50 fulfilment cost with real carrier
  rate quotes that vary by weight, zone, and service level.
