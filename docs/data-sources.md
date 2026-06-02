# Data sources

How to reproduce all datasets used in this portfolio.

---

## M5 Forecasting (Project 01)

**Source:** Kaggle competition — [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

**Description:** Hierarchical daily sales data from Walmart across 3 states,
10 stores, and 3,049 products per store (30,490 total series). Five years of
daily sales history (2011-01-29 to 2016-06-19). Includes item prices and
calendar events (holidays, SNAP days).

**Download:**
```bash
# Requires Kaggle credentials at ~/.kaggle/kaggle.json
# Must first accept M5 competition terms at the Kaggle page above
make data-m5
```

**This portfolio uses:** Store CA_1 only (~3,049 series). This keeps the
dataset manageable on a laptop (~50 MB) while demonstrating all techniques.
The code supports loading all stores via `load_m5_sales(subset_store=None)`.

**Licence:** [Kaggle competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)

---

## Criteo Uplift v2.1 (Project 03)

**Source:** [Criteo AI Lab](https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz)

**Description:** ~13.9 million rows from a real randomised controlled
experiment. Binary treatment assignment, 12 anonymised user features, binary
outcome (visit and conversion). Designed specifically for uplift modelling
research.

**Download:**
```bash
# No authentication required
make data-criteo
```

**This portfolio uses:** 10% random sample (~1.4M rows, `sample_frac=0.1`).
Sufficient for demonstrating uplift models. Pass `sample_frac=1.0` to
`load_criteo()` for the full dataset.

**Citation:**
> Diemert, Eustache, et al. "A Large Scale Benchmark for Uplift Modeling."
> KDD'18 AdKDD Workshop, 2018.

**Licence:** [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Synthetic Returns Data (Project 02)

**Source:** Generated locally via `src/commerce_ml/data/synthetic.py`

**Description:** 100,000 customers, ~350,000 orders, ~42,000 returns.
Three planted fraud archetypes at controlled rates:
- Wardrobers: ~0.5% of customers, ~3% of returns
- Velocity returners: ~0.3% of customers, ~1.5% of returns
- Address-sharing rings: ~0.4% of customers, ~1.5% of returns

Overall fraud rate ~2% of returns.

**Generate:**
```bash
make data-synthetic
```

**Why synthetic?** No clean public returns fraud dataset exists. Generating
synthetic data with documented archetypes makes evaluation honest (known
ground truth) and avoids distributional questions about proprietary data.
The generation code is documented and deterministic (`random_state=42`).
