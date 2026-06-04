"""Dataset download and loading utilities.

Provides loaders for:
- M5 Forecasting dataset (Walmart hierarchical daily sales, Kaggle)
- Criteo Uplift dataset (treatment/control conversion labels, free)

Usage
-----
Download all datasets::

    python -m commerce_ml.data.loaders download-all

Or via Makefile::

    make data-m5
    make data-criteo
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Root data directory — all datasets land here (gitignored).
# loaders.py lives at src/commerce_ml/data/loaders.py, so parents[3] is the repo root.
DATA_DIR = Path(__file__).parents[3] / "data"

M5_DIR = DATA_DIR / "m5"
CRITEO_DIR = DATA_DIR / "criteo"


def get_data_dir() -> Path:
    """Return the root data directory, creating it if it does not exist.

    Returns
    -------
    Path
        Path to the project-level ``data/`` directory (``<repo_root>/data/``).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR

# Kaggle competition slug for M5
_M5_COMPETITION = "m5-forecasting-accuracy"

# Criteo uplift dataset public URL (no auth required)
_CRITEO_URL = (
    "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
)


# ── M5 ───────────────────────────────────────────────────────────────────────


def download_m5(dest: Path = M5_DIR) -> None:
    """Download the M5 Forecasting competition files from Kaggle.

    Requires ``~/.kaggle/kaggle.json`` with valid credentials and acceptance
    of the M5 competition terms at
    https://www.kaggle.com/competitions/m5-forecasting-accuracy.

    Parameters
    ----------
    dest:
        Directory to extract files into. Created if it does not exist.
    """
    try:
        import kaggle  # noqa: F401 — triggers credential check
    except ImportError as e:
        raise ImportError("Install kaggle: `uv add kaggle`") from e

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading M5 from Kaggle -> %s", dest)

    import subprocess

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            _M5_COMPETITION,
            "-p",
            str(dest),
        ],
        check=True,
    )

    zip_path = dest / f"{_M5_COMPETITION}.zip"
    if zip_path.exists():
        logger.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()

    logger.info("M5 download complete: %s", dest)


def load_m5_sales(data_dir: Path = M5_DIR, subset_store: str | None = "CA_1") -> pd.DataFrame:
    """Load M5 daily sales data, optionally filtered to one store.

    The raw M5 dataset has 30,490 series (SKU x store). Filtering to a single
    store reduces this to ~3,000 series — enough to demonstrate all techniques
    while keeping memory manageable on a laptop.

    Parameters
    ----------
    data_dir:
        Directory containing the extracted M5 CSV files.
    subset_store:
        If provided, return only series belonging to this store ID
        (e.g. ``"CA_1"``). Pass ``None`` to load all stores (~60M rows).

    Returns
    -------
    pd.DataFrame
        Wide-format sales DataFrame with columns ``id``, ``item_id``,
        ``dept_id``, ``cat_id``, ``store_id``, ``state_id``, and one column
        per day (``d_1`` … ``d_1913``).

    Raises
    ------
    FileNotFoundError
        If the M5 files have not been downloaded yet (run ``make data-m5``).
    """
    sales_path = data_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        raise FileNotFoundError(
            f"M5 data not found at {sales_path}. Run `make data-m5` first."
        )

    df = pd.read_csv(sales_path)

    if subset_store is not None:
        df = df[df["store_id"] == subset_store].reset_index(drop=True)
        logger.info("Loaded M5 store=%s: %d series", subset_store, len(df))
    else:
        logger.info("Loaded full M5: %d series", len(df))

    return df


def load_m5_calendar(data_dir: Path = M5_DIR) -> pd.DataFrame:
    """Load the M5 calendar file (dates, events, SNAP flags).

    Parameters
    ----------
    data_dir:
        Directory containing the extracted M5 CSV files.

    Returns
    -------
    pd.DataFrame
        Calendar with columns: ``date``, ``wm_yr_wk``, ``weekday``, ``wday``,
        ``month``, ``year``, ``d`` (day column name), ``event_name_1``,
        ``event_type_1``, ``event_name_2``, ``event_type_2``,
        ``snap_CA``, ``snap_TX``, ``snap_WI``.
    """
    cal_path = data_dir / "calendar.csv"
    if not cal_path.exists():
        raise FileNotFoundError(f"M5 calendar not found at {cal_path}. Run `make data-m5`.")

    df = pd.read_csv(cal_path, parse_dates=["date"])
    return df


def load_m5_prices(data_dir: Path = M5_DIR) -> pd.DataFrame:
    """Load the M5 weekly sell prices file.

    Parameters
    ----------
    data_dir:
        Directory containing the extracted M5 CSV files.

    Returns
    -------
    pd.DataFrame
        Prices with columns: ``store_id``, ``item_id``, ``wm_yr_wk``,
        ``sell_price``.
    """
    prices_path = data_dir / "sell_prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"M5 prices not found at {prices_path}. Run `make data-m5`.")

    return pd.read_csv(prices_path)


# ── Criteo ───────────────────────────────────────────────────────────────────


def download_criteo(dest: Path = CRITEO_DIR) -> None:
    """Download the Criteo Uplift v2.1 dataset.

    No authentication required. File is ~700 MB compressed.

    Parameters
    ----------
    dest:
        Directory to save the file into. Created if it does not exist.
    """
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "criteo_uplift_v2.1.csv.gz"

    if out_path.exists():
        logger.info("Criteo file already exists: %s", out_path)
        return

    logger.info("Downloading Criteo uplift dataset -> %s", out_path)
    urllib.request.urlretrieve(_CRITEO_URL, out_path)
    logger.info("Criteo download complete: %s", out_path)


def load_criteo(data_dir: Path = CRITEO_DIR, sample_frac: float = 0.1) -> pd.DataFrame:
    """Load the Criteo Uplift dataset.

    The full dataset has ~13.9M rows. ``sample_frac=0.1`` gives ~1.4M rows,
    which is sufficient for demonstrating uplift models while fitting in RAM.

    Parameters
    ----------
    data_dir:
        Directory containing ``criteo_uplift_v2.1.csv.gz``.
    sample_frac:
        Fraction of rows to load (default 0.1 = 10%). Pass 1.0 for full data.

    Returns
    -------
    pd.DataFrame
        Columns: ``f0``…``f11`` (anonymised features), ``treatment``
        (0/1 binary), ``exposure`` (1 if user was exposed to treatment),
        ``visit`` (site visit within 2 weeks), ``conversion`` (purchase).

    Raises
    ------
    FileNotFoundError
        If the Criteo file has not been downloaded yet (run ``make data-criteo``).
    """
    gz_path = data_dir / "criteo_uplift_v2.1.csv.gz"
    if not gz_path.exists():
        raise FileNotFoundError(
            f"Criteo data not found at {gz_path}. Run `make data-criteo` first."
        )

    df = pd.read_csv(gz_path, compression="gzip")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        logger.info("Loaded Criteo sample (%.0f%%): %d rows", sample_frac * 100, len(df))
    else:
        logger.info("Loaded full Criteo: %d rows", len(df))

    return df


# ── CLI entry point ───────────────────────────────────────────────────────────


def _cli() -> None:
    """Simple CLI for ``python -m commerce_ml.data.loaders``."""
    import sys

    commands = {"download-m5", "download-criteo", "download-all"}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python -m commerce_ml.data.loaders [{' | '.join(sorted(commands))}]")
        sys.exit(1)

    cmd = sys.argv[1]
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if cmd in ("download-m5", "download-all"):
        download_m5()
    if cmd in ("download-criteo", "download-all"):
        download_criteo()


if __name__ == "__main__":
    _cli()


def generate_criteo_like(
    n_rows: int = 200_000,
    treatment_rate: float = 0.50,
    base_conversion_rate: float = 0.04,
    random_state: int = 42,
) -> "pd.DataFrame":
    """Generate a synthetic Criteo-like uplift dataset.

    Produces the same schema as ``load_criteo()`` with planted CATE
    heterogeneity across four segments:

    - **Persuadables** (~25%): low control conversion, high treatment lift.
    - **Sure-things** (~25%): high conversion regardless of treatment.
    - **Lost causes** (~25%): low conversion regardless of treatment.
    - **Sleeping dogs** (~25%): slightly *hurt* by treatment.

    Key design: propensity-ranked users are NOT the same as uplift-ranked
    users — the sure-things segment has the highest propensity but near-zero
    uplift, which is the core demonstration of this project.

    Parameters
    ----------
    n_rows:
        Number of observations to generate.
    treatment_rate:
        Fraction assigned to treatment (RCT-style).
    base_conversion_rate:
        Overall mean conversion rate (controls scale of logistic intercept).
    random_state:
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: f0…f11, treatment, conversion, visit, exposure, segment.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(random_state)

    n = n_rows
    segments = rng.choice(
        ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"],
        size=n,
        p=[0.25, 0.25, 0.25, 0.25],
    )

    # f0-f11: 12 anonymous features with different roles
    # f0-f3: correlated with propensity (sure-things have high f0)
    # f4-f7: correlated with treatment effect (persuadables have high f4)
    # f8-f11: noise features
    F = rng.standard_normal((n, 12))

    # Plant segment-specific signal
    is_persuadable = segments == "persuadable"
    is_sure_thing  = segments == "sure_thing"
    is_lost_cause  = segments == "lost_cause"
    is_sleeping    = segments == "sleeping_dog"

    F[is_sure_thing,  0] += 1.5   # high propensity signal
    F[is_sure_thing,  1] += 1.0
    F[is_persuadable, 4] += 1.5   # high treatment-response signal
    F[is_persuadable, 5] += 1.0
    F[is_lost_cause,  0] -= 1.5   # low propensity
    F[is_sleeping,    0] -= 0.5
    F[is_sleeping,    6] += 1.0   # sleeping-dog signal

    feature_cols = [f"f{i}" for i in range(12)]
    df = pd.DataFrame(F, columns=feature_cols)

    # Treatment assignment: RCT (random 50/50)
    df["treatment"] = (rng.random(n) < treatment_rate).astype(int)

    # Potential outcomes (logistic)
    def sigmoid(x: "np.ndarray") -> "np.ndarray":
        return 1.0 / (1.0 + np.exp(-x))

    intercept = float(np.log(base_conversion_rate / (1 - base_conversion_rate)))

    # Base logit (propensity driver)
    logit_base = intercept + 0.6 * F[:, 0] + 0.4 * F[:, 1] + 0.2 * F[:, 2]

    # Treatment effect (CATE) by segment
    tau = np.zeros(n)
    tau[is_persuadable] =  0.8 + 0.4 * F[is_persuadable, 4]   # high positive uplift
    tau[is_sure_thing]  =  0.05                                 # near-zero uplift
    tau[is_lost_cause]  = -0.05                                 # near-zero (slightly neg)
    tau[is_sleeping]    = -0.5 - 0.2 * F[is_sleeping, 6]       # negative uplift (hurt by treatment)

    p_control  = sigmoid(logit_base)
    p_treated  = sigmoid(logit_base + tau)

    # Observed outcome
    p_observed = np.where(df["treatment"] == 1, p_treated, p_control)
    df["conversion"] = (rng.random(n) < p_observed).astype(int)
    df["visit"]      = (df["conversion"] == 1) | (rng.random(n) < 0.15)
    df["visit"]      = df["visit"].astype(int)
    df["exposure"]   = 1  # everyone was exposed (RCT)
    df["segment"]    = segments  # ground-truth for evaluation

    logger.info(
        "Generated synthetic Criteo-like dataset: %d rows, "
        "treatment_rate=%.1f%%, conversion_rate=%.2f%%",
        n, treatment_rate * 100,
        df["conversion"].mean() * 100,
    )
    return df
