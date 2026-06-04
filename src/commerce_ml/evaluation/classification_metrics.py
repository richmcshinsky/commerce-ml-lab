"""Classification evaluation metrics for fraud detection and intent models.

Implements business-relevant metrics beyond plain accuracy:

- **PR-AUC** (area under precision-recall curve) — preferred over ROC-AUC for
  imbalanced classes (fraud is ~2% of returns; ROC-AUC is overly optimistic).
- **Precision @ K** — what fraction of the top-K flagged returns are truly
  fraudulent? Directly maps to analyst workload.
- **Recall @ FPR** — coverage at a controlled false positive rate.
- **Cost-aware threshold** — selects the operating threshold that minimises
  expected business cost given a stated FP/FN cost ratio.
- **Qini coefficient** — uplift model evaluation metric (incremental conversions
  above a random targeting baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve


def pr_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Area under the precision-recall curve.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0/1).
    y_score:
        Predicted probability scores in [0, 1].

    Returns
    -------
    float
        PR-AUC in [0, 1]. A random classifier scores approximately
        ``prevalence`` (positive class rate).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def precision_at_k(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    k: int,
) -> float:
    """Precision among the top-K highest-scored instances.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Predicted probability scores.
    k:
        Number of top-scored instances to consider.

    Returns
    -------
    float
        Fraction of top-K instances that are true positives.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def recall_at_fpr(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    target_fpr: float = 0.01,
) -> float:
    """Recall achieved at a target false positive rate.

    Finds the highest recall achievable while keeping FPR <= ``target_fpr``.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Predicted probability scores.
    target_fpr:
        Maximum acceptable false positive rate (default 0.01 = 1%).

    Returns
    -------
    float
        Recall at the operating threshold that satisfies the FPR constraint.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = tpr[fpr <= target_fpr]
    return float(eligible.max()) if len(eligible) > 0 else 0.0


def cost_aware_threshold(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> tuple[float, float]:
    """Select the probability threshold that minimises expected business cost.

    At Redo's scale, flagging a legitimate return as fraud (FP) costs customer
    goodwill and support time; missing a fraudulent return (FN) costs the full
    return value. The ratio ``cost_fn / cost_fp`` determines how aggressively
    to flag.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Predicted probability scores.
    cost_fp:
        Business cost of one false positive (unnecessary flag).
    cost_fn:
        Business cost of one false negative (missed fraud).

    Returns
    -------
    threshold : float
        Optimal probability threshold.
    expected_cost : float
        Total expected cost at the optimal threshold (normalised by N).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []

    for t in thresholds:
        predictions = (y_score >= t).astype(float)
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        costs.append(cost_fp * fp + cost_fn * fn)

    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx]) / len(y_true)


def qini_coefficient(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_score: np.ndarray | pd.Series,
) -> float:
    """Qini coefficient for uplift model evaluation.

    Measures the area between the Qini curve (cumulative incremental
    conversions when targeting by uplift score) and the random-targeting
    baseline.

    Parameters
    ----------
    y_true:
        Binary outcome labels (e.g. conversion: 0/1).
    treatment:
        Binary treatment assignment (1 = treated, 0 = control).
    uplift_score:
        Predicted individual treatment effect (ITE) / uplift score.

    Returns
    -------
    float
        Qini coefficient. Higher is better; 0 = same as random targeting.
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "w": np.asarray(treatment, dtype=float),
        "score": np.asarray(uplift_score, dtype=float),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    n = len(df)
    n_treated = df["w"].sum()
    n_control = n - n_treated

    cumulative_treated = (df["y"] * df["w"]).cumsum()
    cumulative_control = (df["y"] * (1 - df["w"])).cumsum()
    fraction_targeted = np.arange(1, n + 1) / n

    # Incremental conversions above random
    incremental = cumulative_treated - cumulative_control * (n_treated / max(n_control, 1))
    random_line = incremental.iloc[-1] * fraction_targeted

    qini_area = float(np.trapz(incremental, fraction_targeted))
    random_area = float(np.trapz(random_line, fraction_targeted))

    return qini_area - random_area
