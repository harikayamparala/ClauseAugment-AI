"""
Model Evaluation Metrics
========================
Computes and returns evaluation metrics for trained classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list, classes: list = None) -> dict:
    """
    Compute classification metrics.

    Returns
    -------
    dict with:
        accuracy, precision, recall, f1,
        confusion_matrix (ndarray), classes (list),
        classification_report (str)
    """
    if not y_true or not y_pred:
        return _empty_metrics()

    # Determine label order
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    # Averaging strategy: macro (treats all classes equally)
    avg = "macro"
    zero_div = 0  # avoid division warnings

    try:
        acc  = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average=avg, zero_division=zero_div))
        rec  = float(recall_score(y_true, y_pred, average=avg, zero_division=zero_div))
        f1   = float(f1_score(y_true, y_pred, average=avg, zero_division=zero_div))
        cm   = confusion_matrix(y_true, y_pred, labels=classes)
        report = classification_report(y_true, y_pred, labels=classes, zero_division=zero_div)
    except Exception as e:
        return {**_empty_metrics(), "error": str(e)}

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "confusion_matrix": cm,
        "classes": classes,
        "classification_report": report,
        "error": None,
    }


def compare_metrics(before: dict, after: dict) -> dict:
    """
    Compute deltas between two metric sets.

    Parameters
    ----------
    before : metrics dict from original dataset
    after  : metrics dict from augmented dataset

    Returns
    -------
    dict of delta values and improvement flags.
    """
    keys = ["accuracy", "precision", "recall", "f1"]
    result = {}
    for k in keys:
        b = before.get(k, 0.0) or 0.0
        a = after.get(k, 0.0) or 0.0
        delta = round(a - b, 4)
        result[k] = {
            "before": round(b, 4),
            "after": round(a, 4),
            "delta": delta,
            "improved": delta > 0,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    return {
        "accuracy":  0.0,
        "precision": 0.0,
        "recall":    0.0,
        "f1":        0.0,
        "confusion_matrix": np.zeros((1, 1), dtype=int),
        "classes": [],
        "classification_report": "",
        "error": None,
    }


def metrics_to_df(metrics: dict) -> pd.DataFrame:
    """Convert metric dict to a formatted DataFrame for display."""
    rows = [
        {"Metric": "Accuracy",  "Score": f"{metrics['accuracy']:.2%}"},
        {"Metric": "Precision", "Score": f"{metrics['precision']:.2%}"},
        {"Metric": "Recall",    "Score": f"{metrics['recall']:.2%}"},
        {"Metric": "F1-Score",  "Score": f"{metrics['f1']:.2%}"},
    ]
    return pd.DataFrame(rows)
