"""Binary HIGH alert classification against a fixed training threshold."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

RISK_LEVELS = ("NORMAL", "WARNING", "HIGH")


def risk_level_metrics(actual_levels, predicted_levels) -> dict:
    """Report one-vs-rest metrics for each risk level and a fixed 3x3 matrix."""
    actual = np.asarray(actual_levels, dtype=str)
    predicted = np.asarray(predicted_levels, dtype=str)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Actual and predicted risk levels must be nonempty and aligned")
    if not np.isin(actual, RISK_LEVELS).all() or not np.isin(predicted, RISK_LEVELS).all():
        raise ValueError("Risk levels must be NORMAL, WARNING or HIGH")
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=list(RISK_LEVELS), zero_division=0)
    return {
        "labels": list(RISK_LEVELS),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(RISK_LEVELS)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(RISK_LEVELS)).tolist(),
    }


def high_warning_metrics(actual, predicted, high_threshold: float,
                         prediction_threshold: float | None = None) -> dict:
    """Compare actual HIGH labels with alerts; thresholds may differ."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Actual and predicted values must be aligned, nonempty and finite")
    if not np.isfinite(high_threshold) or high_threshold < 0:
        raise ValueError("Invalid HIGH threshold")
    alert_threshold = high_threshold if prediction_threshold is None else float(prediction_threshold)
    if not np.isfinite(alert_threshold) or alert_threshold < 0:
        raise ValueError("Invalid prediction threshold")
    truth, alert = a >= high_threshold, p >= alert_threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, alert, labels=[True], zero_division=0)
    alert_count = int(alert.sum())
    return {"precision": float(precision[0]), "recall": float(recall[0]),
            "f1": float(f1[0]), "support": int(support[0]),
            "prediction_threshold": alert_threshold,
            "alert_count": alert_count,
            "alert_rate_percent": float(100 * alert_count / len(alert)),
            "confusion_matrix": confusion_matrix(truth, alert, labels=[False, True]).tolist()}


def calibrate_high_threshold(actual, predicted, actual_threshold: float,
                             max_alert_rate: float = .15, min_precision: float = .25) -> dict:
    """Choose a validation alert threshold maximizing recall under burden/precision limits."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Actual and predicted values must be aligned, nonempty and finite")
    if not np.isfinite(actual_threshold) or actual_threshold < 0:
        raise ValueError("Invalid actual HIGH threshold")
    if not 0 < max_alert_rate <= 1 or not 0 <= min_precision <= 1:
        raise ValueError("Invalid alert rate or precision constraint")
    truth = a >= actual_threshold
    if not truth.any():
        return {"threshold": float(actual_threshold), "method": "fallback_no_validation_high_events",
                "validation": high_warning_metrics(a, p, actual_threshold, actual_threshold),
                "constraints": {"max_alert_rate": max_alert_rate, "min_precision": min_precision}}

    candidates = np.unique(np.concatenate(([0.0], p[p >= 0])))
    feasible = []
    for threshold in candidates:
        predicted_alert = p >= threshold
        count = int(predicted_alert.sum())
        rate = count / len(p)
        true_positive = int(np.count_nonzero(truth & predicted_alert))
        precision = true_positive / count if count else 0.0
        recall = true_positive / int(truth.sum())
        if rate <= max_alert_rate and precision >= min_precision:
            feasible.append((recall, precision, -count, float(threshold)))
    if not feasible:
        chosen = float(actual_threshold)
        method = "fallback_train_quantile_no_feasible_validation_threshold"
    else:
        # Prefer recall, then precision, then fewer alerts.
        chosen = max(feasible)[3]
        method = "validation_recall_subject_to_constraints"
    return {"threshold": chosen, "method": method,
            "validation": high_warning_metrics(a, p, actual_threshold, chosen),
            "constraints": {"max_alert_rate": float(max_alert_rate), "min_precision": float(min_precision)}}
