"""Binary HIGH alert classification against a fixed training threshold."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def high_warning_metrics(actual, predicted, high_threshold: float) -> dict:
    """Report HIGH precision, recall, F1, support and [TN,FP;FN,TP]."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Actual and predicted values must be aligned, nonempty and finite")
    if not np.isfinite(high_threshold) or high_threshold < 0:
        raise ValueError("Invalid HIGH threshold")
    truth, alert = a >= high_threshold, p >= high_threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, alert, labels=[True], zero_division=0)
    return {"precision": float(precision[0]), "recall": float(recall[0]),
            "f1": float(f1[0]), "support": int(support[0]),
            "confusion_matrix": confusion_matrix(truth, alert, labels=[False, True]).tolist()}
