"""SHAP attribution for the raw XGBoost regression output only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.pipeline.dataset import feature_matrix


def explain_xgboost(model, frame: pd.DataFrame, max_samples: int = 100) -> dict:
    """Return global and local machine-readable SHAP values in model feature order.

    Attributions explain model association, not causation. Additivity concerns
    raw regression output, before the nonnegative forecast clip.
    """
    if model.name != "xgboost" or model.estimator is None:
        raise ValueError("A fitted XGBoostModel is required")
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP requires 'shap'; install requirements.txt") from exc
    x = feature_matrix(frame.iloc[:max_samples], model.features)
    if x.empty:
        raise ValueError("SHAP needs at least one observation")
    explanation = shap.TreeExplainer(model.estimator)(x)
    values = np.asarray(explanation.values, dtype=float)
    bases = np.broadcast_to(np.asarray(explanation.base_values, dtype=float), (len(x),))
    raw = np.asarray(model.estimator.predict(x), dtype=float)
    if values.shape != x.shape or not np.allclose(bases + values.sum(axis=1), raw, rtol=1e-3, atol=1e-3):
        raise ValueError("SHAP values do not add to raw XGBoost output")
    importance = sorted(({"feature": name, "mean_abs_shap": float(value)}
        for name, value in zip(model.features, np.abs(values).mean(axis=0))),
        key=lambda row: row["mean_abs_shap"], reverse=True)
    local = []
    for i in range(len(x)):
        local.append({"date": str(pd.Timestamp(frame.iloc[i]["date"]).date()),
            "base_value": float(bases[i]), "raw_model_output": float(raw[i]),
            "clipped_forecast": float(max(0, raw[i])),
            "features": {name: float(x.iloc[i][name]) for name in model.features},
            "shap_values": {name: float(values[i, j]) for j, name in enumerate(model.features)}})
    return {"unit": model.unit, "global_importance": importance, "local": local,
            "note": "SHAP explains raw model output; attributions are not causal."}
