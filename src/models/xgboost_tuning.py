"""Small deterministic chronological validation search, never using test rows."""
from __future__ import annotations

from itertools import product
from src.evaluation.regression import regression_metrics
from .xgboost_model import XGBoostModel


def tune_xgboost(train, validation, features, target, horizon, base_params, seed=42, fast=False):
    """Return best parameters and validation score from 8 or 2 bounded fits."""
    candidates = list(product([2, 3], [.04, .08], [100, 180]))
    if fast:
        candidates = [(2, .04, 40), (3, .08, 40)]
    best = None
    for depth, rate, trees in candidates:
        params = {**base_params, "max_depth": depth, "learning_rate": rate, "n_estimators": trees}
        model = XGBoostModel(features, target, horizon, params, seed).fit(train)
        pred = model.predict(validation, validation["date"])["predicted_volatility"]
        score = regression_metrics(validation[target], pred)["rmse"]
        if best is None or score < best[1]:
            best = (params, score)
    return {"params": best[0], "validation_rmse": best[1], "candidates": len(candidates)}
