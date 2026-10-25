"""XGBoost parameter evaluation across expanding walk-forward folds."""
from __future__ import annotations

from itertools import product
from src.evaluation.regression import regression_metrics
from .xgboost_model import XGBoostModel


def evaluate_xgboost_params(
    validation_folds, features, target, horizon,
    n_estimators=160, max_depth=3, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, n_jobs=2, seed=42,
):
    """Return mean in-sample training RMSE and validation RMSE over CV folds."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "n_jobs": n_jobs,
    }
    fold_errors = []
    for train, validation in validation_folds:
        model = XGBoostModel(features, target, horizon, params, seed).fit(train)
        train_predictions = model.predict(train, train["date"])["predicted_volatility"]
        validation_predictions = model.predict(validation, validation["date"])["predicted_volatility"]
        fold_errors.append({
            "training_rmse": regression_metrics(train[target], train_predictions)["rmse"],
            "validation_rmse": regression_metrics(validation[target], validation_predictions)["rmse"],
        })
    return {
        "params": params,
        "training_rmse": sum(row["training_rmse"] for row in fold_errors) / len(fold_errors),
        "validation_rmse": sum(row["validation_rmse"] for row in fold_errors) / len(fold_errors),
        "fold_errors": fold_errors,
    }


def tune_xgboost(validation_folds, features, target, horizon, base_params, seed=42, fast=False):
    """Choose model parameters by mean validation RMSE across all CV folds."""
    candidates = list(product([2, 3], [.04, .08], [100, 180]))
    if fast:
        candidates = [(2, .04, 40), (3, .08, 40)]
    results = []
    for depth, rate, trees in candidates:
        params = {**base_params, "max_depth": depth, "learning_rate": rate, "n_estimators": trees}
        result = evaluate_xgboost_params(
            validation_folds, features, target, horizon,
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"], n_jobs=params["n_jobs"], seed=seed,
        )
        results.append(result)
    best = min(results, key=lambda row: row["validation_rmse"])
    return {
        "params": best["params"],
        "mean_training_rmse": best["training_rmse"],
        "mean_validation_rmse": best["validation_rmse"],
        "validation_rmse": best["validation_rmse"],
        "candidates": len(candidates),
        "folds": len(validation_folds),
    }
