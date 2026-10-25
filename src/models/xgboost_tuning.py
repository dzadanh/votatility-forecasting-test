"""Joint XGBoost hyperparameter search over chronological validation folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid

from src.evaluation.regression import regression_metrics
from src.pipeline.dataset import feature_matrix


def _chronological_splits(training_frame: pd.DataFrame, validation_folds):
    """Map the precomputed date folds onto the row positions used by sklearn."""
    index = training_frame.index
    splits = []
    for train, validation in validation_folds:
        train_indices = index.get_indexer(train.index)
        validation_indices = index.get_indexer(validation.index)
        if (train_indices < 0).any() or (validation_indices < 0).any():
            raise ValueError("A validation fold contains rows outside the tuning frame")
        if not len(train_indices) or not len(validation_indices):
            raise ValueError("Chronological tuning folds must have nonempty train and validation rows")
        if np.intersect1d(train_indices, validation_indices).size:
            raise ValueError("Training and validation rows overlap within a fold")
        splits.append((train_indices, validation_indices))
    if not splits:
        raise ValueError("At least one chronological validation fold is required")
    return splits


def _baseline_rmse(validation_folds, target: str, features: tuple[str, ...]) -> dict:
    """Score simple fold-local mean and historical-volatility baselines."""
    mean_errors, persistence_errors = [], []
    has_persistence = "rolling_volatility_5" in features
    for train, validation in validation_folds:
        y = validation[target].to_numpy(dtype=float)
        mean_prediction = np.full(len(validation), train[target].mean(), dtype=float)
        mean_errors.append(regression_metrics(y, mean_prediction)["rmse"])
        if has_persistence:
            persistence = validation["rolling_volatility_5"].to_numpy(dtype=float)
            persistence_errors.append(regression_metrics(y, np.maximum(persistence, 0))["rmse"])
    result = {"training_target_mean": float(np.mean(mean_errors))}
    if persistence_errors:
        result["rolling_volatility_5"] = float(np.mean(persistence_errors))
    return result


def tune_xgboost(
    training_frame: pd.DataFrame,
    validation_folds,
    features: tuple[str, ...],
    target: str,
    base_params: dict,
    tuning_config: dict | None = None,
    seed: int = 42,
    fast: bool = False,
) -> dict:
    """Search joint parameter combinations using existing chronological folds.

    GridSearchCV selects parameters using only each fold's train/validation rows.
    Its refit is disabled; the caller later fits the final wrapper once on the
    complete pre-test training frame and evaluates the independent test period.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost requires 'xgboost'; install requirements.txt") from exc

    tuning_config = dict(tuning_config or {})
    param_grid = tuning_config.get("fast_param_grid" if fast else "param_grid")
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("XGBoost tuning requires a nonempty parameter grid")
    if "n_jobs" in param_grid:
        raise ValueError("n_jobs controls runtime and must not be searched as a model parameter")
    if "n_jobs" not in base_params:
        base_params = {**base_params, "n_jobs": 1}

    x = feature_matrix(training_frame, features)
    y = training_frame[target].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Training target must be finite")
    cv_splits = _chronological_splits(training_frame, validation_folds)

    search_jobs = int(tuning_config.get("search_n_jobs", 1))
    if search_jobs == 0:
        raise ValueError("search_n_jobs cannot be zero")
    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        **base_params,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv_splits,
        n_jobs=search_jobs,
        refit=False,
        return_train_score=False,
        error_score="raise",
        pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
    )
    search.fit(x, y)

    best_index = int(search.best_index_)
    fold_rmse = []
    for fold_index in range(len(cv_splits)):
        fold_mse = -float(search.cv_results_[f"split{fold_index}_test_score"][best_index])
        fold_rmse.append(float(np.sqrt(max(fold_mse, 0.0))))

    return {
        "params": search.best_params_,
        "mean_validation_rmse": float(np.mean(fold_rmse)),
        "validation_rmse": float(np.sqrt(max(-search.best_score_, 0.0))),
        "validation_mse": float(-search.best_score_),
        "fold_validation_rmse": fold_rmse,
        "baseline_validation_rmse": _baseline_rmse(validation_folds, target, features),
        "candidates": len(ParameterGrid(param_grid)),
        "folds": len(cv_splits),
        "scoring": "neg_mean_squared_error",
        "refit": False,
        "search_n_jobs": search_jobs,
        "estimator_n_jobs": int(base_params["n_jobs"]),
    }
