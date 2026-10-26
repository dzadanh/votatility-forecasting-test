"""Select XGBoost for HIGH ranking while guarding volatility forecast error."""
from __future__ import annotations

from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
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


def _high_average_precision(estimator, x, y, *, high_threshold: float) -> float:
    """Rank future HIGH events using a threshold fixed before all CV windows."""
    high = np.asarray(y, dtype=float) >= high_threshold
    if not high.any():
        return 0.0
    if high.all():
        return 1.0
    return float(average_precision_score(high, estimator.predict(x)))


def _select_candidate(mse: np.ndarray, ap: np.ndarray, rmse_tolerance: float):
    """Maximize HIGH AP among candidates near the best validation RMSE."""
    if not np.isfinite(mse).all() or not np.isfinite(ap).all():
        raise ValueError("XGBoost validation produced nonfinite scores")
    best_rmse = float(np.sqrt(max(float(mse.min()), 0.0)))
    rmse_limit = best_rmse * (1 + rmse_tolerance)
    eligible = np.flatnonzero(np.sqrt(np.maximum(mse, 0.0)) <= rmse_limit + 1e-12)
    best_index = int(max(eligible, key=lambda i: (ap[i], -mse[i])))
    return best_index, best_rmse, rmse_limit, eligible


def tune_xgboost(
    training_frame: pd.DataFrame,
    validation_folds,
    features: tuple[str, ...],
    target: str,
    base_params: dict,
    tuning_config: dict | None = None,
    seed: int = 42,
    fast: bool = False,
    high_quantile: float = .70,
) -> dict:
    """Choose HIGH average precision within a validation RMSE tolerance.

    The HIGH definition for tuning is fitted on the first fold's training rows,
    before every validation window. The final warning threshold is fitted by
    the caller on all pre-test rows. No test observation selects parameters.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost requires 'xgboost'; install requirements.txt") from exc

    tuning_config = dict(tuning_config or {})
    param_grid = tuning_config.get("fast_param_grid" if fast else "param_grid")
    if not isinstance(param_grid, (dict, list)) or not param_grid:
        raise ValueError("XGBoost tuning requires a nonempty parameter grid")
    grids = [param_grid] if isinstance(param_grid, dict) else param_grid
    if any(not isinstance(grid, dict) or not grid for grid in grids):
        raise ValueError("Every XGBoost parameter grid must be a nonempty mapping")
    if any("n_jobs" in grid for grid in grids):
        raise ValueError("n_jobs controls runtime and must not be searched as a model parameter")
    if not 0 < high_quantile < 1:
        raise ValueError("HIGH quantile must be between zero and one")
    rmse_tolerance = float(tuning_config.get("rmse_tolerance", .05))
    if not 0 <= rmse_tolerance < 1:
        raise ValueError("RMSE tolerance must be in [0, 1)")
    if "n_jobs" not in base_params:
        base_params = {**base_params, "n_jobs": 1}

    x = feature_matrix(training_frame, features)
    y = training_frame[target].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Training target must be finite")
    cv_splits = _chronological_splits(training_frame, validation_folds)
    high_threshold = float(np.quantile(validation_folds[0][0][target], high_quantile))

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
        scoring={"high_ap": partial(_high_average_precision, high_threshold=high_threshold),
                 "neg_mse": "neg_mean_squared_error"},
        cv=cv_splits,
        n_jobs=search_jobs,
        refit=False,
        return_train_score=False,
        error_score="raise",
        pre_dispatch=search_jobs if search_jobs > 0 else "2*n_jobs",
    )
    search.fit(x, y)

    results = search.cv_results_
    mse = -np.asarray(results["mean_test_neg_mse"], dtype=float)
    ap = np.asarray(results["mean_test_high_ap"], dtype=float)
    best_index, best_rmse, rmse_limit, eligible = _select_candidate(mse, ap, rmse_tolerance)
    fold_rmse = [float(np.sqrt(max(-float(results[f"split{i}_test_neg_mse"][best_index]), 0.0)))
                 for i in range(len(cv_splits))]
    fold_ap = [float(results[f"split{i}_test_high_ap"][best_index])
               for i in range(len(cv_splits))]

    return {
        "params": results["params"][best_index],
        "mean_validation_rmse": float(np.mean(fold_rmse)),
        "validation_rmse": float(np.sqrt(max(float(mse[best_index]), 0.0))),
        "validation_mse": float(mse[best_index]),
        "validation_high_ap": float(ap[best_index]),
        "fold_validation_rmse": fold_rmse,
        "fold_validation_high_ap": fold_ap,
        "high_definition": {"quantile": float(high_quantile),
                            "threshold_from_first_fold_train": high_threshold},
        "selection": {"metric": "high_average_precision_subject_to_rmse_guard",
                      "rmse_tolerance": rmse_tolerance,
                      "best_candidate_rmse": best_rmse,
                      "maximum_eligible_rmse": rmse_limit},
        "candidate_results": [
            {"params": params, "validation_rmse": float(np.sqrt(max(value, 0.0))),
             "validation_high_ap": float(ap[i]), "eligible": bool(i in eligible)}
            for i, (params, value) in enumerate(zip(results["params"], mse))
        ],
        "baseline_validation_rmse": _baseline_rmse(validation_folds, target, features),
        "candidates": len(ParameterGrid(param_grid)),
        "folds": len(cv_splits),
        "scoring": ["high_average_precision", "neg_mean_squared_error"],
        "refit": False,
        "search_n_jobs": search_jobs,
        "estimator_n_jobs": int(base_params["n_jobs"]),
    }
