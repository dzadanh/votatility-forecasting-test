"""Purged, expanding-window comparison of LSTM settings before the test split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.regression import regression_metrics
from src.evaluation.warning_metrics import calibrate_high_threshold, high_warning_metrics
from src.models.lstm_model import LSTMModel
from src.pipeline.dataset import Dataset, valid_history
from src.risk.warning_engine import WarningEngine


@dataclass(frozen=True)
class TemporalFold:
    train: pd.DataFrame
    stop: pd.DataFrame
    score: pd.DataFrame


def expanding_folds(dataset: Dataset, count: int = 3, score_size: int = 160,
                    stop_size: int = 120) -> list[TemporalFold]:
    """Reserve the final pre-test blocks for scoring; purge target overlap at both cuts."""
    if min(count, score_size, stop_size) < 1:
        raise ValueError("Fold count and window sizes must be positive")
    pretest = dataset.frame.loc[dataset.frame.date <= dataset.validation.date.max()].dropna(
        subset=[*dataset.features, dataset.target]
    ).reset_index(drop=True)
    folds = []
    for number in range(count):
        score_end = len(pretest) - (count - number - 1) * score_size
        score_start = score_end - score_size
        stop_end = score_start - dataset.horizon
        stop_start = stop_end - stop_size
        train_end = stop_start - dataset.horizon
        if train_end < max(100, dataset.horizon * 2):
            raise ValueError("Insufficient pre-test history for requested folds")
        folds.append(TemporalFold(
            pretest.iloc[:train_end].copy(),
            pretest.iloc[stop_start:stop_end].copy(),
            pretest.iloc[score_start:score_end].copy(),
        ))
    return folds


def candidate_grid(config: dict) -> list[dict]:
    """Small search that directly tests whether a HIGH head improves ranking."""
    base = dict(config["models"]["lstm"])
    length = int(config["data"]["sequence_length"])
    return [
        {"name": "regression_only", "sequence_length": length,
         "params": {**base, "high_loss_weight": 0.0}},
        {"name": "high_weight_025", "sequence_length": length,
         "params": {**base, "high_loss_weight": 0.25}},
        {"name": "high_weight_05", "sequence_length": length,
         "params": {**base, "high_loss_weight": 0.5}},
        {"name": "high_weight_1", "sequence_length": length,
         "params": {**base, "high_loss_weight": 1.0}},
        {"name": "short_10_high", "sequence_length": 10,
         "params": {**base, "high_loss_weight": 0.5}},
    ]


def tune_lstm(ticker: str, config: dict, dataset: Dataset | None = None,
              candidates: list[dict] | None = None, folds: list[TemporalFold] | None = None) -> dict:
    """Choose by mean out-of-fold HIGH AP; never inspect test to select settings."""
    if dataset is None:
        from src.pipeline.dataset import prepare_dataset
        dataset = prepare_dataset(ticker, config)
    if candidates is None:
        candidates = candidate_grid(config)
    if folds is None:
        folds = expanding_folds(dataset)
    if not candidates or not folds:
        raise ValueError("At least one candidate and fold are required")
    history = valid_history(dataset)
    results = []
    for candidate in candidates:
        fold_results = []
        for fold in folds:
            warning = WarningEngine(**config["warning"]).fit(fold.train[dataset.target])
            model = LSTMModel(dataset.features, dataset.target, dataset.horizon,
                              candidate["sequence_length"], candidate["params"],
                              config["seed"]).fit(fold.train, fold.stop, history,
                                                   high_threshold=warning.high_threshold)
            forecast = model.predict(history, fold.score.date)
            predicted = forecast["predicted_volatility"].to_numpy(float)
            actual = fold.score[dataset.target].to_numpy(float)
            score_kind = "high_probability" if "high_probability" in forecast else "predicted_volatility"
            high_scores = forecast[score_kind].to_numpy(float)
            stop_predictions = model.predict(history, fold.stop.date)[score_kind]
            policy = config.get("lstm_walk_forward", config.get("alert_policy", {}))
            calibration = calibrate_high_threshold(
                fold.stop[dataset.target], stop_predictions, warning.high_threshold,
                max_alert_rate=float(policy.get("max_alert_rate", 0.15)),
                min_precision=float(policy.get("min_precision", 0.25)),
                fallback_threshold=1.0 if score_kind == "high_probability" else None,
                fallback_to_budget=bool(policy.get("fallback_to_budget", False)),
            )
            high = actual >= warning.high_threshold
            fold_results.append({
                "train_last": str(fold.train.date.max().date()),
                "stop_first": str(fold.stop.date.min().date()),
                "stop_last": str(fold.stop.date.max().date()),
                "score_first": str(fold.score.date.min().date()),
                "score_last": str(fold.score.date.max().date()),
                "train_count": len(fold.train),
                "stop_count": len(fold.stop),
                "score_count": len(fold.score),
                "best_epoch": model.best_epoch,
                "metrics": regression_metrics(actual, predicted),
                "high_average_precision": float(average_precision_score(high, high_scores))
                    if high.any() and not high.all() else None,
                "high_prevalence": float(high.mean()),
                "high_warning": high_warning_metrics(
                    actual, high_scores, warning.high_threshold, calibration["threshold"]),
                "high_score_kind": score_kind,
                "calibration_method": calibration["method"],
            })
        results.append({
            **candidate,
            "mean_rmse": float(np.mean([item["metrics"]["rmse"] for item in fold_results])),
            "mean_high_average_precision": float(np.mean([
                item["high_average_precision"] for item in fold_results
                if item["high_average_precision"] is not None
            ])) if any(item["high_average_precision"] is not None for item in fold_results) else None,
            "mean_high_recall": float(np.mean([item["high_warning"]["recall"] for item in fold_results])),
            "mean_high_precision": float(np.mean([item["high_warning"]["precision"] for item in fold_results])),
            "mean_alert_rate_percent": float(np.mean([
                item["high_warning"]["alert_rate_percent"] for item in fold_results])),
            "folds": fold_results,
        })
    winner = max(results, key=lambda item: (item["mean_high_average_precision"]
                 if item["mean_high_average_precision"] is not None else -1,
                 item["mean_high_recall"], -item["mean_rmse"]))
    baseline = next((item for item in results if item["name"] == "regression_only"), None)
    min_gain = 0.01
    if (baseline is not None and baseline["mean_high_average_precision"] is not None and
        winner is not baseline and
        (winner["mean_high_average_precision"] is None or
         winner["mean_high_average_precision"] < baseline["mean_high_average_precision"] + min_gain)):
        winner = baseline
    return {
        "ticker": ticker.upper(),
        "selection_metric": "mean_high_average_precision_across_purged_expanding_folds",
        "selected_candidate": winner["name"],
        "minimum_ap_gain_for_high_head": min_gain,
        "test_used_for_selection": False,
        "fold_count": len(folds),
        "horizon": dataset.horizon,
        "candidates": results,
    }
