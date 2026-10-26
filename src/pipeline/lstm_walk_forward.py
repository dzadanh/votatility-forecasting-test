"""Offline expanding walk-forward LSTM evaluation with a fixed HIGH definition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import ROOT, load_config
from src.evaluation.regression import regression_metrics
from src.evaluation.warning_metrics import (
    calibrate_high_threshold, high_alert_metrics, risk_level_metrics,
)
from src.models.base import UNIT
from src.models.lstm_model import LSTMModel
from src.models.lstm_tuning import expanding_folds, tune_lstm
from src.pipeline.dataset import prepare_dataset, valid_history
from src.risk.warning_engine import WarningEngine


def _calibrate(model, history, stop, target, warning, policy):
    forecast = model.predict(history, stop.date)
    probability = "high_probability" in forecast
    scores = forecast["high_probability" if probability else "predicted_volatility"]
    calibrated = calibrate_high_threshold(
        stop[target], scores, warning.high_threshold,
        max_alert_rate=float(policy.get("max_alert_rate", 0.15)),
        min_precision=float(policy.get("min_precision", 0.25)),
        fallback_threshold=1.0 if probability else None,
        fallback_to_budget=bool(policy.get("fallback_to_budget", False)),
    )
    if probability:
        warning.set_model_high_probability_threshold("lstm", calibrated["threshold"])
    else:
        warning.set_model_high_threshold("lstm", calibrated["threshold"])
    calibrated["score_kind"] = "high_probability" if probability else "predicted_volatility"
    return calibrated


def walk_forward_windows(labeled, score_start, horizon, stop_size, min_train=100):
    """Expose the exact purged train and stop windows available before a score block."""
    stop_end = score_start - horizon
    stop_start = stop_end - stop_size
    train_end = stop_start - horizon
    if train_end < min_train or stop_end > len(labeled):
        raise ValueError("Insufficient history for walk-forward train/stop windows")
    return labeled.iloc[:train_end], labeled.iloc[stop_start:stop_end]


def _fit_at_cut(labeled, history, score_start, horizon, stop_size, candidate,
                dataset, config, fixed_high):
    # A target at origin t consumes returns t+1..t+h. Both gaps are h origins.
    train, stop = walk_forward_windows(labeled, score_start, horizon, stop_size,
                                       max(100, candidate["sequence_length"]))
    model = LSTMModel(dataset.features, dataset.target, dataset.horizon,
                      candidate["sequence_length"], candidate["params"],
                      config["seed"]).fit(train, stop, history, high_threshold=fixed_high)
    warning = WarningEngine(**config["warning"])
    warning.warning_threshold = float(np.quantile(dataset.train[dataset.target], config["warning"]["q_warning"]))
    warning.high_threshold = fixed_high
    calibration = _calibrate(model, history, stop, dataset.target, warning,
                             config.get("lstm_walk_forward", config.get("alert_policy", {})))
    return model, warning, calibration, train, stop


def run_lstm_walk_forward(ticker: str, config_path: str | Path | None = None,
                          artifact_dir: str | Path | None = None,
                          block_size: int = 60, stop_size: int = 120,
                          csv_path: str | Path | None = None) -> dict:
    """Select on pre-test folds, then refit only from history available at each block."""
    if block_size < 1 or stop_size < 1:
        raise ValueError("Block and stop sizes must be positive")
    config = load_config(config_path or ROOT / "config.yaml")
    dataset = prepare_dataset(ticker, config, csv_path)
    history = valid_history(dataset)
    labeled = dataset.frame.dropna(subset=[*dataset.features, dataset.target]).reset_index(drop=True)
    reference = WarningEngine(**config["warning"]).fit(dataset.train[dataset.target])
    report = tune_lstm(ticker, config, dataset=dataset,
                       folds=expanding_folds(dataset, stop_size=stop_size))
    candidate = next(item for item in report["candidates"]
                     if item["name"] == report["selected_candidate"])
    chosen = {"name": candidate["name"], "sequence_length": candidate["sequence_length"],
              "params": candidate["params"]}
    rows, blocks = [], []
    test = dataset.test.reset_index(drop=True)
    for number, offset in enumerate(range(0, len(test), block_size), start=1):
        block = test.iloc[offset:offset + block_size]
        start = int(labeled.index[labeled.date.eq(block.date.iloc[0])][0])
        model, warning, calibration, train, stop = _fit_at_cut(
            labeled, history, start, dataset.horizon, stop_size, chosen,
            dataset, config, reference.high_threshold)
        forecast = model.predict(history, block.date)
        truth = block.set_index("date")[dataset.target]
        for predicted in forecast.itertuples():
            actual = float(truth.loc[predicted.date])
            probability = getattr(predicted, "high_probability", None)
            level = warning.classify(predicted.predicted_volatility, "lstm", probability)
            rows.append({"date": predicted.date, "model": "lstm", "partition": "test",
                         "actual_volatility": actual,
                         "predicted_volatility": float(predicted.predicted_volatility),
                         "raw_prediction": float(predicted.raw_prediction),
                         "high_probability": probability, "risk_level": level,
                         "actual_risk_level": reference.classify(actual),
                         "block_number": number})
        blocks.append({"block": number, "train_last": str(train.date.max().date()),
                       "stop_first": str(stop.date.min().date()),
                       "stop_last": str(stop.date.max().date()),
                       "test_first": str(block.date.min().date()),
                       "test_last": str(block.date.max().date()),
                       "test_count": len(block), "best_epoch": model.best_epoch,
                       "calibration": calibration})
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    actual_high = predictions.actual_risk_level.eq("HIGH")
    scores = (predictions.high_probability if predictions.high_probability.notna().all()
              else predictions.predicted_volatility)
    metrics = {**regression_metrics(predictions.actual_volatility, predictions.predicted_volatility),
               "high_warning": high_alert_metrics(actual_high, predictions.risk_level.eq("HIGH")),
               "risk_levels": risk_level_metrics(predictions.actual_risk_level, predictions.risk_level),
               "high_average_precision": (float(average_precision_score(actual_high, scores))
                   if actual_high.any() and not actual_high.all() else None),
               "high_score_kind": "high_probability" if predictions.high_probability.notna().all()
                                  else "predicted_volatility"}
    # Final live model uses labels known at the latest CSV date; it was not used
    # for historical test forecasts, which each had their own refit and threshold.
    final_start = len(labeled) + dataset.horizon
    final_model, final_warning, final_calibration, final_train, final_stop = _fit_at_cut(
        labeled, history, final_start, dataset.horizon, stop_size, chosen,
        dataset, config, reference.high_threshold)
    latest_date = history.date.max()
    latest = final_model.predict(history, [latest_date]).iloc[0]
    latest_probability = latest.get("high_probability", None)
    latest_row = {"date": latest_date, "model": "lstm", "partition": "latest",
                  "actual_volatility": None,
                  "predicted_volatility": float(latest.predicted_volatility),
                  "raw_prediction": float(latest.raw_prediction),
                  "high_probability": latest_probability,
                  "risk_level": final_warning.classify(latest.predicted_volatility,
                                                       "lstm", latest_probability),
                  "actual_risk_level": None, "block_number": None}
    predictions = pd.concat([predictions, pd.DataFrame([latest_row])], ignore_index=True)
    high = predictions.risk_level.eq("HIGH")
    predictions["high_alert_event"] = high & ~high.shift(fill_value=False)
    out = Path(artifact_dir or ROOT / "artifacts" / "lstm_high_walkforward") / ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(csv_path or Path(config["paths"]["raw"]) / f"{ticker.lower()}.csv")
    if not source.is_absolute():
        source = ROOT / source
    metadata = {
        "ticker": ticker.upper(), "target": dataset.target, "horizon": dataset.horizon,
        "features": list(dataset.features), "unit": UNIT,
        "sequence_length": chosen["sequence_length"], "seed": config["seed"],
        "fast": False, "source_csv": str(csv_path or Path(config["paths"]["raw"]) / f"{ticker.lower()}.csv"),
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_settings": {"lstm": chosen["params"]},
        "evaluation_mode": "expanding_walk_forward_refit_each_block",
        "selection_metric": report["selection_metric"],
        "selected_candidate": chosen["name"], "test_used_for_selection": False,
        "historical_test_inspected_during_development": True,
        "block_size": block_size, "stop_size": stop_size,
        "alert_policy": config.get("lstm_walk_forward", config.get("alert_policy", {})),
        "fixed_actual_high_threshold": reference.high_threshold,
        "fixed_actual_warning_threshold": reference.warning_threshold,
        "splits": {name: {"first": str(getattr(dataset, name).date.min().date()),
                          "last": str(getattr(dataset, name).date.max().date()),
                          "count": len(getattr(dataset, name))}
                   for name in ("train", "validation", "test")},
        "matched_test_count": len(test), "dropped_test_origins": 0,
        "latest_date": str(latest_date.date()),
        "blocks": blocks,
        "final_model": {"train_last": str(final_train.date.max().date()),
                        "stop_first": str(final_stop.date.min().date()),
                        "stop_last": str(final_stop.date.max().date()),
                        "best_epoch": final_model.best_epoch,
                        "validation_rmse": final_model.validation_loss ** .5,
                        "validation_high_ap": final_model.training_history[final_model.best_epoch - 1]["validation_high_ap"],
                        "calibration": final_calibration,
                        "history": final_model.training_history},
    }
    final_model.save(out / "lstm.joblib")
    final_warning.save(out / "warning.json")
    predictions.to_csv(out / "train_predictions.csv", index=False)
    for name, value in (("metrics.json", {"lstm": metrics}),
                        ("metadata.json", metadata), ("tuning.json", report)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return {"artifacts": str(out), "metrics": metrics,
            "selected_candidate": chosen["name"], "blocks": len(blocks)}
