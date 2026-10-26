"""Offline experiment orchestration; models own their learning logic."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import ROOT, load_config
from src.evaluation.backtest import align_forecasts
from src.evaluation.regression import regression_metrics
from src.evaluation.warning_metrics import calibrate_high_threshold, high_warning_metrics, risk_level_metrics
from src.explainability.shap_explainer import explain_xgboost
from src.models.base import UNIT
from src.models.garch_model import GARCHModel
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel
from src.models.xgboost_tuning import tune_xgboost
from src.pipeline.dataset import prepare_dataset, valid_history
from src.risk.warning_engine import WarningEngine

MODEL_NAMES = ("garch", "xgboost", "lstm")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_training(ticker: str = "ACB", model: str = "all", fast: bool = False,
                 config_path: str | Path | None = None, artifact_dir: str | Path | None = None,
                 csv_path: str | Path | None = None, lstm_candidate: dict | None = None) -> dict:
    """Train selected models, evaluate once on matched test dates, save outputs."""
    config = load_config(config_path or ROOT / "config.yaml")
    if model != "all" and model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    if lstm_candidate is not None and (model != "lstm" or fast):
        raise ValueError("A tuned LSTM candidate requires --model lstm without --fast")
    names = MODEL_NAMES if model == "all" else (model,)
    dataset = prepare_dataset(ticker, config, csv_path)
    has_validation_folds = hasattr(dataset, "validation_folds")
    validation_folds = (dataset.validation_folds if has_validation_folds
                        else ((dataset.train, dataset.validation),))
    processed_dir = Path(config.get("paths", {}).get("processed", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    history = valid_history(dataset)
    out = Path(artifact_dir or ROOT / config["paths"]["artifacts"]) / ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    warning = WarningEngine(**config["warning"]).fit(dataset.train[dataset.target])
    alert_policy = config.get("alert_policy", {})
    max_alert_rate = float(alert_policy.get("max_alert_rate", 0.15))
    min_precision = float(alert_policy.get("min_precision", 0.25))
    model_policy_overrides = alert_policy.get("model_overrides", {})
    sequence_length = int(lstm_candidate["sequence_length"]) if lstm_candidate else int(config["data"]["sequence_length"])
    metadata = {"ticker": ticker.upper(), "target": dataset.target, "horizon": dataset.horizon,
                "features": list(dataset.features), "unit": UNIT, "sequence_length": sequence_length,
                "seed": config["seed"], "fast": fast, "source_csv": str(csv_path or
                    Path(config["paths"]["raw"]) / f"{ticker.lower()}.csv"), "model_settings": {},
                "splits": {"train": {"first": str(dataset.train.date.min().date()),
                                      "last": str(dataset.train.date.max().date()), "count": len(dataset.train)},
                           "validation_folds": [{"train_count": len(tr), "first": str(val.date.min().date()),
                                                 "last": str(val.date.max().date()), "count": len(val)}
                                                for tr, val in validation_folds],
                           "test": {"first": str(dataset.test.date.min().date()),
                                    "last": str(dataset.test.date.max().date()), "count": len(dataset.test)}}}
    if not has_validation_folds:
        metadata["splits"]["validation"] = {
            "first": str(dataset.validation.date.min().date()),
            "last": str(dataset.validation.date.max().date()),
            "count": len(dataset.validation),
        }
    if lstm_candidate:
        metadata["lstm_selection"] = {"candidate": lstm_candidate["name"],
                                      "source": "purged_expanding_fold_mean_high_average_precision"}
    models, forecasts, alert_calibration = {}, [], {}
    for name in names:
        if name == "garch":
            settings = config["models"][name]
            fitted = GARCHModel(dataset.horizon, settings["mean"], settings["distribution"]).fit(dataset.train)
        elif name == "xgboost":
            settings = dict(config["models"][name])
            if has_validation_folds:
                tuning_config = settings.pop("tuning", {})
                tuning = tune_xgboost(
                    dataset.train, validation_folds, dataset.features,
                    dataset.target, settings, tuning_config, config["seed"], fast=fast,
                    high_quantile=warning.q_high,
                )
                settings.update(tuning["params"])
            else:
                if fast:
                    settings["n_estimators"] = min(settings["n_estimators"], 40)
                tuning = tune_xgboost(dataset.train, dataset.validation, dataset.features,
                                      dataset.target, dataset.horizon, settings, config["seed"], fast=fast)
                settings = tuning["params"]
            metadata["xgboost_tuning"] = tuning
            fitted = XGBoostModel(dataset.features, dataset.target, dataset.horizon, settings, config["seed"]).fit(dataset.train)
        else:
            settings = dict(lstm_candidate["params"] if lstm_candidate else config["models"][name])
            if fast:
                settings["epochs"] = min(settings["epochs"], 3)
                settings["patience"] = min(settings["patience"], 2)
            fold_train, fold_validation = validation_folds[-1]
            fitted = LSTMModel(dataset.features, dataset.target, dataset.horizon,
                               sequence_length, settings, config["seed"]).fit(
                                   fold_train, fold_validation, history,
                                   final_training_frame=dataset.train if has_validation_folds else None,
                                   high_threshold=warning.high_threshold)
            metadata["lstm_training"] = {
                "best_epoch": fitted.best_epoch,
                "validation_rmse": fitted.validation_loss ** 0.5,
                "validation_high_ap": fitted.training_history[fitted.best_epoch - 1]["validation_high_ap"],
                "history": fitted.training_history,
            }
        metadata["model_settings"][name] = settings
        fitted.save(out / f"{name}.joblib")
        models[name] = fitted
        if has_validation_folds:
            fold_forecasts = []
            for fold_train, fold_validation in validation_folds:
                if name == "garch":
                    cv_model = GARCHModel(dataset.horizon, settings["mean"], settings["distribution"]).fit(fold_train)
                elif name == "xgboost":
                    cv_model = XGBoostModel(dataset.features, dataset.target, dataset.horizon,
                                            settings, config["seed"]).fit(fold_train)
                else:
                    cv_model = LSTMModel(dataset.features, dataset.target, dataset.horizon,
                                         sequence_length, settings, config["seed"]).fit(
                                             fold_train, fold_validation, history,
                                             high_threshold=warning.high_threshold)
                fold_forecasts.append(cv_model.predict(history, fold_validation["date"]))
            validation_forecast = pd.concat(fold_forecasts, ignore_index=True).sort_values("date").reset_index(drop=True)
        else:
            validation_forecast = fitted.predict(history, dataset.validation["date"])
        validation_truth = dataset.validation.sort_values("date").reset_index(drop=True)
        high_probability = "high_probability" in validation_forecast
        validation_scores = (validation_forecast["high_probability"] if high_probability
                             else validation_forecast["predicted_volatility"])
        model_policy = model_policy_overrides.get(name, {})
        model_max_alert_rate = float(model_policy.get("max_alert_rate", max_alert_rate))
        model_min_precision = float(model_policy.get("min_precision", min_precision))
        calibration = calibrate_high_threshold(
            validation_truth[dataset.target],
            validation_scores,
            warning.high_threshold,
            max_alert_rate=model_max_alert_rate,
            min_precision=model_min_precision,
            fallback_threshold=1.0 if high_probability else None,
        )
        if high_probability:
            warning.set_model_high_probability_threshold(name, calibration["threshold"])
            calibration["score_kind"] = "high_probability"
        else:
            warning.set_model_high_threshold(name, calibration["threshold"])
            calibration["score_kind"] = "predicted_volatility"
        alert_calibration[name] = calibration
        forecasts.append(fitted.predict(history, dataset.test["date"]))
    warning.save(out / "warning.json")
    metadata["alert_calibration"] = alert_calibration
    metadata["alert_policy"] = {"max_alert_rate": max_alert_rate, "min_precision": min_precision,
                                 "model_overrides": model_policy_overrides,
                                 "actual_high_threshold": warning.high_threshold}
    comparison = align_forecasts(forecasts, dataset.test, dataset.target)
    comparison["ticker"] = ticker.upper()
    comparison["partition"] = "test"
    comparison["risk_level"] = [warning.classify(row.predicted_volatility, model=row.model,
                                  high_probability=row.high_probability if pd.notna(row.high_probability) else None)
                                 for row in comparison.itertuples()]
    comparison["actual_risk_level"] = warning.classify(comparison["actual_volatility"])
    latest_date = history["date"].max()
    latest = []
    for name, fitted in models.items():
        row = fitted.predict(history, [latest_date]).iloc[0].to_dict()
        row.update({"ticker": ticker.upper(), "partition": "latest", "actual_volatility": None,
                    "risk_level": warning.classify(row["predicted_volatility"], model=name,
                        high_probability=row.get("high_probability")), "actual_risk_level": None})
        latest.append(row)
    predictions = pd.concat([comparison, pd.DataFrame(latest)], ignore_index=True)
    predictions = predictions.sort_values(["model", "date"]).reset_index(drop=True)
    predicted_high = predictions["risk_level"].eq("HIGH")
    previous_high = predicted_high.groupby(predictions["model"]).shift(fill_value=False)
    predictions["high_alert_event"] = predicted_high & ~previous_high
    predictions.to_csv(out / "train_predictions.csv", index=False)
    metrics = {}
    for name, part in comparison.groupby("model"):
        high_probability = name in warning.model_high_probability_thresholds
        scores = part["high_probability"] if high_probability else part["predicted_volatility"]
        threshold = (warning.model_high_probability_thresholds[name] if high_probability else
                     warning.model_high_thresholds.get(name, warning.high_threshold))
        actual_high = part["actual_volatility"] >= warning.high_threshold
        metrics[name] = {**regression_metrics(part["actual_volatility"], part["predicted_volatility"]),
                         "high_warning": high_warning_metrics(part["actual_volatility"], scores,
                                                               warning.high_threshold, threshold),
                         "risk_levels": risk_level_metrics(part["actual_risk_level"], part["risk_level"])}
        if high_probability:
            metrics[name]["high_average_precision"] = (float(average_precision_score(actual_high, scores))
                if actual_high.any() and not actual_high.all() else None)
    _write_json(out / "metrics.json", metrics)
    metadata["matched_test_count"] = int(comparison["date"].nunique())
    metadata["dropped_test_origins"] = int(len(dataset.test) - metadata["matched_test_count"])
    metadata["latest_date"] = str(latest_date.date())
    _write_json(out / "metadata.json", metadata)
    if "xgboost" in models:
        sample = pd.concat([history.loc[history["date"].isin(comparison["date"])].tail(29),
                            history.tail(1)], ignore_index=True)
        _write_json(out / "shap.json", explain_xgboost(models["xgboost"], sample))
    return {"artifacts": str(out), "metrics": metrics, "metadata": metadata}
