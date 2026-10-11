"""Offline experiment orchestration; models own their learning logic."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.config import ROOT, load_config
from src.evaluation.backtest import align_forecasts
from src.evaluation.regression import regression_metrics
from src.evaluation.warning_metrics import high_warning_metrics
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
                 csv_path: str | Path | None = None) -> dict:
    """Train selected models, evaluate once on matched test dates, save outputs."""
    config = load_config(config_path or ROOT / "config.yaml")
    if model != "all" and model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    names = MODEL_NAMES if model == "all" else (model,)
    dataset = prepare_dataset(ticker, config, csv_path)
    history = valid_history(dataset)
    out = Path(artifact_dir or ROOT / config["paths"]["artifacts"]) / ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    warning = WarningEngine(**config["warning"]).fit(dataset.train[dataset.target])
    warning.save(out / "warning.json")
    metadata = {"ticker": ticker.upper(), "target": dataset.target, "horizon": dataset.horizon,
                "features": list(dataset.features), "unit": UNIT, "sequence_length": config["data"]["sequence_length"],
                "seed": config["seed"], "fast": fast, "source_csv": str(csv_path or
                    Path(config["paths"]["raw"]) / f"{ticker.lower()}.csv"), "model_settings": {},
                "splits": {name: {"first": str(getattr(dataset, name)["date"].min().date()),
                                   "last": str(getattr(dataset, name)["date"].max().date()),
                                   "count": len(getattr(dataset, name))}
                           for name in ("train", "validation", "test")}}
    models, forecasts = {}, []
    for name in names:
        if name == "garch":
            settings = config["models"][name]
            fitted = GARCHModel(dataset.horizon, settings["mean"], settings["distribution"]).fit(dataset.train)
        elif name == "xgboost":
            settings = dict(config["models"][name])
            if fast:
                settings["n_estimators"] = min(settings["n_estimators"], 40)
            tuning = tune_xgboost(dataset.train, dataset.validation, dataset.features, dataset.target,
                                  dataset.horizon, settings, config["seed"], fast=fast)
            settings = tuning["params"]
            metadata["xgboost_tuning"] = tuning
            fitted = XGBoostModel(dataset.features, dataset.target, dataset.horizon, settings, config["seed"]).fit(dataset.train)
        else:
            settings = dict(config["models"][name])
            if fast:
                settings["epochs"] = min(settings["epochs"], 3)
                settings["patience"] = min(settings["patience"], 2)
            fitted = LSTMModel(dataset.features, dataset.target, dataset.horizon,
                               config["data"]["sequence_length"], settings, config["seed"]).fit(
                                   dataset.train, dataset.validation, history)
        metadata["model_settings"][name] = settings
        fitted.save(out / f"{name}.joblib")
        models[name] = fitted
        forecasts.append(fitted.predict(history, dataset.test["date"]))
    comparison = align_forecasts(forecasts, dataset.test, dataset.target)
    comparison["ticker"] = ticker.upper()
    comparison["partition"] = "test"
    comparison["risk_level"] = warning.classify(comparison["predicted_volatility"])
    comparison["actual_risk_level"] = warning.classify(comparison["actual_volatility"])
    latest_date = history["date"].max()
    latest = []
    for name, fitted in models.items():
        row = fitted.predict(history, [latest_date]).iloc[0].to_dict()
        row.update({"ticker": ticker.upper(), "partition": "latest", "actual_volatility": None,
                    "risk_level": warning.classify(row["predicted_volatility"]), "actual_risk_level": None})
        latest.append(row)
    predictions = pd.concat([comparison, pd.DataFrame(latest)], ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    metrics = {}
    for name, part in comparison.groupby("model"):
        metrics[name] = {**regression_metrics(part["actual_volatility"], part["predicted_volatility"]),
                         "high_warning": high_warning_metrics(part["actual_volatility"], part["predicted_volatility"],
                                                               warning.high_threshold)}
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
