"""Compare a small LSTM parameter set on validation without touching saved models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import average_precision_score

from src.config import load_config
from src.evaluation.regression import regression_metrics
from src.models.lstm_model import LSTMModel
from src.pipeline.dataset import prepare_dataset, valid_history
from src.risk.warning_engine import WarningEngine


def compare_lstm(ticker: str) -> dict:
    config = load_config()
    dataset = prepare_dataset(ticker, config)
    history = valid_history(dataset)
    base = dict(config["models"]["lstm"])
    length = int(config["data"]["sequence_length"])
    candidates = (
        ("baseline", length, {}),
        ("short_context", 10, {}),
        ("long_context", 40, {}),
        ("lower_learning_rate", length, {"learning_rate": 0.0003}),
    )
    high_threshold = WarningEngine(**config["warning"]).fit(
        dataset.train[dataset.target]
    ).high_threshold
    actual = dataset.validation[dataset.target].to_numpy(float)
    high = actual >= high_threshold
    results = []
    for name, sequence_length, overrides in candidates:
        params = {**base, **overrides}
        model = LSTMModel(
            dataset.features, dataset.target, dataset.horizon,
            sequence_length, params, config["seed"],
        ).fit(dataset.train, dataset.validation, history, high_threshold=high_threshold)
        forecast = model.predict(history, dataset.validation["date"])
        predicted = forecast["predicted_volatility"].to_numpy(float)
        high_scores = (forecast["high_probability"] if "high_probability" in forecast
                       else forecast["predicted_volatility"])
        results.append({
            "name": name,
            "sequence_length": sequence_length,
            "params": params,
            "validation": regression_metrics(actual, predicted),
            "high_average_precision": float(average_precision_score(high, high_scores))
                if high.any() and not high.all() else None,
            "high_prevalence": float(high.mean()),
            "best_epoch": model.best_epoch,
            "history": model.training_history,
        })
    winner = min(results, key=lambda item: item["validation"]["rmse"])
    return {
        "ticker": ticker.upper(),
        "selection_split": "validation",
        "selected_by": "validation_rmse",
        "selected_candidate": winner["name"],
        "target": dataset.target,
        "train_last": str(dataset.train["date"].max().date()),
        "validation_first": str(dataset.validation["date"].min().date()),
        "validation_last": str(dataset.validation["date"].max().date()),
        "candidates": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = compare_lstm(args.ticker)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {args.output}")
    print(json.dumps({
        "ticker": report["ticker"],
        "selected_candidate": report["selected_candidate"],
        "validation_rmse": {
            item["name"]: item["validation"]["rmse"] for item in report["candidates"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
