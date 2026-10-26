"""Compare saved baseline and walk-forward LSTM predictions on identical test origins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import ROOT


def _summary(folder: Path) -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["lstm"]
    predictions = pd.read_csv(folder / "train_predictions.csv")
    predictions = predictions[(predictions.model == "lstm") &
                              (predictions.partition == "test")].sort_values("date").reset_index(drop=True)
    truth = predictions.actual_risk_level.eq("HIGH")
    score_kind = "high_probability" if "high_probability" in predictions and predictions.high_probability.notna().all() else "predicted_volatility"
    ap = float(average_precision_score(truth, predictions[score_kind])) if truth.any() and not truth.all() else None
    high = metrics["high_warning"]
    return ({"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"],
             "high_average_precision": ap, "score_kind": score_kind,
             "high_precision": high["precision"], "high_recall": high["recall"],
             "high_f1": high["f1"], "high_support": high["support"],
             "alerts": high["alert_count"], "true_positives": high["confusion_matrix"][1][1],
             "false_positives": high["confusion_matrix"][0][1],
             "test_count": len(predictions)}, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "artifacts" / "baseline_before_high_walkforward_20261026")
    parser.add_argument("--new", type=Path,
                        default=ROOT / "artifacts" / "lstm_high_walkforward")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "lstm_high_walkforward" / "comparison.json")
    args = parser.parse_args()
    result = {}
    for ticker in ("ACB", "FPT", "HPG", "VIC"):
        old, old_rows = _summary(args.baseline / ticker)
        new, new_rows = _summary(args.new / ticker)
        if (not old_rows.date.equals(new_rows.date) or
            not old_rows.actual_volatility.equals(new_rows.actual_volatility) or
            not old_rows.actual_risk_level.equals(new_rows.actual_risk_level)):
            raise ValueError(f"{ticker}: old and new test dates, targets, or HIGH labels differ")
        result[ticker] = {"baseline": old, "walk_forward_high": new,
                          "delta": {key: (new[key] - old[key] if new[key] is not None and old[key] is not None else None) for key in
                                    ("rmse", "mae", "r2", "high_average_precision",
                                     "high_precision", "high_recall", "high_f1",
                                     "alerts", "true_positives", "false_positives")},
                          "matched_test_targets": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
