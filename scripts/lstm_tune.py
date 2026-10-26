"""Walk-forward LSTM search and optional isolated fit for a local ticker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ROOT, load_config
from src.models.lstm_tuning import tune_lstm
from src.pipeline.train_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output", type=Path, help="JSON search report path")
    parser.add_argument("--fit-selected", action="store_true",
                        help="Fit selected candidate in artifacts/lstm_tuned, leaving main artifacts intact")
    args = parser.parse_args()

    report = tune_lstm(args.ticker, load_config())
    if args.fit_selected:
        selected = next(item for item in report["candidates"]
                        if item["name"] == report["selected_candidate"])
        trained = run_training(args.ticker, model="lstm",
                               artifact_dir=ROOT / "artifacts" / "lstm_tuned",
                               lstm_candidate=selected)
        report["isolated_artifacts"] = trained["artifacts"]
        report["selected_test_metrics"] = trained["metrics"]["lstm"]
        report["test_interpretation"] = (
            "This historical test period has already been inspected in prior experiments; "
            "its result is descriptive, not an independent confirmation of tuning."
        )
    output = args.output or ROOT / "artifacts" / "lstm_tuning" / f"{args.ticker.lower()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "ticker": report["ticker"],
        "selected_candidate": report["selected_candidate"],
        "selection_metric": report["selection_metric"],
        "mean_high_average_precision": {item["name"]: item["mean_high_average_precision"]
                                        for item in report["candidates"]},
        "mean_rmse": {item["name"]: item["mean_rmse"] for item in report["candidates"]},
        "report": str(output),
        "isolated_artifacts": report.get("isolated_artifacts"),
    }, indent=2))


if __name__ == "__main__":
    main()
