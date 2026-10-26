"""Train selected offline model(s) and persist a comparable experiment."""
import argparse
import json
from src.pipeline.train_pipeline import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="ACB")
    parser.add_argument("--model", choices=["all", "garch", "xgboost", "lstm"], default="all")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--csv", help="Optional local OHLCV CSV path")
    parser.add_argument("--artifact-dir", help="Output root for a separate experiment")
    args = parser.parse_args()
    result = run_training(args.ticker, args.model, args.fast,
                          csv_path=args.csv, artifact_dir=args.artifact_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
