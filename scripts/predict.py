"""Latest local forecast CLI."""
import argparse
import json
from src.pipeline.predict_pipeline import predict_latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="ACB")
    parser.add_argument("--model", choices=["garch", "xgboost", "lstm"], default="xgboost")
    parser.add_argument("--csv", help="Optional local OHLCV CSV path")
    parser.add_argument("--artifact-dir", help="Parent folder containing the ticker artifact folder")
    args = parser.parse_args()
    print(json.dumps(predict_latest(args.ticker, args.model, csv_path=args.csv,
                                    artifact_dir=args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
