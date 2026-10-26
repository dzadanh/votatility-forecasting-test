"""Run the HIGH-focused LSTM walk-forward experiment on existing local CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.lstm_walk_forward import run_lstm_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--block-size", type=int, default=60)
    parser.add_argument("--stop-size", type=int, default=120)
    parser.add_argument("--csv", type=Path, help="Existing local OHLCV CSV; no download occurs")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    result = run_lstm_walk_forward(args.ticker, artifact_dir=args.artifact_dir,
                                   block_size=args.block_size, stop_size=args.stop_size,
                                   csv_path=args.csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
