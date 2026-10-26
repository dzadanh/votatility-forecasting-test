"""Rebuild the feature dataset and annotate its walk-forward partitions."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ROOT, load_config
from src.pipeline.dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="ACB")
    parser.add_argument("--csv", help="Optional local OHLCV CSV path")
    args = parser.parse_args()
    config = load_config()
    dataset = prepare_dataset(args.ticker, config, args.csv)
    processed_dir = Path(config.get("paths", {}).get("processed", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    output = processed_dir / f"{args.ticker.casefold()}_features_target_{dataset.horizon}d.csv"
    dataset.frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(dataset.frame)} rows to {output.name}")
    print(f"Train: {len(dataset.train)} | validation: {len(dataset.validation_folds)} folds x "
          f"{len(dataset.validation_folds[0][1])} | test: {len(dataset.test)}")


if __name__ == "__main__":
    main()
