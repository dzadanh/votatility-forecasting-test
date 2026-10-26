"""Display the saved, once-computed test evaluation."""
import argparse
import json
from pathlib import Path
from src.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="ACB")
    parser.add_argument("--artifact-dir", help="Root containing the ticker's saved artifacts")
    args = parser.parse_args()
    root = Path(args.artifact_dir) if args.artifact_dir else ROOT / load_config()["paths"]["artifacts"]
    path = root / args.ticker.upper() / "metrics.json"
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
