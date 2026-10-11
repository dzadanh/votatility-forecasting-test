"""Display the saved, once-computed test evaluation."""
import argparse
import json
from src.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="ACB")
    args = parser.parse_args()
    path = ROOT / load_config()["paths"]["artifacts"] / args.ticker.upper() / "metrics.json"
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
