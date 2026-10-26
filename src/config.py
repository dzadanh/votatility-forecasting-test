"""Project configuration and schema validation."""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read configuration and reject incompatible feature or split settings."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data, split = config["data"], config["split"]
    if data["horizon"] < 2 or data["sequence_length"] < 1:
        raise ValueError("Horizon must be >=2 and sequence length must be positive")
    features = data["features"]
    if len(features) != len(set(features)) or any(x.startswith("target_") or x == "date" for x in features):
        raise ValueError("Feature schema contains duplicates or forbidden columns")
    if not 0 < split["test"] < 1:
        raise ValueError("Final test fraction must be between zero and one")
    if any(not isinstance(split[key], int) or split[key] <= 0
           for key in ("n_splits", "test_size", "gap")):
        raise ValueError("TimeSeriesSplit n_splits, test_size and gap must be positive integers")
    if split["gap"] < data["horizon"]:
        raise ValueError("Split gap must be at least the target horizon to prevent leakage")
    return config
