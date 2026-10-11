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
    if abs(sum(split.values()) - 1.0) > 1e-8 or any(v <= 0 for v in split.values()):
        raise ValueError("Chronological split fractions must be positive and sum to one")
    return config
