"""Relative features should not depend on price or volume denominations."""
import numpy as np

from src.config import load_config
from src.features.builder import build_features


def test_relative_features_are_scale_invariant(synthetic_ohlcv):
    original = build_features(synthetic_ohlcv)
    rescaled = synthetic_ohlcv.copy()
    rescaled[["open", "high", "low", "close"]] *= 10
    rescaled["volume"] *= 7
    changed = build_features(rescaled)

    names = [f"close_to_ma_{window}" for window in (5, 10, 20, 50)] + ["volume_to_ma_20"]
    assert set(names).issubset(load_config()["data"]["features"])
    for name in names:
        np.testing.assert_allclose(original[name], changed[name], rtol=1e-12, atol=1e-12, equal_nan=True)
