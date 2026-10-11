"""Deterministic OHLCV fixture independent of market data or network."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    rng = np.random.default_rng(42)
    n = 180
    close = 20 * np.exp(np.cumsum(rng.normal(0, .015, n)))
    open_ = close * (1 + rng.normal(0, .002, n))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": open_,
        "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * .99,
        "close": close, "volume": rng.integers(1000, 10000, n)})
