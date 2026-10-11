"""Shared forecast output contract; fitting inputs remain model-specific."""
from __future__ import annotations
import numpy as np
import pandas as pd

FORECAST_COLUMNS = ["date", "model", "predicted_volatility"]
UNIT = "percentage_points_daily_return_sample_std"


def forecast_frame(dates, model: str, values) -> pd.DataFrame:
    """Validate and construct a date-aligned nonnegative forecast table."""
    dates = pd.to_datetime(dates)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(dates) != len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecast dates/values must align and values be finite, nonnegative")
    if pd.Index(dates).duplicated().any():
        raise ValueError("Forecast dates must be unique")
    return pd.DataFrame({"date": dates, "model": model, "predicted_volatility": values})


def validate_origins(history: pd.DataFrame, origin_dates) -> pd.DatetimeIndex:
    """Require sorted unique history and origins present in that history."""
    dates = pd.DatetimeIndex(pd.to_datetime(history["date"]))
    origins = pd.DatetimeIndex(pd.to_datetime(origin_dates))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("History dates must be sorted and unique")
    if origins.has_duplicates or not origins.is_monotonic_increasing or not origins.isin(dates).all():
        raise ValueError("Origins must be sorted, unique and present in history")
    return origins
