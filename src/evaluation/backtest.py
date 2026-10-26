"""Align date-indexed historical forecasts before comparative evaluation."""
from __future__ import annotations

import pandas as pd


def align_forecasts(forecasts: list[pd.DataFrame], truth: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return matched origin dates for every provided model and known truth."""
    if not forecasts:
        raise ValueError("At least one forecast is required")
    dates = set(pd.to_datetime(truth["date"]))
    for forecast in forecasts:
        dates &= set(pd.to_datetime(forecast["date"]))
    dates = sorted(dates)
    if not dates:
        raise ValueError("No common forecast origins")
    actual = truth.set_index("date").loc[dates, target]
    rows = []
    for forecast in forecasts:
        matched = forecast.set_index("date").loc[dates]
        for date, row in matched.iterrows():
            rows.append({"date": date, "model": row["model"], "actual_volatility": float(actual.loc[date]),
                         "predicted_volatility": float(row["predicted_volatility"]),
                         "raw_prediction": float(row.get("raw_prediction", row["predicted_volatility"])),
                         "high_probability": float(row["high_probability"]) if "high_probability" in row else float("nan")})
    return pd.DataFrame(rows).sort_values(["date", "model"]).reset_index(drop=True)
