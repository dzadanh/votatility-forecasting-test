from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(
    data: pd.DataFrame,
    ma_windows: tuple[int, ...] = (5, 10, 20, 50),
    volatility_windows: tuple[int, ...] = (5, 20),
    return_windows: tuple[int, ...] = (5, 10, 20),
    volume_window: int = 20,
) -> pd.DataFrame:
    """Tạo return và feature chỉ dùng dữ liệu hiện tại hoặc quá khứ.

    Return log được biểu diễn theo phần trăm (x100), phù hợp quy ước GARCH.
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Thiếu cột đầu vào: {', '.join(sorted(missing))}")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("Dữ liệu phải được sắp xếp theo ngày tăng dần.")
    if data["date"].duplicated().any():
        raise ValueError("Dữ liệu có ngày giao dịch bị trùng.")
    for window in (*ma_windows, *volatility_windows, *return_windows, volume_window):
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Các rolling window phải là số nguyên dương.")

    result = data.copy()
    returns = np.log(result["close"] / result["close"].shift(1)) * 100
    result["return_1d"] = returns
    for window in return_windows:
        result[f"return_{window}d"] = np.log(result["close"] / result["close"].shift(window)) * 100
    for window in ma_windows:
        moving_average = result["close"].rolling(window, min_periods=window).mean()
        result[f"close_to_ma_{window}"] = result["close"] / moving_average - 1
    for window in volatility_windows:
        result[f"rolling_volatility_{window}"] = returns.rolling(window, min_periods=window).std()

    previous_volume = result["volume"].shift(1)
    result["volume_change"] = (result["volume"] - previous_volume) / previous_volume.replace(0, np.nan)
    volume_average = result["volume"].rolling(volume_window, min_periods=volume_window).mean()
    result[f"volume_to_ma_{volume_window}"] = result["volume"] / volume_average.replace(0, np.nan) - 1
    result["high_low_range"] = (result["high"] - result["low"]) / result["close"]
    result["open_close_range"] = (result["close"] - result["open"]) / result["open"]
    return result.replace([np.inf, -np.inf], np.nan)
