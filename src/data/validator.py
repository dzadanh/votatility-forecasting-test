from __future__ import annotations

import warnings

import pandas as pd

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
DATE_ALIASES = ("time", "datetime", "timestamp")
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]


def validate_data(data: pd.DataFrame, *, clean: bool = False) -> pd.DataFrame:
    """Normalize the OHLCV schema and validate values without mutating input.

    With clean=True, rows with missing/unparseable values and duplicate dates
    are removed (keeping the final duplicate). Invalid OHLC relationships,
    nonpositive prices, and negative volume always raise ValueError.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data phải là pandas DataFrame.")
    if data.empty:
        raise ValueError("Dataset không có dòng dữ liệu nào.")

    result = data.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if "date" not in result.columns:
        alias = next((name for name in DATE_ALIASES if name in result.columns), None)
        if alias:
            result = result.rename(columns={alias: "date"})
    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}")

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    invalid = result[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if invalid.any():
        count = int(invalid.sum())
        if not clean:
            raise ValueError(f"Có {count} dòng thiếu hoặc sai định dạng ngày/giá trị OHLCV.")
        warnings.warn(f"Đã loại {count} dòng thiếu hoặc sai định dạng ngày/giá trị OHLCV.", stacklevel=2)
        result = result.loc[~invalid].copy()
    if result.empty:
        raise ValueError("Không còn dòng hợp lệ sau khi làm sạch.")

    invalid_ohlc = (
        (result["high"] < result[["open", "close"]].max(axis=1))
        | (result["low"] > result[["open", "close"]].min(axis=1))
        | (result["high"] < result["low"])
    )
    invalid_rows = result.loc[invalid_ohlc, ["date", "open", "high", "low", "close"]]
    if invalid_ohlc.any():
        raise ValueError(f"Có {int(invalid_ohlc.sum())} dòng vi phạm quan hệ OHLC." + invalid_rows.to_string(index=False))
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Các giá trị OHLC phải lớn hơn 0."+ result.loc[(result[["open", "high", "low", "close"]] <= 0).any(axis=1), ["date", "open", "high", "low", "close"]].to_string(index=False))
    if (result["volume"] < 0).any():
        raise ValueError("Volume không được âm." + result.loc[result["volume"] < 0, ["date", "volume"]].to_string(index=False))

    duplicate = result["date"].duplicated(keep="last")
    if duplicate.any():
        count = int(duplicate.sum())
        if not clean:
            raise ValueError(f"Có {count} ngày giao dịch bị trùng.")
        warnings.warn(f"Đã loại {count} dòng trùng ngày (giữ bản ghi cuối).", stacklevel=2)
        result = result.loc[~duplicate].copy()
    if not clean and not result["date"].is_monotonic_increasing:
        raise ValueError("Dữ liệu phải được sắp xếp theo ngày tăng dần.")
    return result.sort_values("date").reset_index(drop=True)
