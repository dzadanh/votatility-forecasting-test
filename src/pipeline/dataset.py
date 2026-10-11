"""Offline dataset adapter and leakage-safe chronological partitions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import ROOT
from src.data.preprocessing import clean_data, preprocess_data
from src.features.builder import build_features
from src.features.target import add_future_volatility_target


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    features: tuple[str, ...]
    target: str
    horizon: int


def prepare_dataset(ticker: str, config: dict, csv_path: str | Path | None = None) -> Dataset:
    """Recompute features from raw CSV, retain unlabeled latest rows, then purge boundaries."""
    horizon = int(config["data"]["horizon"])
    features = tuple(config["data"]["features"])
    target = f"target_volatility_{horizon}d"
    raw_dir = Path(config["paths"]["raw"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if csv_path is not None:
        explicit_path = Path(csv_path)
        if not explicit_path.is_absolute():
            explicit_path = ROOT / explicit_path
        cleaned = clean_data(pd.read_csv(explicit_path))
    elif raw_dir.resolve() != (ROOT / "data" / "raw").resolve():
        cleaned = clean_data(pd.read_csv(raw_dir / f"{ticker.lower()}.csv"))
    else:
        cleaned = preprocess_data(ticker, save=False)
    frame = add_future_volatility_target(build_features(cleaned), horizon)
    frame["date"] = pd.to_datetime(frame["date"])
    if not frame.date.is_monotonic_increasing or frame.date.duplicated().any():
        raise ValueError("Dates must be sorted and unique")
    absent = set(features + ("return_1d", target)) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing dataset columns: {sorted(absent)}")
    frame = frame.replace([np.inf, -np.inf], np.nan).copy()
    frame["ticker"] = ticker.upper()
    labeled = frame.dropna(subset=[*features, target]).copy()
    n = len(labeled)
    a = int(n * config["split"]["train"])
    b = a + int(n * config["split"]["validation"])
    # Each origin's target covers the next h observed sessions. A gap of h
    # labeled origins ensures previous targets end before the next partition.
    train = labeled.iloc[:max(0, a - horizon)].copy()
    validation = labeled.iloc[a:max(a, b - horizon)].copy()
    test = labeled.iloc[b:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Not enough labeled rows after horizon purging")
    return Dataset(frame, train, validation, test, features, target, horizon)


def valid_history(dataset: Dataset) -> pd.DataFrame:
    """Keep all causal feature rows, including recent rows without target labels."""
    return dataset.frame.dropna(subset=list(dataset.features)).copy().reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Select the exact ordered feature schema and reject nonfinite values."""
    if any(x.startswith("target_") or x == "date" for x in features):
        raise ValueError("Target/date cannot be input features")
    if any(x not in frame.columns for x in features):
        raise ValueError(f"Missing features: {sorted(set(features) - set(frame.columns))}")
    x = frame.loc[:, features]
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN or infinity")
    return x
