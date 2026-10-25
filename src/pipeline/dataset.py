"""Offline dataset adapter with an expanding TimeSeriesSplit and final holdout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import ROOT
from src.data.preprocessing import clean_data, preprocess_data
from src.features.builder import build_features
from src.features.target import add_future_volatility_target


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    train: pd.DataFrame
    validation_folds: tuple[tuple[pd.DataFrame, pd.DataFrame], ...]
    test: pd.DataFrame
    features: tuple[str, ...]
    target: str
    horizon: int

    @property
    def validation(self) -> pd.DataFrame:
        """All non-overlapping validation windows in chronological order."""
        return pd.concat([validation for _, validation in self.validation_folds], ignore_index=True)


def prepare_dataset(ticker: str, config: dict, csv_path: str | Path | None = None) -> Dataset:
    """Build causal features, expanding validation folds and a purged final test."""
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
    test_size = int(np.ceil(n * config["split"]["test"]))
    test_start = n - test_size
    n_splits = int(config["split"]["n_splits"])
    validation_size = int(config["split"]["test_size"])
    gap = int(config["split"]["gap"])
    development_end = test_start - gap
    if development_end <= 0:
        raise ValueError("Not enough labeled rows for the final test gap")

    development = labeled.iloc[:development_end]
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=validation_size, gap=gap, max_train_size=None)
    folds = []
    try:
        for train_indices, validation_indices in cv.split(development):
            folds.append((development.iloc[train_indices].copy(), development.iloc[validation_indices].copy()))
    except ValueError as exc:
        raise ValueError("Not enough labeled rows for configured TimeSeriesSplit folds and final test") from exc

    train = development.copy()
    test = labeled.iloc[test_start:].copy()
    if not len(train) or not len(test) or len(folds) != n_splits:
        raise ValueError("Not enough labeled rows after walk-forward partitioning")

    frame["partition"] = "unused"
    frame["validation_fold"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[frame.date.isin(train.date), "partition"] = "train"
    frame.loc[frame.date.isin(test.date), "partition"] = "test"
    frame.loc[frame.date.isin(labeled.iloc[development_end:test_start].date), "partition"] = "gap"
    for fold_index, (_, validation) in enumerate(folds, start=1):
        frame.loc[frame.date.isin(validation.date), "validation_fold"] = fold_index

    return Dataset(frame, train, tuple(folds), test, features, target, horizon)


def valid_history(dataset: Dataset) -> pd.DataFrame:
    """Keep causal feature rows, including latest rows without target labels."""
    return dataset.frame.dropna(subset=list(dataset.features)).copy().reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Select the ordered feature schema and reject nonfinite values."""
    if any(x.startswith("target_") or x == "date" for x in features):
        raise ValueError("Target/date cannot be input features")
    if any(x not in frame.columns for x in features):
        raise ValueError(f"Missing features: {sorted(set(features) - set(frame.columns))}")
    x = frame.loc[:, features]
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN or infinity")
    return x
