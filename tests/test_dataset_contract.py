"""Feature scale, future target and chronological purge contracts."""
import numpy as np
import pytest
from src.config import load_config
from src.features.builder import build_features
from src.features.target import add_future_volatility_target
from src.pipeline import dataset as ds


def test_feature_target_scale_and_alignment(synthetic_ohlcv):
    frame = add_future_volatility_target(build_features(synthetic_ohlcv))
    assert frame.loc[1, "return_1d"] == pytest.approx(100 * np.log(frame.loc[1, "close"] / frame.loc[0, "close"]))
    assert frame.loc[70, "target_volatility_5d"] == pytest.approx(frame.loc[71:75, "return_1d"].std(ddof=1))
    assert frame["target_volatility_5d"].tail(5).isna().all()


def test_purged_split_and_no_target_features(monkeypatch, synthetic_ohlcv):
    monkeypatch.setattr(ds, "preprocess_data", lambda ticker, save=False: synthetic_ohlcv)
    data = ds.prepare_dataset("TEST", load_config())
    assert data.train.date.max() < data.validation.date.min() < data.test.date.min()
    all_dates = data.frame.date.reset_index(drop=True)
    train_end = all_dates[all_dates == data.train.date.max()].index[0]
    validation_start = all_dates[all_dates == data.validation.date.min()].index[0]
    validation_end = all_dates[all_dates == data.validation.date.max()].index[0]
    test_start = all_dates[all_dates == data.test.date.min()].index[0]
    assert validation_start - train_end > data.horizon
    assert test_start - validation_end > data.horizon
    assert not any(name.startswith("target_") for name in data.features)
    assert list(ds.feature_matrix(data.train, data.features)) == list(data.features)
    assert data.frame[data.target].tail(5).isna().all()


def test_forbidden_features(synthetic_ohlcv):
    with pytest.raises(ValueError):
        ds.feature_matrix(synthetic_ohlcv, ("close", "target_future"))


def test_explicit_local_csv(monkeypatch, synthetic_ohlcv, tmp_path):
    csv = tmp_path / "sample.csv"
    synthetic_ohlcv.to_csv(csv, index=False)
    monkeypatch.setattr(ds, "preprocess_data", lambda *args, **kwargs: pytest.fail("Default loader should not be called"))
    data = ds.prepare_dataset("TEST", load_config(), csv_path=csv)
    assert len(data.frame) == len(synthetic_ohlcv)
