"""Small model smoke tests, persistence and causal origin behavior."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.evaluation.warning_metrics import high_warning_metrics
from src.features.builder import build_features
from src.features.target import add_future_volatility_target
from src.models.garch_model import GARCHModel
from src.models.lstm_model import LSTMModel, make_sequences
from src.models.xgboost_model import XGBoostModel
from src.pipeline.dataset import feature_matrix
from src.risk.warning_engine import WarningEngine


@pytest.fixture
def frames(synthetic_ohlcv):
    data = add_future_volatility_target(build_features(synthetic_ohlcv))
    features = tuple(load_config()["data"]["features"])
    history = data.dropna(subset=list(features)).reset_index(drop=True)
    train = history.iloc[:80].copy()
    val = history.iloc[85:105].copy()
    return features, history, train, val


def test_garch_causal_and_persistent(frames, tmp_path):
    _, history, train, val = frames
    model = GARCHModel().fit(train)
    dates = val.date.iloc[:3]
    expected = model.predict(history, dates)
    later = history.copy()
    later.loc[later.date > dates.max(), "return_1d"] = 1000
    pd.testing.assert_frame_equal(expected, model.predict(later, dates))
    assert (expected.predicted_volatility >= 0).all()
    path = tmp_path / "g.joblib"
    model.save(path)
    pd.testing.assert_frame_equal(expected, GARCHModel.load(path).predict(history, dates))


def test_xgboost_schema_and_persistence(frames, tmp_path):
    features, history, train, val = frames
    model = XGBoostModel(features, "target_volatility_5d", 5,
                         {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}).fit(train)
    forecast = model.predict(history, val.date)
    assert len(forecast) == len(val) and np.isfinite(forecast.predicted_volatility).all()
    with pytest.raises(ValueError):
        model.predict(history.drop(columns=[features[0]]), val.date)
    path = tmp_path / "x.joblib"
    model.save(path)
    pd.testing.assert_frame_equal(forecast, XGBoostModel.load(path).predict(history, val.date))


def test_lstm_train_only_scaling_and_smoke(frames, tmp_path):
    features, history, train, val = frames
    model = LSTMModel(features, "target_volatility_5d", 5, 10,
                      {"hidden_size": 8, "epochs": 2, "batch_size": 16, "patience": 2}, 42).fit(train, val, history)
    np.testing.assert_allclose(model.scaler.mean_, feature_matrix(train, features).mean().to_numpy())
    dates, sequences = make_sequences(history, val.date, features, 10, model.scaler)
    assert len(dates) == len(val) and sequences.shape == (len(val), 10, len(features))
    forecast = model.predict(history, val.date)
    assert (forecast.predicted_volatility >= 0).all()
    path = tmp_path / "l.joblib"
    model.save(path)
    pd.testing.assert_frame_equal(forecast, LSTMModel.load(path).predict(history, val.date))


def test_warning_boundaries_and_metrics(tmp_path):
    engine = WarningEngine(.5, .75).fit([0, 1, 2, 3, 4])
    assert engine.classify([engine.warning_threshold - .01, engine.warning_threshold,
                            engine.high_threshold]) .tolist() == ["NORMAL", "WARNING", "HIGH"]
    path = tmp_path / "w.json"
    engine.save(path)
    assert WarningEngine.load(path).high_threshold == engine.high_threshold
    score = high_warning_metrics([0, 3, 4, 1], [0, 4, 0, 1], 3)
    assert score["confusion_matrix"] == [[2, 0], [1, 1]]
    assert high_warning_metrics([0, 1], [0, 1], 3)["support"] == 0
    with pytest.raises(ValueError):
        WarningEngine().fit([1, 1, 1])
