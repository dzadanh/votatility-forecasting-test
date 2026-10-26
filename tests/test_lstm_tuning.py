"""The tuning windows must keep future labels out of every earlier partition."""
import numpy as np
import pandas as pd
import pytest

from src.models.lstm_tuning import candidate_grid, expanding_folds
from src.pipeline.dataset import Dataset
from src.pipeline.lstm_walk_forward import walk_forward_windows


def test_expanding_folds_purge_and_exclude_test():
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=800, freq="B"),
        "feature": np.arange(800, dtype=float),
        "target_volatility_5d": np.ones(800),
    })
    dataset = Dataset(frame, frame.iloc[:650],
                      ((frame.iloc[:500], frame.iloc[505:650]),),
                      frame.iloc[655:], ("feature",), "target_volatility_5d", 5)
    folds = expanding_folds(dataset, count=3, score_size=80, stop_size=50)
    assert len(folds) == 3
    assert [len(fold.score) for fold in folds] == [80, 80, 80]
    assert [len(fold.train) for fold in folds] == sorted(len(fold.train) for fold in folds)
    for fold in folds:
        train_end = frame.index[frame.date == fold.train.date.max()][0]
        stop_start = frame.index[frame.date == fold.stop.date.min()][0]
        stop_end = frame.index[frame.date == fold.stop.date.max()][0]
        score_start = frame.index[frame.date == fold.score.date.min()][0]
        assert stop_start - train_end > dataset.horizon
        assert score_start - stop_end > dataset.horizon
        assert fold.score.date.max() <= dataset.validation.date.max()


def test_candidate_grid_keeps_user_epoch_limit():
    config = {"data": {"sequence_length": 20},
              "models": {"lstm": {"epochs": 500, "patience": 5, "hidden_size": 32}}}
    candidates = candidate_grid(config)
    assert all(candidate["params"]["epochs"] == 500 for candidate in candidates)
    assert {candidate["params"]["high_loss_weight"] for candidate in candidates} == {0.0, 0.25, 0.5, 1.0}
    assert any(candidate["sequence_length"] == 10 for candidate in candidates)


def test_walk_forward_uses_only_matured_labels():
    labeled = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=500, freq="B")})
    train, stop = walk_forward_windows(labeled, score_start=400, horizon=5, stop_size=120)
    assert len(stop) == 120
    assert train.index.max() + 5 < stop.index.min()
    assert stop.index.max() + 5 < 400
    with pytest.raises(ValueError):
        walk_forward_windows(labeled, score_start=100, horizon=5, stop_size=120)
