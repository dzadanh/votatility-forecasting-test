"""Selection must favor HIGH ranking without discarding forecast accuracy."""
import numpy as np
import pytest

from src.models.xgboost_tuning import _select_candidate


def test_high_ap_wins_only_within_rmse_guard():
    mse = np.array([1.0, 1.08, 1.30])
    ap = np.array([0.25, 0.60, 0.95])

    selected, best_rmse, limit, eligible = _select_candidate(mse, ap, .05)

    assert selected == 1
    assert eligible.tolist() == [0, 1]
    assert best_rmse == pytest.approx(1.0)
    assert limit == pytest.approx(1.05)


def test_nonfinite_validation_scores_fail_selection():
    with pytest.raises(ValueError, match="nonfinite"):
        _select_candidate(np.array([1.0, np.nan]), np.array([.3, .4]), .05)
