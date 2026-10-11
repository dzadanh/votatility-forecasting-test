"""SHAP additivity and optional complete offline ACB smoke run."""
import numpy as np
import pytest
from src.config import load_config
from src.explainability.shap_explainer import explain_xgboost
from src.features.builder import build_features
from src.features.target import add_future_volatility_target
from src.models.xgboost_model import XGBoostModel
from src.pipeline.train_pipeline import run_training
from src.pipeline.predict_pipeline import predict_latest


def test_shap_additivity(synthetic_ohlcv):
    frame = add_future_volatility_target(build_features(synthetic_ohlcv))
    features = tuple(load_config()["data"]["features"])
    frame = frame.dropna(subset=[*features, "target_volatility_5d"])
    model = XGBoostModel(features, "target_volatility_5d", 5, {"n_estimators": 5, "n_jobs": 1}).fit(frame.iloc[:75])
    explanation = explain_xgboost(model, frame.iloc[75:78])
    assert len(explanation["global_importance"]) == len(features)
    row = explanation["local"][0]
    assert row["base_value"] + sum(row["shap_values"].values()) == pytest.approx(row["raw_model_output"], abs=1e-3)


@pytest.mark.slow
def test_offline_acb_end_to_end(tmp_path):
    result = run_training("ACB", "all", fast=True, artifact_dir=tmp_path)
    assert result["metadata"]["matched_test_count"] > 0
    assert set(result["metrics"]) == {"garch", "xgboost", "lstm"}
    latest = predict_latest("ACB", "xgboost", artifact_dir=tmp_path)
    assert latest["ground_truth_available"] is False
    assert np.isfinite(latest["predicted_volatility"])
