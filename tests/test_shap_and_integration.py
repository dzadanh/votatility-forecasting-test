"""SHAP additivity and optional complete offline ACB smoke run."""
import numpy as np
import pytest
import yaml
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
    assert len(result["metadata"]["lstm_training"]["history"]) >= 1
    assert result["metadata"]["lstm_training"]["best_epoch"] >= 1
    latest = predict_latest("ACB", "xgboost", artifact_dir=tmp_path)
    assert latest["ground_truth_available"] is False
    assert np.isfinite(latest["predicted_volatility"])


@pytest.mark.slow
def test_high_head_pipeline_saves_usable_alert_model(tmp_path):
    config = load_config()
    config["models"]["lstm"]["high_loss_weight"] = 0.5
    path = tmp_path / "high_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_training("ACB", "lstm", fast=True, artifact_dir=tmp_path,
                          config_path=path)
    assert result["metadata"]["alert_calibration"]["lstm"]["score_kind"] == "high_probability"
    latest = predict_latest("ACB", "lstm", artifact_dir=tmp_path,
                            config_path=path)
    assert 0 <= latest["high_probability"] <= 1
    assert latest["risk_level"] in {"NORMAL", "WARNING", "HIGH"}
