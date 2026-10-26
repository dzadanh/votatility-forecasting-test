"""Latest offline prediction from persisted model and fresh local raw CSV."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import ROOT, load_config
from src.models.base import UNIT
from src.models.garch_model import GARCHModel
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel
from src.pipeline.dataset import prepare_dataset, valid_history
from src.risk.warning_engine import WarningEngine

LOADERS = {"garch": GARCHModel, "xgboost": XGBoostModel, "lstm": LSTMModel}


def predict_latest(ticker="ACB", model="xgboost", config_path=None, artifact_dir=None, csv_path=None) -> dict:
    """Reject incompatible saved schemas, then predict the newest causal origin."""
    if model not in LOADERS:
        raise ValueError(f"Unknown model: {model}")
    config = load_config(config_path or ROOT / "config.yaml")
    out = Path(artifact_dir or ROOT / config["paths"]["artifacts"]) / ticker.upper()
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    default_source = str(Path(config["paths"]["raw"]) / f"{ticker.lower()}.csv")
    source = csv_path or (metadata.get("source_csv") if metadata.get("source_csv") != default_source else None)
    dataset = prepare_dataset(ticker, config, source)
    expected = {"ticker": ticker.upper(), "target": dataset.target, "horizon": dataset.horizon,
                "features": list(dataset.features), "unit": UNIT}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved model metadata is incompatible with current data schema/configuration")
    if model not in metadata.get("model_settings", {}):
        raise ValueError(f"Model {model} was not included in the saved experiment")
    fitted = LOADERS[model].load(out / f"{model}.joblib")
    if (getattr(fitted, "horizon", None) != dataset.horizon or
        getattr(fitted, "target", None) != dataset.target or
        getattr(fitted, "unit", None) != UNIT or
        (model != "garch" and getattr(fitted, "features", None) != dataset.features)):
        raise ValueError("Saved model schema is incompatible")
    if model == "lstm" and metadata.get("sequence_length") != fitted.sequence_length:
        raise ValueError("Saved LSTM sequence length does not match its metadata")
    history = valid_history(dataset)
    latest = history["date"].max()
    prediction = fitted.predict(history, [latest]).iloc[0].to_dict()
    prediction["date"] = str(latest.date())
    prediction["ticker"] = ticker.upper()
    prediction["risk_level"] = WarningEngine.load(out / "warning.json").classify(
        prediction["predicted_volatility"], model=model,
        high_probability=prediction.get("high_probability"))
    prediction["ground_truth_available"] = False
    return prediction
