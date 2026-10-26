"""XGBoost regression wrapper with strict feature schema and target metadata."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .base import UNIT, forecast_frame, validate_origins
from src.pipeline.dataset import feature_matrix


class XGBoostModel:
    name = "xgboost"

    def __init__(self, features: tuple[str, ...], target: str, horizon: int, params: dict | None = None, seed: int = 42):
        self.features, self.target, self.horizon, self.seed = tuple(features), target, horizon, seed
        self.unit = UNIT
        self.params = dict(params or {})
        self.estimator = None

    def fit(self, training_frame: pd.DataFrame) -> "XGBoostModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost requires 'xgboost'; install requirements.txt") from exc
        x = feature_matrix(training_frame, self.features)
        y = training_frame[self.target].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Training target must be finite")
        # Grid-level parallelism is configured separately; keep each estimator single-threaded
        # by default to avoid nested CPU oversubscription during cross-validation.
        estimator_params = {"n_jobs": 1, **self.params}
        if "tuning" in estimator_params:
            raise ValueError("Tuning configuration belongs in the training pipeline, not XGBRegressor params")
        self.estimator = XGBRegressor(
            objective="reg:squarederror", random_state=self.seed, **estimator_params
        )
        self.estimator.fit(x, y)
        return self

    def predict_raw(self, history_frame: pd.DataFrame, origin_dates) -> pd.DataFrame:
        if self.estimator is None:
            raise ValueError("XGBoost is not fitted")
        origins = validate_origins(history_frame, origin_dates)
        # Extra feature columns are not passed to the estimator; its input is exactly the whitelist.
        rows = history_frame.set_index("date").loc[origins].reset_index()
        x = feature_matrix(rows, self.features)
        raw = np.asarray(self.estimator.predict(x), dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("XGBoost produced nonfinite predictions")
        return pd.DataFrame({"date": origins, "raw_prediction": raw})

    def predict(self, history_frame: pd.DataFrame, origin_dates) -> pd.DataFrame:
        raw = self.predict_raw(history_frame, origin_dates)
        result = forecast_frame(raw["date"], self.name, np.maximum(raw["raw_prediction"], 0))
        result["raw_prediction"] = raw["raw_prediction"].to_numpy()
        return result

    def save(self, path: str | Path) -> None:
        if self.estimator is None:
            raise ValueError("XGBoost is not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("Artifact is not an XGBoostModel")
        return model
