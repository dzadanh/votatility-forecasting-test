"""Train-quantile risk labels, independent of the forecasting model."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class WarningEngine:
    labels = ("NORMAL", "WARNING", "HIGH")

    def __init__(self, q_warning: float = .70, q_high: float = .90):
        if not 0 < q_warning < q_high < 1:
            raise ValueError("Require 0 < q_warning < q_high < 1")
        self.q_warning, self.q_high = q_warning, q_high
        self.warning_threshold: float | None = None
        self.high_threshold: float | None = None
        self.model_high_thresholds: dict[str, float] = {}
        self.model_high_probability_thresholds: dict[str, float] = {}

    def fit(self, train_targets) -> "WarningEngine":
        values = np.asarray(train_targets, dtype=float).reshape(-1)
        if not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Training targets must be nonempty, finite and nonnegative")
        self.warning_threshold, self.high_threshold = map(float, np.quantile(values, [self.q_warning, self.q_high]))
        if self.warning_threshold >= self.high_threshold:
            raise ValueError("Training quantiles coincide; risk levels are degenerate")
        return self

    def set_model_high_threshold(self, model: str, threshold: float) -> None:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Model HIGH threshold must be finite and nonnegative")
        self.model_high_thresholds[str(model)] = float(threshold)

    def set_model_high_probability_threshold(self, model: str, threshold: float) -> None:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Model HIGH probability threshold must be between zero and one")
        self.model_high_probability_thresholds[str(model)] = float(threshold)

    def classify(self, values, model: str | None = None, high_probability=None):
        if self.warning_threshold is None or self.high_threshold is None:
            raise ValueError("WarningEngine is not fitted")
        scalar = np.isscalar(values)
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError("Risk values must be finite and nonnegative")
        high_threshold = self.model_high_thresholds.get(model, self.high_threshold) if model else self.high_threshold
        if model in self.model_high_probability_thresholds:
            if high_probability is None:
                raise ValueError("HIGH probability is required for this model")
            probability = np.asarray(high_probability, dtype=float)
            if probability.shape != arr.shape or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
                raise ValueError("HIGH probability must align with values and be between zero and one")
            high = probability >= self.model_high_probability_thresholds[model]
        else:
            high = arr >= high_threshold
        result = np.where(high, "HIGH", np.where(arr >= self.warning_threshold, "WARNING", "NORMAL"))
        return str(result.item()) if scalar else result

    def save(self, path: str | Path) -> None:
        if self.high_threshold is None:
            raise ValueError("WarningEngine is not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"q_warning": self.q_warning, "q_high": self.q_high,
            "warning_threshold": self.warning_threshold, "high_threshold": self.high_threshold,
            "model_high_thresholds": self.model_high_thresholds,
            "model_high_probability_thresholds": self.model_high_probability_thresholds}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WarningEngine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        engine = cls(data["q_warning"], data["q_high"])
        engine.warning_threshold, engine.high_threshold = data["warning_threshold"], data["high_threshold"]
        engine.model_high_thresholds = {str(k): float(v) for k, v in data.get("model_high_thresholds", {}).items()}
        engine.model_high_probability_thresholds = {str(k): float(v) for k, v in data.get("model_high_probability_thresholds", {}).items()}
        return engine
