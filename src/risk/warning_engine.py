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

    def fit(self, train_targets) -> "WarningEngine":
        values = np.asarray(train_targets, dtype=float).reshape(-1)
        if not len(values) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Training targets must be nonempty, finite and nonnegative")
        self.warning_threshold, self.high_threshold = map(float, np.quantile(values, [self.q_warning, self.q_high]))
        if self.warning_threshold >= self.high_threshold:
            raise ValueError("Training quantiles coincide; risk levels are degenerate")
        return self

    def classify(self, values):
        if self.warning_threshold is None or self.high_threshold is None:
            raise ValueError("WarningEngine is not fitted")
        scalar = np.isscalar(values)
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError("Risk values must be finite and nonnegative")
        result = np.where(arr >= self.high_threshold, "HIGH", np.where(arr >= self.warning_threshold, "WARNING", "NORMAL"))
        return str(result.item()) if scalar else result

    def save(self, path: str | Path) -> None:
        if self.high_threshold is None:
            raise ValueError("WarningEngine is not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"q_warning": self.q_warning, "q_high": self.q_high,
            "warning_threshold": self.warning_threshold, "high_threshold": self.high_threshold}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WarningEngine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        engine = cls(data["q_warning"], data["q_high"])
        engine.warning_threshold, engine.high_threshold = data["warning_threshold"], data["high_threshold"]
        return engine
