"""Fixed-parameter GARCH(1,1) baseline for daily percentage-point returns."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .base import UNIT, forecast_frame, validate_origins


class GARCHModel:
    """Fit on training returns and filter later observed returns causally.

    The square root of mean future conditional variance approximates the
    sample standard deviation target; these are not mathematically identical.
    """

    name = "garch"

    def __init__(self, horizon: int = 5, mean: str = "Zero", distribution: str = "normal"):
        if horizon < 2 or mean != "Zero" or distribution != "normal":
            raise ValueError("This baseline requires horizon >= 2, Zero mean and normal errors")
        self.horizon, self.mean, self.distribution = horizon, mean, distribution
        self.target = f"target_volatility_{horizon}d"
        self.unit = UNIT
        self.params: dict[str, float] | None = None
        self.initial_variance: float | None = None
        self.train_end: pd.Timestamp | None = None

    def fit(self, training_frame: pd.DataFrame) -> "GARCHModel":
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError("GARCH requires the 'arch' package; install requirements.txt") from exc
        y = training_frame["return_1d"].to_numpy(dtype=float)
        if len(y) < 30 or not np.isfinite(y).all() or np.var(y) <= 0:
            raise ValueError("GARCH requires at least 30 finite, variable training returns")
        result = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
            disp="off", show_warning=False)
        self.params = {k: float(result.params[k]) for k in ("omega", "alpha[1]", "beta[1]")}
        self.initial_variance = float(np.var(y, ddof=1))
        self.train_end = pd.Timestamp(training_frame["date"].max())
        return self

    def predict(self, history_frame: pd.DataFrame, origin_dates) -> pd.DataFrame:
        if self.params is None or self.initial_variance is None:
            raise ValueError("GARCH is not fitted")
        origins = validate_origins(history_frame, origin_dates)
        returns = history_frame["return_1d"].to_numpy(dtype=float)
        if not np.isfinite(returns).all():
            raise ValueError("GARCH history returns must be finite")
        omega, alpha, beta = (self.params[k] for k in ("omega", "alpha[1]", "beta[1]"))
        variance = self.initial_variance
        estimates = []
        for ret in returns:
            variance = max(omega + alpha * ret * ret + beta * variance, 1e-12)
            future = variance
            conditional = [future]
            for _ in range(self.horizon - 1):
                future = max(omega + (alpha + beta) * future, 1e-12)
                conditional.append(future)
            estimates.append(float(np.sqrt(np.mean(conditional))))
        lookup = pd.Series(estimates, index=pd.DatetimeIndex(history_frame["date"]))
        return forecast_frame(origins, self.name, lookup.loc[origins].to_numpy())

    def save(self, path: str | Path) -> None:
        if self.params is None:
            raise ValueError("GARCH is not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "GARCHModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("Artifact is not a GARCHModel")
        return model
