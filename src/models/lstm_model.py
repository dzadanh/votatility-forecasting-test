"""Small CPU LSTM with training-only scaling and causal history sequences."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .base import UNIT, forecast_frame, validate_origins
from src.pipeline.dataset import feature_matrix


def make_sequences(history: pd.DataFrame, origins, features: tuple[str, ...], length: int, scaler):
    """Create windows ending at each origin; return valid dates and arrays."""
    selected = feature_matrix(history, features)
    scaled = scaler.transform(selected)
    dates = pd.DatetimeIndex(history["date"])
    positions = dates.get_indexer(pd.DatetimeIndex(pd.to_datetime(origins)))
    good = positions >= length - 1
    return dates[positions[good]], np.stack([scaled[p-length+1:p+1] for p in positions[good]]).astype("float32") if good.any() else np.empty((0, length, len(features)), dtype="float32")


class LSTMModel:
    name = "lstm"

    def __init__(self, features: tuple[str, ...], target: str, horizon: int, sequence_length: int = 20,
                 params: dict | None = None, seed: int = 42):
        self.features, self.target, self.horizon = tuple(features), target, horizon
        self.sequence_length, self.params, self.seed = sequence_length, dict(params or {}), seed
        self.unit = UNIT
        self.scaler = None
        self.state = None
        self.validation_loss = None

    def _network(self):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("LSTM requires 'torch'; install requirements.txt") from exc
        hidden = int(self.params.get("hidden_size", 32))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(len(self_features), hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1]).squeeze(-1)

        self_features = self.features
        return Net()

    def fit(self, training_frame: pd.DataFrame, validation_frame: pd.DataFrame, history_frame: pd.DataFrame) -> "LSTMModel":
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError("LSTM requires torch and scikit-learn") from exc
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler = StandardScaler().fit(feature_matrix(training_frame, self.features))
        history = history_frame.loc[history_frame["date"] <= validation_frame["date"].max()].copy()
        train_dates, train_x = make_sequences(history, training_frame["date"], self.features, self.sequence_length, self.scaler)
        val_dates, val_x = make_sequences(history, validation_frame["date"], self.features, self.sequence_length, self.scaler)
        if not len(train_x) or not len(val_x):
            raise ValueError("Insufficient LSTM sequence history")
        train_y = training_frame.set_index("date").loc[train_dates, self.target].to_numpy(dtype="float32")
        val_y = validation_frame.set_index("date").loc[val_dates, self.target].to_numpy(dtype="float32")
        train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
        val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)
        net = self._network()
        optimizer = torch.optim.Adam(net.parameters(), lr=float(self.params.get("learning_rate", .001)),
                                     weight_decay=float(self.params.get("weight_decay", 1e-4)))
        loss_fn = torch.nn.MSELoss()
        batch_size = int(self.params.get("batch_size", 32))
        epochs = int(self.params.get("epochs", 25))
        patience = int(self.params.get("patience", 5))
        best, stale = float("inf"), 0
        for _ in range(epochs):
            net.train()
            for start in range(0, len(train_x), batch_size):
                optimizer.zero_grad()
                loss = loss_fn(net(train_x[start:start+batch_size]), train_y[start:start+batch_size])
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                score = float(loss_fn(net(val_x), val_y))
            if score < best - 1e-8:
                best, stale = score, 0
                self.state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        self.validation_loss = best
        return self

    def predict(self, history_frame: pd.DataFrame, origin_dates) -> pd.DataFrame:
        if self.state is None or self.scaler is None:
            raise ValueError("LSTM is not fitted")
        import torch
        origins = validate_origins(history_frame, origin_dates)
        dates, x = make_sequences(history_frame, origins, self.features, self.sequence_length, self.scaler)
        if len(dates) != len(origins):
            raise ValueError("Insufficient history for one or more LSTM origins")
        net = self._network()
        net.load_state_dict(self.state)
        net.eval()
        with torch.no_grad():
            raw = net(torch.from_numpy(x)).numpy()
        result = forecast_frame(dates, self.name, np.maximum(raw, 0))
        result["raw_prediction"] = raw
        return result

    def save(self, path: str | Path) -> None:
        if self.state is None:
            raise ValueError("LSTM is not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("Artifact is not an LSTMModel")
        return model
