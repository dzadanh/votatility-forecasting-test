"""Small CPU LSTM with training-only scaling and causal history sequences."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None
        self.high_threshold: float | None = None

    def _network(self):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("LSTM requires 'torch'; install requirements.txt") from exc
        hidden = int(self.params.get("hidden_size", 32))
        high_head = float(self.params.get("high_loss_weight", 0)) > 0

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(len(self_features), hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)
                if high_head:
                    self.high_head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1]
                volatility = self.head(last).squeeze(-1)
                return (volatility, self.high_head(last).squeeze(-1)) if high_head else volatility

        self_features = self.features
        return Net()

    def fit(self, training_frame: pd.DataFrame, validation_frame: pd.DataFrame,
            history_frame: pd.DataFrame,
            final_training_frame: pd.DataFrame | None = None,
            high_threshold: float | None = None) -> "LSTMModel":
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
        high_weight = float(self.params.get("high_loss_weight", 0))
        if high_weight < 0:
            raise ValueError("high_loss_weight must be nonnegative")
        if high_weight and (high_threshold is None or not np.isfinite(high_threshold) or high_threshold < 0):
            raise ValueError("A fixed training HIGH threshold is required for the HIGH head")
        self.high_threshold = float(high_threshold) if high_threshold is not None else None
        train_high = (train_y >= high_threshold).astype("float32") if high_weight else None
        val_high = (val_y >= high_threshold).astype("float32") if high_weight else None
        train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
        val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)
        if high_weight:
            train_high, val_high = torch.from_numpy(train_high), torch.from_numpy(val_high)
        net = self._network()
        optimizer = torch.optim.Adam(net.parameters(), lr=float(self.params.get("learning_rate", .001)),
                                     weight_decay=float(self.params.get("weight_decay", 1e-4)))
        loss_fn = torch.nn.MSELoss()
        if high_weight:
            positives = float(train_high.sum())
            negatives = len(train_high) - positives
            pos_weight = max(1.0, negatives / positives) if positives else 1.0
            high_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        batch_size = int(self.params.get("batch_size", 32))
        epochs = int(self.params.get("epochs", 25))
        patience = int(self.params.get("patience", 5))
        best, stale = float("inf"), 0
        self.training_history = []
        self.best_epoch = None
        for epoch in range(1, epochs + 1):
            net.train()
            for start in range(0, len(train_x), batch_size):
                optimizer.zero_grad()
                output = net(train_x[start:start+batch_size])
                if high_weight:
                    volatility, logits = output
                    loss = (loss_fn(volatility, train_y[start:start+batch_size]) +
                            high_weight * high_loss_fn(logits, train_high[start:start+batch_size]))
                else:
                    loss = loss_fn(output, train_y[start:start+batch_size])
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                train_output, val_output = net(train_x), net(val_x)
                train_volatility = train_output[0] if high_weight else train_output
                val_volatility = val_output[0] if high_weight else val_output
                train_score = float(loss_fn(train_volatility, train_y))
                score = float(loss_fn(val_volatility, val_y))
                high_ap = (float(average_precision_score(val_high.numpy(),
                           torch.sigmoid(val_output[1]).numpy()))
                           if high_weight and 0 < int(val_high.sum()) < len(val_high) else None)
            self.training_history.append({"epoch": epoch, "train_mse": train_score,
                                          "validation_mse": score, "validation_high_ap": high_ap})
            # The HIGH head is selected by ranking quality when the stop window has both classes.
            selection = -high_ap if high_ap is not None else score
            if selection < best - 1e-8:
                best, stale = selection, 0
                self.best_epoch = epoch
                self.validation_loss = score
                self.state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if final_training_frame is not None:
            self.scaler = StandardScaler().fit(
                feature_matrix(final_training_frame, self.features)
            )
            final_history = history_frame.loc[
                history_frame["date"] <= final_training_frame["date"].max()
            ].copy()
            final_dates, final_x = make_sequences(
                final_history, final_training_frame["date"],
                self.features, self.sequence_length, self.scaler
            )
            final_y = final_training_frame.set_index("date").loc[
                final_dates, self.target
            ].to_numpy(dtype="float32")

            final_x = torch.from_numpy(final_x)
            final_y = torch.from_numpy(final_y)
            final_high = (final_y >= self.high_threshold).float() if high_weight else None

            torch.manual_seed(self.seed)
            net = self._network()
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=float(self.params.get("learning_rate", .001)),
                weight_decay=float(self.params.get("weight_decay", 1e-4)),
            )

            for _ in range(self.best_epoch):
                net.train()
                for start in range(0, len(final_x), batch_size):
                    end = start + batch_size
                    optimizer.zero_grad()
                    output = net(final_x[start:end])
                    if high_weight:
                        volatility, logits = output
                        loss = (
                            loss_fn(volatility, final_y[start:end])
                            + high_weight * high_loss_fn(logits, final_high[start:end])
                        )
                    else:
                        loss = loss_fn(output, final_y[start:end])
                    loss.backward()
                    optimizer.step()

            self.state = {
                k: v.detach().clone() for k, v in net.state_dict().items()
            }
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
            output = net(torch.from_numpy(x))
            raw = (output[0] if isinstance(output, tuple) else output).numpy()
        result = forecast_frame(dates, self.name, np.maximum(raw, 0))
        result["raw_prediction"] = raw
        if isinstance(output, tuple):
            result["high_probability"] = torch.sigmoid(output[1]).numpy()
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
