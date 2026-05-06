import numpy as np
import pandas as pd

from app.config import DATE_COL, TARGET_COL, WEEKLY_FREQ
from app.models.base import ForecastModel


class LstmForecaster(ForecastModel):
    name = "LSTM"

    def __init__(self, lookback: int = 30, epochs: int = 50):
        self.lookback = lookback
        self.epochs = epochs
        self._model = None
        self._min = 0.0
        self._max = 1.0

    def _scale(self, values):
        return (values - self._min) / max(self._max - self._min, 1e-9)

    def _inverse_scale(self, values):
        return values * max(self._max - self._min, 1e-9) + self._min

    def fit(self, train_df: pd.DataFrame) -> "LstmForecaster":
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        values = train_df[TARGET_COL].to_numpy(dtype=float)
        self._min = float(np.min(values))
        self._max = float(np.max(values))
        scaled = self._scale(values)

        x, y = [], []
        for i in range(self.lookback, len(scaled)):
            x.append(scaled[i - self.lookback : i])
            y.append(scaled[i])
        x = np.asarray(x).reshape(-1, self.lookback, 1)
        y = np.asarray(y)

        model = Sequential(
            [
                LSTM(64, input_shape=(self.lookback, 1)),
                Dropout(0.15),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(x, y, epochs=self.epochs, batch_size=16, verbose=0)
        self._model = model
        return self

    def predict(self, periods: int, history_df: pd.DataFrame) -> pd.Series:
        values = list(self._scale(history_df[TARGET_COL].to_numpy(dtype=float)))
        preds = []
        for _ in range(periods):
            x = np.asarray(values[-self.lookback:]).reshape(1, self.lookback, 1)
            pred_scaled = float(self._model.predict(x, verbose=0)[0][0])
            values.append(pred_scaled)
            preds.append(max(float(self._inverse_scale(pred_scaled)), 0.0))
        dates = pd.date_range(
            start=history_df[DATE_COL].max() + pd.offsets.Week(weekday=6),
            periods=periods,
            freq=WEEKLY_FREQ,
        )
        return pd.Series(preds, index=dates)

