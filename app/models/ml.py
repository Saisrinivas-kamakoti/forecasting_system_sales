import numpy as np
import pandas as pd

from app.config import DATE_COL, TARGET_COL, WEEKLY_FREQ
from app.features import FEATURE_COLUMNS, build_supervised_features
from app.models.base import ForecastModel


class XGBoostForecaster(ForecastModel):
    name = "XGBoost"

    def __init__(self):
        self._model = None

    def fit(self, train_df: pd.DataFrame) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        features = build_supervised_features(train_df, target_col=TARGET_COL)
        self._model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )
        self._model.fit(features[FEATURE_COLUMNS], features[TARGET_COL])
        return self

    def predict(self, periods: int, history_df: pd.DataFrame) -> pd.Series:
        history = history_df[[DATE_COL, TARGET_COL]].copy().sort_values(DATE_COL)
        predictions = []
        dates = pd.date_range(
            start=history[DATE_COL].max() + pd.offsets.Week(weekday=6),
            periods=periods,
            freq=WEEKLY_FREQ,
        )
        for date in dates:
            candidate = pd.concat(
                [history, pd.DataFrame([{DATE_COL: date, TARGET_COL: np.nan}])],
                ignore_index=True,
            )
            feature_row = build_supervised_features(candidate, target_col=TARGET_COL, require_target=False).tail(1)
            yhat = float(self._model.predict(feature_row[FEATURE_COLUMNS])[0])
            yhat = max(yhat, 0.0)
            predictions.append(yhat)
            history = pd.concat([history, pd.DataFrame([{DATE_COL: date, TARGET_COL: yhat}])], ignore_index=True)
        return pd.Series(predictions, index=dates)
