import pandas as pd

from app.config import DATE_COL, TARGET_COL, WEEKLY_FREQ
from app.models.base import ForecastModel


class SarimaForecaster(ForecastModel):
    name = "SARIMA"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fit_result = None

    def fit(self, train_df: pd.DataFrame) -> "SarimaForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = train_df.set_index(DATE_COL)[TARGET_COL].asfreq(WEEKLY_FREQ)
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit_result = model.fit(disp=False)
        return self

    def predict(self, periods: int, history_df: pd.DataFrame) -> pd.Series:
        forecast = self._fit_result.forecast(periods)
        forecast.index = pd.date_range(
            start=history_df[DATE_COL].max() + pd.offsets.Week(weekday=6),
            periods=periods,
            freq=WEEKLY_FREQ,
        )
        return forecast.clip(lower=0)


class ProphetForecaster(ForecastModel):
    name = "Prophet"

    def __init__(self):
        self._model = None

    def fit(self, train_df: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet

        prophet_df = train_df[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        model.fit(prophet_df)
        self._model = model
        return self

    def predict(self, periods: int, history_df: pd.DataFrame) -> pd.Series:
        future = self._model.make_future_dataframe(periods=periods, freq=WEEKLY_FREQ, include_history=False)
        forecast = self._model.predict(future)
        return pd.Series(forecast["yhat"].clip(lower=0).to_numpy(), index=pd.to_datetime(forecast["ds"]))

