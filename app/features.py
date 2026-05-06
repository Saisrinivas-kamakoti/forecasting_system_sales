import pandas as pd


def _is_us_holiday(date_value: pd.Timestamp) -> int:
    """Small built-in holiday flag to avoid a runtime dependency in feature creation."""
    date_value = pd.Timestamp(date_value)
    month_day = (date_value.month, date_value.day)
    fixed_holidays = {(1, 1), (7, 4), (12, 25)}
    if month_day in fixed_holidays:
        return 1
    # Thanksgiving: fourth Thursday in November.
    if date_value.month == 11 and date_value.day_name() == "Thursday":
        return int(22 <= date_value.day <= 28)
    return 0


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
    out["quarter"] = out[date_col].dt.quarter
    out["holiday_flag"] = out[date_col].map(_is_us_holiday).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Total",
    lags: tuple[int, ...] = (1, 7, 30),
    rolling_windows: tuple[int, ...] = (4, 8, 12),
) -> pd.DataFrame:
    out = df.copy().sort_values("Date")
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    shifted = out[target_col].shift(1)
    for window in rolling_windows:
        out[f"rolling_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window=window, min_periods=2).std()
    return out


def build_supervised_features(
    df: pd.DataFrame,
    target_col: str = "Total",
    require_target: bool = True,
) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lag_features(out, target_col=target_col)
    feature_cols = FEATURE_COLUMNS + ([target_col] if require_target else [])
    return out.dropna(subset=feature_cols).reset_index(drop=True)


FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "week_of_year",
    "quarter",
    "holiday_flag",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_4",
    "rolling_std_4",
    "rolling_mean_8",
    "rolling_std_8",
    "rolling_mean_12",
    "rolling_std_12",
]
