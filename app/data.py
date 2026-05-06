from pathlib import Path

import pandas as pd

from app.config import DATE_COL, STATE_COL, TARGET_COL, WEEKLY_FREQ


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {STATE_COL, DATE_COL, TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


def prepare_weekly_state_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to weekly state-level sales and fill missing weeks per state."""
    weekly_frames = []
    for state, group in df.groupby(STATE_COL):
        group = (
            group[[DATE_COL, TARGET_COL]]
            .dropna(subset=[DATE_COL])
            .set_index(DATE_COL)
            .sort_index()
            .resample(WEEKLY_FREQ)[TARGET_COL]
            .sum(min_count=1)
            .to_frame()
        )
        group[TARGET_COL] = group[TARGET_COL].interpolate(method="time").ffill().bfill()
        group[STATE_COL] = state
        group = group.reset_index()
        weekly_frames.append(group)
    weekly = pd.concat(weekly_frames, ignore_index=True)
    return weekly[[STATE_COL, DATE_COL, TARGET_COL]].sort_values([STATE_COL, DATE_COL])


def time_series_split(df: pd.DataFrame, validation_weeks: int):
    if len(df) <= validation_weeks + 30:
        raise ValueError("Not enough observations for lag-30 features and validation split.")
    train = df.iloc[:-validation_weeks].copy()
    validation = df.iloc[-validation_weeks:].copy()
    return train, validation

