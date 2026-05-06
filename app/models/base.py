from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastModel(ABC):
    name: str

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "ForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, periods: int, history_df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

