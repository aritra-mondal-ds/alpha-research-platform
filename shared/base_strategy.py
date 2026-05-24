# shared/base_strategy.py
"""
Abstract base strategy — identical contract to crypto/src/strategies/base.py.
Centralised here so forex and commodities inherit without duplication.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    NAME:    str = "base"
    VERSION: str = "1.0"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns pd.Series of {-1, 0, 1} indexed identically to df.
        Signal at t must only use df.iloc[:t+1] — no lookahead.
        """

    def _validate_signals(self, signals: pd.Series, df: pd.DataFrame):
        assert signals.index.equals(df.index),    "Signal/DataFrame index mismatch"
        assert signals.isin([-1, 0, 1]).all(),    "Invalid signal values"
        assert not signals.isnull().any(),         "NaN in signals"

    def _require_columns(self, df: pd.DataFrame, cols: list):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{self.NAME}: missing columns {missing}")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.NAME})"
