# crypto/src/strategies/base.py
"""
Abstract base class for all trading strategies.

Every strategy must implement generate_signals() and respect the
lookahead contract: signal at time t may only use data up to and
including bar t. No future data, ever.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    # Subclasses set these for identification
    NAME: str = "base"
    VERSION: str = "1.0"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate raw entry/exit signals from a feature DataFrame.

        Args:
            df: Feature DataFrame with OHLCV + indicator columns,
                UTC-indexed, sorted ascending.

        Returns:
            pd.Series of int, same index as df:
                +1  →  LONG  signal (buy / enter long)
                 0  →  HOLD  (no action / maintain position)
                -1  →  SHORT signal (sell / enter short / exit long)

        Lookahead contract:
            Signal at index t must only reference df.iloc[:t+1].
            Vectorised operations on the full df are fine as long as
            they don't depend on future values (e.g. no shift(-n)).
        """

    # ── Shared validation helpers ─────────────────────────────────────────────
    def _validate_signals(self, signals: pd.Series, df: pd.DataFrame):
        assert signals.index.equals(df.index), (
            "Signal index does not match DataFrame index."
        )
        assert signals.isin([-1, 0, 1]).all(), (
            f"Invalid signal values found: {signals.unique()}"
        )
        assert not signals.isnull().any(), "NaN values in signals."

    def _require_columns(self, df: pd.DataFrame, cols: list[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.NAME}: required columns not found in DataFrame: {missing}\n"
                f"Available: {list(df.columns)}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.NAME}, version={self.VERSION})"