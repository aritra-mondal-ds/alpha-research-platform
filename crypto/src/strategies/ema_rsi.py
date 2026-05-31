# # crypto/src/strategies/ema_rsi.py

"""
Strategy 1: EMA Crossover + RSI Filter

Entry logic (LONG):
    - EMA9 crosses above EMA21 (momentum shift)
    - RSI14 > rsi_long_threshold (trend confirmation, not overbought)
    - Price above EMA50 (macro trend filter)

Exit logic (signal = -1):
    - EMA9 crosses below EMA21  OR
    - RSI14 > rsi_exit_ob (overbought exit)  OR
    - Price drops below EMA50 (trend breakdown)

No short entries in default mode (long_only=True).
"""

import pandas as pd
from loguru import logger
from src.strategies.base import BaseStrategy


class EMACrossoverRSI(BaseStrategy):

    NAME = "ema_crossover_rsi"
    VERSION = "1.0"

    def __init__(
        self,
        rsi_long_threshold: float = 50.0,
        rsi_exit_ob: float = 75.0,
        rsi_short_threshold: float = 50.0,
        rsi_exit_os: float = 25.0,
        use_ema50_filter: bool = True,
        long_only: bool = True,
    ):
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_exit_ob = rsi_exit_ob
        self.rsi_short_threshold = rsi_short_threshold
        self.rsi_exit_os = rsi_exit_os
        self.use_ema50_filter = use_ema50_filter
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        required = ["ema_9", "ema_21", "rsi_14", "close"]
        if self.use_ema50_filter:
            required.append("ema_50")
        self._require_columns(df, required)

        fast_above_slow = (df["ema_9"] > df["ema_21"]).astype("boolean")
        prev_fast_above = fast_above_slow.shift(1)
        cross_up = fast_above_slow & ~prev_fast_above.fillna(False)
        cross_down = ~fast_above_slow & prev_fast_above.fillna(True)

        rsi_long_ok = df["rsi_14"] > self.rsi_long_threshold
        rsi_exit_ob = df["rsi_14"] > self.rsi_exit_ob
        rsi_short_ok = df["rsi_14"] < self.rsi_short_threshold
        rsi_exit_os = df["rsi_14"] < self.rsi_exit_os
        price_above_ema50 = (
            df["close"] > df["ema_50"] if self.use_ema50_filter
            else pd.Series(True, index=df.index)
        )

        signals = pd.Series(0, index=df.index, dtype=int)

        long_entry = cross_up & rsi_long_ok & price_above_ema50
        signals[long_entry] = 1

        long_exit = cross_down | rsi_exit_ob | ~price_above_ema50
        signals[long_exit & (signals != 1)] = -1

        if not self.long_only:
            short_entry = cross_down & rsi_short_ok & ~price_above_ema50
            signals[short_entry] = -1

        self._validate_signals(signals, df)

        n_long = (signals == 1).sum()
        n_exit = (signals == -1).sum()
        logger.info(
            f"{self.NAME}: {n_long} long entries | {n_exit} exit signals | {len(df):,} total bars"
        )
        return signals
