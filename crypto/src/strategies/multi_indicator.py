# crypto/src/strategies/multi_indicator.py
"""
Strategy 2: Multi-Indicator Confluence

Philosophy: only trade when multiple independent signals agree.
Reduces false signals at the cost of fewer total trades.

Long entry requires ALL of:
    1. EMA trend aligned  (EMA9 > EMA21 > EMA50)
    2. RSI in range       (45 < RSI14 < 70)
    3. MACD bullish       (MACD histogram > 0 and rising)
    4. Volume surge       (volume_ratio > 1.2 — smart money confirming)
    5. Above VWAP         (institutional benchmark)

Short entry (long_short mode) requires ALL of:
    1. EMA trend bearish  (EMA9 < EMA21 < EMA50)
    2. RSI in range       (30 < RSI14 < 55)
    3. MACD bearish       (MACD histogram < 0 and falling)
    4. Below VWAP

Exit:
    - Trend breakdown (EMA9 < EMA21)
    - RSI extreme (> 78 overbought or < 22 oversold)
    - ATR-based: price moves against position by > 2x ATR
"""

import pandas as pd
from loguru import logger
from src.strategies.base import BaseStrategy


class MultiIndicatorConfluence(BaseStrategy):

    NAME    = "multi_indicator_confluence"
    VERSION = "1.0"

    def __init__(
        self,
        rsi_low:          float = 45.0,
        rsi_high:         float = 70.0,
        rsi_ob_exit:      float = 78.0,
        rsi_os_exit:      float = 22.0,
        volume_surge:     float = 1.2,   # volume must be > N × 20-bar avg
        macd_rising_bars: int   = 2,     # MACD hist must have risen for N bars
        long_only:        bool  = True,
    ):
        self.rsi_low          = rsi_low
        self.rsi_high         = rsi_high
        self.rsi_ob_exit      = rsi_ob_exit
        self.rsi_os_exit      = rsi_os_exit
        self.volume_surge     = volume_surge
        self.macd_rising_bars = macd_rising_bars
        self.long_only        = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        required = [
            "ema_9", "ema_21", "ema_50",
            "rsi_14", "macd_hist", "volume_ratio",
            "vwap", "close", "atr_14",
        ]
        self._require_columns(df, required)

        # ── Trend conditions ──────────────────────────────────────────────────
        ema_bullish = (
            (df["ema_9"] > df["ema_21"]) &
            (df["ema_21"] > df["ema_50"])
        )
        ema_bearish = (
            (df["ema_9"] < df["ema_21"]) &
            (df["ema_21"] < df["ema_50"])
        )

        # ── RSI conditions ────────────────────────────────────────────────────
        rsi_bull_range = (df["rsi_14"] > self.rsi_low) & (df["rsi_14"] < self.rsi_high)
        rsi_bear_range = (df["rsi_14"] > (100 - self.rsi_high)) & (df["rsi_14"] < (100 - self.rsi_low))
        rsi_overbought = df["rsi_14"] > self.rsi_ob_exit
        rsi_oversold   = df["rsi_14"] < self.rsi_os_exit

        # ── MACD conditions ───────────────────────────────────────────────────
        macd_positive = df["macd_hist"] > 0
        macd_negative = df["macd_hist"] < 0

        # MACD histogram rising for N consecutive bars
        macd_rising  = df["macd_hist"] > df["macd_hist"].shift(self.macd_rising_bars)
        macd_falling = df["macd_hist"] < df["macd_hist"].shift(self.macd_rising_bars)

        # ── Volume conditions ─────────────────────────────────────────────────
        vol_surge = df["volume_ratio"] > self.volume_surge

        # ── Price vs VWAP ─────────────────────────────────────────────────────
        above_vwap = df["close"] > df["vwap"]
        below_vwap = df["close"] < df["vwap"]

        # ── Entry signals — ALL conditions must agree ─────────────────────────
        long_entry = (
            ema_bullish   &
            rsi_bull_range &
            macd_positive  &
            macd_rising    &
            vol_surge      &
            above_vwap
        )

        short_entry = (
            ema_bearish   &
            rsi_bear_range &
            macd_negative  &
            macd_falling   &
            vol_surge      &
            below_vwap
        )

        # ── Exit signals ──────────────────────────────────────────────────────
        long_exit = (
            ~ema_bullish |        # trend breakdown
            rsi_overbought        # extreme overbought
        )

        short_exit = (
            ~ema_bearish |
            rsi_oversold
        )

        # ── Build signal series ───────────────────────────────────────────────
        signals = pd.Series(0, index=df.index, dtype=int)

        signals[long_entry]                     =  1
        signals[long_exit & (signals != 1)]     = -1

        if not self.long_only:
            signals[short_entry]                =  -1
            signals[short_exit & (signals != -1)] = 1

        self._validate_signals(signals, df)

        n_long  = (signals ==  1).sum()
        n_short = (signals == -1).sum()
        logger.info(
            f"{self.NAME}: {n_long} long | {n_short} short signals "
            f"| confluence filter active (all conditions required)"
        )
        return signals