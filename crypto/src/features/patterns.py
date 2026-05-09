# crypto/src/features/patterns.py
"""
Candlestick pattern detection using pure pandas/numpy.
No external CDL library required.

All patterns are computed using only current and previous bars — no lookahead.
Each pattern column is binary: 1 = pattern present, 0 = not present.
"""

import numpy as np
import pandas as pd
from loguru import logger


class ChartPatternFeatures:

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns and append as binary columns.

        Args:
            df: OHLCV DataFrame (post-indicator computation is fine).

        Returns:
            DataFrame with pattern columns appended (filled with 0 for NaN).
        """
        out = df.copy()

        # ── Candle anatomy ────────────────────────────────────────────────────
        body       = out["close"] - out["open"]
        abs_body   = body.abs()
        full_range = out["high"] - out["low"] + 1e-9
        upper_wick = out["high"] - out[["close", "open"]].max(axis=1)
        lower_wick = out[["close", "open"]].min(axis=1) - out["low"]

        body_ratio  = abs_body / full_range           # 0..1
        upper_ratio = upper_wick / full_range
        lower_ratio = lower_wick / full_range

        # Previous-bar values
        prev_open  = out["open"].shift(1)
        prev_close = out["close"].shift(1)
        prev_body  = prev_close - prev_open
        prev_abs   = prev_body.abs()

        # ── Single-bar patterns ───────────────────────────────────────────────

        # Doji: body < 5% of range — indecision
        out["pat_doji"] = (body_ratio < 0.05).astype(int)

        # Hammer: bullish reversal — lower wick ≥ 2× body, small upper wick
        out["pat_hammer"] = (
            (body > 0) &
            (lower_ratio >= 0.5) &
            (lower_wick >= 2.0 * abs_body) &
            (upper_ratio < 0.15)
        ).astype(int)

        # Inverted Hammer: same shape, bearish body
        out["pat_inv_hammer"] = (
            (body < 0) &
            (upper_ratio >= 0.5) &
            (upper_wick >= 2.0 * abs_body) &
            (lower_ratio < 0.15)
        ).astype(int)

        # Shooting Star: bearish, long upper wick — appears at highs
        out["pat_shooting_star"] = (
            (body < 0) &
            (upper_ratio >= 0.5) &
            (upper_wick >= 2.0 * abs_body) &
            (lower_ratio < 0.1)
        ).astype(int)

        # Marubozu: body > 90% of range — strong momentum, no wicks
        out["pat_marubozu_bull"] = (
            (body > 0) & (body_ratio >= 0.90)
        ).astype(int)

        out["pat_marubozu_bear"] = (
            (body < 0) & (body_ratio >= 0.90)
        ).astype(int)

        # Spinning Top: small body (10–30% of range), wicks on both sides
        out["pat_spinning_top"] = (
            (body_ratio >= 0.10) &
            (body_ratio < 0.30) &
            (upper_ratio >= 0.15) &
            (lower_ratio >= 0.15)
        ).astype(int)

        # ── Two-bar patterns ──────────────────────────────────────────────────

        # Bullish Engulfing: prev bearish, current bullish & fully covers prev
        out["pat_engulf_bull"] = (
            (prev_body < 0) &
            (body > 0) &
            (out["open"] <= prev_close) &
            (out["close"] >= prev_open)
        ).astype(int)

        # Bearish Engulfing: prev bullish, current bearish & fully covers prev
        out["pat_engulf_bear"] = (
            (prev_body > 0) &
            (body < 0) &
            (out["open"] >= prev_close) &
            (out["close"] <= prev_open)
        ).astype(int)

        # Bullish Harami: current inside previous bearish candle
        out["pat_harami_bull"] = (
            (prev_body < 0) &
            (body > 0) &
            (out["open"] > prev_close) &
            (out["close"] < prev_open)
        ).astype(int)

        # Bearish Harami: current inside previous bullish candle
        out["pat_harami_bear"] = (
            (prev_body > 0) &
            (body < 0) &
            (out["open"] < prev_close) &
            (out["close"] > prev_open)
        ).astype(int)

        # Piercing Line: gap down open, closes above mid of previous bearish bar
        out["pat_piercing"] = (
            (prev_body < 0) &
            (body > 0) &
            (out["open"] < prev_close) &
            (out["close"] > (prev_open + prev_close) / 2)
        ).astype(int)

        # Dark Cloud Cover: gap up open, closes below mid of previous bullish bar
        out["pat_dark_cloud"] = (
            (prev_body > 0) &
            (body < 0) &
            (out["open"] > prev_close) &
            (out["close"] < (prev_open + prev_close) / 2)
        ).astype(int)

        # ── Composite signal columns ──────────────────────────────────────────

        # Bullish pattern present (any)
        bullish_cols = [
            "pat_hammer", "pat_engulf_bull",
            "pat_harami_bull", "pat_piercing", "pat_marubozu_bull"
        ]
        out["pat_any_bull"] = (out[bullish_cols].sum(axis=1) > 0).astype(int)

        # Bearish pattern present (any)
        bearish_cols = [
            "pat_shooting_star", "pat_engulf_bear",
            "pat_harami_bear", "pat_dark_cloud", "pat_marubozu_bear"
        ]
        out["pat_any_bear"] = (out[bearish_cols].sum(axis=1) > 0).astype(int)

        # Fill NaN from shift(1) operations with 0
        pat_cols = [c for c in out.columns if c.startswith("pat_")]
        out[pat_cols] = out[pat_cols].fillna(0).astype(int)

        logger.info(
            f"ChartPatternFeatures: {len(pat_cols)} pattern columns added "
            f"({out[pat_cols].sum().sum():,} total pattern occurrences)"
        )
        return out