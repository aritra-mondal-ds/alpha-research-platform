# crypto/src/features/technical.py
"""
Technical indicator feature computation.
Uses the `ta` library exclusively (pandas-ta dropped — incompatible with Python 3.11).

Lookahead-bias contract:
    Every feature at index t is computed using only df.iloc[0..t].
    No shift(-n), no future data, no centered rolling windows.
"""

import numpy as np
import pandas as pd
import ta
from loguru import logger


class TechnicalFeatures:

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators on an OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume],
                UTC-indexed, sorted ascending.

        Returns:
            DataFrame with original OHLCV + all indicator columns.
            Rows with NaN (warmup period) are dropped.
        """
        self._check_input(df)
        out = df.copy()

        out = self._trend(out)
        out = self._momentum(out)
        out = self._volatility(out)
        out = self._volume(out)
        out = self._price_derived(out)

        before = len(out)
        out.dropna(inplace=True)
        dropped = before - len(out)
        logger.info(
            f"TechnicalFeatures: {len(out):,} rows "
            f"({dropped} warmup rows dropped)"
        )
        return out

    # ── Trend ─────────────────────────────────────────────────────────────────
    def _trend(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["ema_9"]   = ta.trend.ema_indicator(close, window=9)
        df["ema_21"]  = ta.trend.ema_indicator(close, window=21)
        df["ema_50"]  = ta.trend.ema_indicator(close, window=50)
        df["ema_200"] = ta.trend.ema_indicator(close, window=200)

        # Crossover: 1 when fast > slow (bullish alignment)
        df["ema_cross_9_21"]  = (df["ema_9"]  > df["ema_21"]).astype(int)
        df["ema_cross_21_50"] = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_cross_50_200"]= (df["ema_50"] > df["ema_200"]).astype(int)

        # Price relative to EMAs (normalised distance)
        df["price_vs_ema50"]  = (close - df["ema_50"])  / df["ema_50"]
        df["price_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]

        # MACD
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()
        df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)

        # ADX — trend strength (no direction)
        adx = ta.trend.ADXIndicator(df["high"], df["low"], close, window=14)
        df["adx"]     = adx.adx()
        df["adx_pos"] = adx.adx_pos()   # +DI
        df["adx_neg"] = adx.adx_neg()   # -DI

        return df

    # ── Momentum ──────────────────────────────────────────────────────────────
    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["rsi_14"] = ta.momentum.rsi(close, window=14)
        df["rsi_7"]  = ta.momentum.rsi(close, window=7)

        # RSI regime flags
        df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], close, window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Williams %R
        df["williams_r"] = ta.momentum.williams_r(
            df["high"], df["low"], close, lbp=14
        )

        # Rate of Change
        df["roc_5"]  = ta.momentum.roc(close, window=5)
        df["roc_10"] = ta.momentum.roc(close, window=10)

        return df

    # ── Volatility ────────────────────────────────────────────────────────────
    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()   # (upper-lower)/mid — normalised
        df["bb_pct"]   = bb.bollinger_pband()   # position within bands [0..1]

        df["atr_14"]  = ta.volatility.average_true_range(
            df["high"], df["low"], close, window=14
        )
        df["atr_pct"] = df["atr_14"] / close    # normalised ATR

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df["high"], df["low"], close, window=20)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        df["kc_pct"]   = kc.keltner_channel_pband()

        return df

    # ── Volume ────────────────────────────────────────────────────────────────
    def _volume(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"]
        volume = df["volume"]

        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"]  = volume / (df["volume_sma_20"] + 1e-9)

        df["obv"] = ta.volume.on_balance_volume(close, volume)

        vwap = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], close, volume
        )
        df["vwap"]     = vwap.volume_weighted_average_price()
        df["vwap_gap"] = (close - df["vwap"]) / (df["vwap"] + 1e-9)

        # Money Flow Index
        df["mfi_14"] = ta.volume.money_flow_index(
            df["high"], df["low"], close, volume, window=14
        )

        return df

    # ── Price-Derived ─────────────────────────────────────────────────────────
    def _price_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["log_return"]       = np.log(close / close.shift(1))
        df["return_3"]         = close.pct_change(3)
        df["return_5"]         = close.pct_change(5)
        df["return_24"]        = close.pct_change(24)
        df["high_low_spread"]  = (df["high"] - df["low"]) / (close + 1e-9)
        df["close_pct_range"]  = (
            (close - df["low"]) / (df["high"] - df["low"] + 1e-9)
        )

        # Rolling realised volatility
        df["realised_vol_24"] = df["log_return"].rolling(24).std()
        df["realised_vol_168"]= df["log_return"].rolling(168).std()  # 1 week

        return df

    # ── Input validation ──────────────────────────────────────────────────────
    def _check_input(self, df: pd.DataFrame):
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be sorted ascending.")
        if len(df) < 200:
            raise ValueError(
                f"Need at least 200 rows for EMA-200 warmup. Got {len(df)}."
            )