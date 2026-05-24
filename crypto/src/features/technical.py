# crypto/src/features/technical.py
"""
Technical indicator feature computation.
Uses the `ta` library exclusively.

Zero-volume handling:
    Forex pairs (e.g. AUDUSD=X) report 0 volume via yfinance.
    Volume-dependent indicators (VWAP, OBV, MFI) fall back to
    price-based proxies so no NaN rows are introduced.

Lookahead-bias contract:
    Every feature at index t uses only df.iloc[0..t].
"""

import numpy as np
import pandas as pd
import ta
from loguru import logger


class TechnicalFeatures:

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_input(df)
        out = df.copy()

        # Detect zero-volume asset (forex pairs report 0 volume via yfinance)
        self._zero_volume = (df["volume"].sum() == 0)
        if self._zero_volume:
            logger.info(
                "Zero-volume asset detected — "
                "volume indicators replaced with price-based proxies"
            )

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

        df["ema_cross_9_21"]   = (df["ema_9"]  > df["ema_21"]).astype(int)
        df["ema_cross_21_50"]  = (df["ema_21"] > df["ema_50"]).astype(int)
        df["ema_cross_50_200"] = (df["ema_50"] > df["ema_200"]).astype(int)

        df["price_vs_ema50"]  = (close - df["ema_50"])  / (df["ema_50"]  + 1e-9)
        df["price_vs_ema200"] = (close - df["ema_200"]) / (df["ema_200"] + 1e-9)

        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()
        df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)

        adx = ta.trend.ADXIndicator(df["high"], df["low"], close, window=14)
        df["adx"]     = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        return df

    # ── Momentum ──────────────────────────────────────────────────────────────
    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["rsi_14"] = ta.momentum.rsi(close, window=14)
        df["rsi_7"]  = ta.momentum.rsi(close, window=7)

        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)

        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], close, window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["williams_r"] = ta.momentum.williams_r(
            df["high"], df["low"], close, lbp=14
        )
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
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"]   = bb.bollinger_pband()

        df["atr_14"]  = ta.volatility.average_true_range(
            df["high"], df["low"], close, window=14
        )
        df["atr_pct"] = df["atr_14"] / (close + 1e-9)

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

        if self._zero_volume:
            # ── Zero-volume fallback (forex pairs) ────────────────────────────
            # volume_ratio: ATR momentum as a proxy for "volume surge"
            atr     = ta.volatility.average_true_range(
                df["high"], df["low"], close, window=14
            )
            atr_sma = atr.rolling(20).mean()
            df["volume_ratio"] = atr / (atr_sma + 1e-9)

            # OBV proxy: directional price accumulation
            price_dir  = np.sign(close.diff().fillna(0))
            df["obv"]  = price_dir.cumsum()

            # VWAP proxy: EMA of typical price (no volume dependency)
            typical   = (df["high"] + df["low"] + close) / 3
            df["vwap"] = ta.trend.ema_indicator(typical, window=14)
            df["vwap_gap"] = (close - df["vwap"]) / (df["vwap"] + 1e-9)

            # MFI proxy: RSI on typical price
            df["mfi_14"] = ta.momentum.rsi(typical, window=14)

        else:
            # ── Standard volume indicators (crypto, ETFs) ─────────────────────
            df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-9)
            df["obv"]          = ta.volume.on_balance_volume(close, volume)

            try:
                vwap_obj   = ta.volume.VolumeWeightedAveragePrice(
                    df["high"], df["low"], close, volume
                )
                df["vwap"] = vwap_obj.volume_weighted_average_price()
                # If VWAP produced all NaN, fall back to EMA of typical price
                if df["vwap"].isna().all():
                    raise ValueError("VWAP produced all NaN")
            except Exception:
                typical    = (df["high"] + df["low"] + close) / 3
                df["vwap"] = ta.trend.ema_indicator(typical, window=14)

            df["vwap_gap"] = (close - df["vwap"]) / (df["vwap"] + 1e-9)
            df["mfi_14"]   = ta.volume.money_flow_index(
                df["high"], df["low"], close, volume, window=14
            )

        return df

    # ── Price-Derived ─────────────────────────────────────────────────────────
    def _price_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["log_return"]       = np.log(close / (close.shift(1) + 1e-9))
        df["return_3"]         = close.pct_change(3)
        df["return_5"]         = close.pct_change(5)
        df["return_24"]        = close.pct_change(24)
        df["high_low_spread"]  = (df["high"] - df["low"]) / (close + 1e-9)
        df["close_pct_range"]  = (
            (close - df["low"]) / (df["high"] - df["low"] + 1e-9)
        )
        df["realised_vol_24"]  = df["log_return"].rolling(24).std()
        df["realised_vol_168"] = df["log_return"].rolling(168).std()

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