# mt5_integration/signal_engine.py
"""
Signal engine for 5-minute scalping.

Uses EMA(9/21) + RSI(40 threshold) for all assets — more aggressive than the
hourly settings (EMA 50/75, RSI 75) to generate 100+ signals/day.

All src.* imports are inside __init__() and methods to avoid the
`config.settings` namespace clash when running from mt5_integration/.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

_REPO_ROOT  = Path(__file__).resolve().parents[1]
_CRYPTO_SRC = _REPO_ROOT / "crypto"
if str(_CRYPTO_SRC) not in sys.path:
    sys.path.append(str(_CRYPTO_SRC))

from mt5_config import ASSETS, EMA_FAST, EMA_SLOW, RSI_THRESHOLD


@dataclass
class Signal:
    asset:         str
    direction:     int    # -1 SHORT | 0 HOLD | 1 LONG
    confidence:    float
    strategy:      str
    atr_pct:       float
    current_price: float


class SignalEngine:

    def __init__(self):
        # Lazy imports — prevents config.settings namespace crash
        from src.features.technical import TechnicalFeatures
        from src.features.patterns  import ChartPatternFeatures

        self._tech     = TechnicalFeatures()
        self._patterns = ChartPatternFeatures()
        logger.info(
            f"SignalEngine ready — strategy: EMA({EMA_FAST}/{EMA_SLOW}) "
            f"RSI threshold={RSI_THRESHOLD}"
        )

    def generate(
        self, symbol: str, df_primary: pd.DataFrame, df_daily: pd.DataFrame
    ) -> Signal:
        close_px = float(df_primary["close"].iloc[-1])
        df_feat  = self._build_features(df_primary, df_daily)

        if len(df_feat) == 0:
            logger.warning(
                f"{symbol}: Empty features (need ≥200 bars, "
                f"got {len(df_primary)}). Returning HOLD."
            )
            return Signal(symbol, 0, 0.0, "insufficient_data", 0.0, close_px)

        atr_pct   = float(df_feat["atr_pct"].iloc[-1]) \
                    if "atr_pct" in df_feat.columns else 0.0
        direction = self._ema_rsi_signal(df_feat)

        logger.info(
            f"{symbol}: {'LONG' if direction==1 else 'SHORT' if direction==-1 else 'HOLD'}"
            f" | strategy=ema_rsi_{EMA_FAST}/{EMA_SLOW}"
            f" | px={close_px:.5f} | atr_pct={atr_pct:.5f}"
        )
        return Signal(symbol, direction, 0.0, "ema_rsi", atr_pct, close_px)

    def generate_all(self, data: dict) -> dict[str, Signal]:
        signals = {}
        for symbol, (df_primary, df_daily) in data.items():
            try:
                signals[symbol] = self.generate(symbol, df_primary, df_daily)
            except Exception as e:
                logger.error(f"{symbol}: Signal failed — {e}")
                import traceback; traceback.print_exc()
                px = float(df_primary["close"].iloc[-1]) if len(df_primary) > 0 else 0.0
                signals[symbol] = Signal(symbol, 0, 0.0, "error", 0.0, px)
        return signals

    # ── Feature building ──────────────────────────────────────────────────────
    def _build_features(
        self, df_primary: pd.DataFrame, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        df = self._tech.compute(df_primary.copy())
        if len(df) == 0:
            return df
        df = self._patterns.compute(df)

        # Merge daily macro context (same reindex fix as pipeline.py)
        df_d = self._tech.compute(df_daily.copy())
        if len(df_d) > 0:
            daily_cols = [
                "ema_50","ema_200","rsi_14","atr_pct",
                "bb_pct","bb_width","adx","macd_hist","realised_vol_24",
            ]
            avail   = [c for c in daily_cols if c in df_d.columns]
            sub     = df_d[avail].add_prefix("d1_")
            reindexed = sub.reindex(df.index, method="ffill")
            for col in reindexed.columns:
                df[col] = reindexed[col].values

        df.dropna(inplace=True)
        return df

    # ── EMA + RSI scalping signal ─────────────────────────────────────────────
    def _ema_rsi_signal(self, df: pd.DataFrame) -> int:
        """
        Aggressive EMA(9/21) crossover + RSI(14) for 5m scalping.

        Entry rules:
            LONG:  fast EMA crosses above slow EMA AND RSI > RSI_THRESHOLD
            SHORT: fast EMA crosses below slow EMA AND RSI < (100-RSI_THRESHOLD)

        Position management (latest bar only):
            If currently LONG  → hold while fast > slow, exit when fast < slow
            If currently SHORT → hold while fast < slow, exit when fast > slow
        """
        if f"ema_{EMA_FAST}" not in df.columns or f"ema_{EMA_SLOW}" not in df.columns:
            logger.warning("EMA columns not found — check TechnicalFeatures output.")
            return 0

        fast     = df[f"ema_{EMA_FAST}"]
        slow     = df[f"ema_{EMA_SLOW}"]
        rsi      = df["rsi_14"] if "rsi_14" in df.columns else pd.Series(50, index=df.index)

        fast_above = fast > slow
        cross_up   = fast_above & ~fast_above.shift(1).fillna(False)
        cross_down = ~fast_above & fast_above.shift(1).fillna(True)

        # Latest bar
        latest_idx = df.index[-1]
        is_long    = bool(fast_above.iloc[-1])
        crossed_up = bool(cross_up.iloc[-1])
        crossed_dn = bool(cross_down.iloc[-1])
        latest_rsi = float(rsi.iloc[-1])

        # New entry signals (crossover on this bar)
        if crossed_up and latest_rsi > RSI_THRESHOLD:
            return 1    # LONG entry

        if crossed_dn and latest_rsi < (100 - RSI_THRESHOLD):
            return -1   # SHORT entry

        # Trend continuation (already in a trade — stay in)
        if is_long:
            return 1
        else:
            return -1