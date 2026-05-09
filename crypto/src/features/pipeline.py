# crypto/src/features/pipeline.py
"""
Feature pipeline orchestrator.

Execution order:
    1. Load raw 1h OHLCV
    2. Load raw 1d OHLCV
    3. Compute technical indicators on 1h
    4. Compute candlestick patterns on 1h
    5. Compute technical indicators on 1d
    6. Merge daily macro features into 1h (forward-fill — no lookahead)
    7. Build classification target
    8. Drop NaN rows (warmup + last horizon rows)
    9. Save to data/processed/

Lookahead guarantee:
    Daily features are merged using .ffill() after resampling to hourly
    frequency. A daily bar closing at 00:00 UTC propagates its value
    forward into all subsequent hourly bars on the same day — never
    backward into past bars.
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg

from src.data.loader         import DataLoader
from src.features.technical  import TechnicalFeatures
from src.features.patterns   import ChartPatternFeatures
from src.features.target     import TargetBuilder


# ── Feature columns used by ML models ────────────────────────────────────────
# Populated automatically by the pipeline; imported elsewhere for consistency.
FEATURE_COLS: list[str] = []   # filled after first pipeline run


class FeaturePipeline:

    def __init__(self):
        self.loader   = DataLoader()
        self.tech     = TechnicalFeatures()
        self.patterns = ChartPatternFeatures()
        self.targets  = TargetBuilder()

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        symbol:      str   = cfg.SYMBOL,
        primary_tf:  str   = cfg.PRIMARY_TF,
        higher_tf:   str   = cfg.HIGHER_TF,
        horizon:     int   = 12,
        threshold:   float = 0.015,
        save:        bool  = True,
    ) -> pd.DataFrame:
        """
        Build the complete feature matrix and return it.

        Args:
            symbol:     Trading pair, e.g. "BTC/USDT"
            primary_tf: Primary timeframe, e.g. "1h"
            higher_tf:  Macro context timeframe, e.g. "1d"
            horizon:    Forward-look window for target label (bars)
            threshold:  Minimum return for LONG/SHORT label
            save:       Write result to data/processed/ parquet

        Returns:
            Feature DataFrame ready for train/val/test splitting.
        """
        global FEATURE_COLS

        logger.info("FeaturePipeline starting ...")

        # ── Step 1 & 2: Load raw data ─────────────────────────────────────────
        df_1h = self.loader.load(symbol, primary_tf)
        df_1d = self.loader.load(symbol, higher_tf)
        logger.info(f"Loaded  1h: {len(df_1h):,} rows | 1d: {len(df_1d):,} rows")

        # ── Step 3: Technical indicators on 1h ───────────────────────────────
        logger.info("Computing 1h technical indicators ...")
        df_1h = self.tech.compute(df_1h)

        # ── Step 4: Candlestick patterns on 1h ───────────────────────────────
        logger.info("Computing candlestick patterns ...")
        df_1h = self.patterns.compute(df_1h)

        # ── Step 5: Technical indicators on 1d ───────────────────────────────
        logger.info("Computing 1d macro indicators ...")
        df_1d_feats = self.tech.compute(df_1d)

        # ── Step 6: Merge macro context into 1h ──────────────────────────────
        logger.info("Merging daily macro features into 1h ...")
        df_1h = self._merge_macro(df_1h, df_1d_feats, primary_tf)

        # ── Step 7: Build target label ────────────────────────────────────────
        logger.info(f"Building classification target [horizon={horizon}h, threshold=±{threshold*100:.1f}%] ...")
        df_1h["target"] = self.targets.build_classification(
            df_1h, horizon=horizon, threshold=threshold
        )

        # ── Step 8: Drop NaN (warmup + last horizon rows) ─────────────────────
        before = len(df_1h)
        df_1h.dropna(inplace=True)
        logger.info(
            f"After dropna: {len(df_1h):,} rows "
            f"({before - len(df_1h)} dropped)"
        )

        # ── Record feature columns (everything except target) ─────────────────
        exclude  = {"open", "high", "low", "close", "volume", "target"}
        FEATURE_COLS = [c for c in df_1h.columns if c not in exclude]
        logger.info(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

        # ── Step 9: Save ──────────────────────────────────────────────────────
        if save:
            self._save(df_1h, symbol, primary_tf)

        logger.info("FeaturePipeline complete.")
        return df_1h

    # ─────────────────────────────────────────────────────────────────────────
    def _merge_macro(
        self,
        df_primary:  pd.DataFrame,
        df_daily:    pd.DataFrame,
        primary_tf:  str,
    ) -> pd.DataFrame:
        """
        Forward-fill daily indicator values into the primary (1h) DataFrame.

        Why forward-fill?
            A daily bar closes at 00:00 UTC. We propagate its value into
            subsequent hourly bars using ffill — never backward. This
            means an hourly bar at 15:00 uses the daily value from 00:00
            the same day, which is already finalised. No lookahead.
        """
        # Select a subset of daily cols to avoid collision with hourly cols
        daily_indicator_cols = [
            "ema_50", "ema_200", "rsi_14", "atr_pct",
            "bb_pct", "bb_width", "adx", "macd_hist",
            "realised_vol_24",
        ]
        available = [c for c in daily_indicator_cols if c in df_daily.columns]
        daily_sub = df_daily[available].add_prefix("d1_")

        # Resample daily → primary_tf frequency, then forward-fill
        daily_resampled = (
            daily_sub
            .resample(primary_tf)
            .ffill()
        )

        merged = df_primary.join(daily_resampled, how="left")
        merged[daily_sub.columns] = merged[daily_sub.columns].ffill()

        missing = merged[daily_sub.columns].isnull().sum().sum()
        if missing > 0:
            logger.warning(
                f"Macro merge: {missing} NaN values remain — "
                f"check date alignment between timeframes."
            )
        else:
            logger.info(
                f"Macro merge: {len(daily_sub.columns)} daily features "
                f"merged into {len(merged):,} hourly rows — no NaNs."
            )
        return merged

    def _save(self, df: pd.DataFrame, symbol: str, primary_tf: str):
        safe     = symbol.replace("/", "_")
        out_path = cfg.DATA_DIR / "processed" / f"{safe}_{primary_tf}_features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        logger.info(f"Feature matrix saved → {out_path.name}  ({df.shape})")