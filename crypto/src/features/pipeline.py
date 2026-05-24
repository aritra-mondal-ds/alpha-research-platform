# crypto/src/features/pipeline.py
"""
Feature pipeline orchestrator.

Fixes vs original:
    _merge_macro() now uses reindex(method='ffill') instead of
    resample().join(). This handles any trading schedule:
        - Crypto:      24/7  (UTC-aligned hourly bars)
        - Forex:       24/5  (UTC-aligned, no weekends)
        - NYSE stocks: 9:30-16:00 ET (bars at :30-minute offset from UTC)
    The old resample+join approach created a UTC-aligned hourly grid
    that never matched NYSE timestamps (13:30, 14:30 UTC), producing
    all-NaN daily features and 0 rows after dropna().
"""

import sys
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg

from src.data.loader        import DataLoader
from src.features.technical import TechnicalFeatures
from src.features.patterns  import ChartPatternFeatures
from src.features.target    import TargetBuilder

FEATURE_COLS: list[str] = []


class FeaturePipeline:

    def __init__(self):
        self.loader   = DataLoader()
        self.tech     = TechnicalFeatures()
        self.patterns = ChartPatternFeatures()
        self.targets  = TargetBuilder()

    def run(
        self,
        symbol:      str   = cfg.SYMBOL,
        primary_tf:  str   = cfg.PRIMARY_TF,
        higher_tf:   str   = cfg.HIGHER_TF,
        horizon:     int   = 12,
        threshold:   float = 0.015,
        save:        bool  = True,
    ) -> pd.DataFrame:
        global FEATURE_COLS

        logger.info("FeaturePipeline starting ...")

        # ── Load raw data ─────────────────────────────────────────────────────
        df_1h = self.loader.load(symbol, primary_tf)
        df_1d = self.loader.load(symbol, higher_tf)
        logger.info(
            f"Loaded  1h: {len(df_1h):,} rows | 1d: {len(df_1d):,} rows"
        )

        # ── Technical indicators on 1h ────────────────────────────────────────
        logger.info("Computing 1h technical indicators ...")
        df_1h = self.tech.compute(df_1h)

        if len(df_1h) == 0:
            raise ValueError(
                "No rows remain after computing 1h technical indicators. "
                "Check that the 1h dataset has at least 200 bars."
            )

        # ── Candlestick patterns on 1h ────────────────────────────────────────
        logger.info("Computing candlestick patterns ...")
        df_1h = self.patterns.compute(df_1h)

        # ── Technical indicators on 1d ────────────────────────────────────────
        logger.info("Computing 1d macro indicators ...")
        df_1d_feats = self.tech.compute(df_1d)

        if len(df_1d_feats) == 0:
            logger.warning(
                "No rows in 1d feature set after warmup — "
                "skipping macro merge (daily features will be absent)."
            )
        else:
            # ── Merge macro context into 1h ───────────────────────────────────
            logger.info("Merging daily macro features into 1h ...")
            df_1h = self._merge_macro(df_1h, df_1d_feats)

        # ── Build target label ────────────────────────────────────────────────
        logger.info(
            f"Building classification target "
            f"[horizon={horizon}h, threshold=±{threshold*100:.1f}%] ..."
        )
        df_1h["target"] = self.targets.build_classification(
            df_1h, horizon=horizon, threshold=threshold
        )

        # ── Drop NaN (warmup + last horizon rows) ─────────────────────────────
        before = len(df_1h)
        df_1h.dropna(inplace=True)
        logger.info(
            f"After dropna: {len(df_1h):,} rows "
            f"({before - len(df_1h)} dropped)"
        )

        if len(df_1h) == 0:
            raise ValueError(
                "Feature matrix is empty after dropna. "
                "Possible causes:\n"
                "  1. Dataset too short (need > 200 + horizon bars)\n"
                "  2. All-NaN columns from failed indicator computation\n"
                "  3. Macro merge produced all-NaN daily features\n"
                "Run with --mode features to inspect intermediate outputs."
            )

        # ── Record feature columns ────────────────────────────────────────────
        exclude      = {"open", "high", "low", "close", "volume", "target"}
        FEATURE_COLS = [c for c in df_1h.columns if c not in exclude]
        logger.info(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

        # ── Save ──────────────────────────────────────────────────────────────
        if save:
            self._save(df_1h, symbol, primary_tf)

        logger.info("FeaturePipeline complete.")
        return df_1h

    # ─────────────────────────────────────────────────────────────────────────
    def _merge_macro(
        self,
        df_primary: pd.DataFrame,
        df_daily:   pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill daily indicator values into the primary (1h) DataFrame.

        Uses reindex(method='ffill') instead of resample().join() so that
        ANY trading schedule works correctly:

            Crypto  (24/7)  → bars at 00:00, 01:00 ... 23:00 UTC  ✓
            Forex   (24/5)  → bars at 00:00, 01:00 ... 23:00 UTC  ✓
            NYSE    (9:30-16:00 ET) → bars at 13:30, 14:30 UTC   ✓ (was broken)

        The old approach used resample("1h").ffill() which created an exact
        hourly grid at :00 minutes. NYSE bars fall at :30 minutes, so the
        join produced all-NaN daily features for stocks/ETFs.

        reindex(method='ffill') maps each daily value to the nearest
        PRECEDING bar in df_primary, regardless of the exact minute offset.
        """
        daily_indicator_cols = [
            "ema_50", "ema_200", "rsi_14", "atr_pct",
            "bb_pct", "bb_width", "adx", "macd_hist",
            "realised_vol_24",
        ]
        available = [c for c in daily_indicator_cols if c in df_daily.columns]
        daily_sub = df_daily[available].add_prefix("d1_")

        # Sort by index (should already be sorted, but be safe)
        daily_sub   = daily_sub.sort_index()
        df_out      = df_primary.copy()

        # Reindex: for each primary-tf timestamp, use the most recent daily value
        # method='ffill' → carry forward the last known daily bar
        # tolerance is not set → any gap is filled (daily values persist for days)
        daily_reindexed = daily_sub.reindex(
            df_out.index, method="ffill"
        )

        for col in daily_reindexed.columns:
            df_out[col] = daily_reindexed[col].values

        # Diagnostic: count remaining NaN in daily feature columns
        nan_count = df_out[daily_reindexed.columns].isnull().sum().sum()
        if nan_count > 0:
            logger.warning(
                f"Macro merge: {nan_count} NaN values remain after reindex+ffill. "
                f"This usually means the 1h data starts before the daily "
                f"indicator warmup period ends. "
                f"1h start: {df_out.index[0].date()}  |  "
                f"1d features start: {daily_sub.index[0].date()}"
            )
        else:
            logger.info(
                f"Macro merge: {len(daily_reindexed.columns)} daily features "
                f"merged into {len(df_out):,} rows — no NaN ✓"
            )

        return df_out

    def _save(self, df: pd.DataFrame, symbol: str, primary_tf: str):
        safe     = symbol.replace("/", "_").replace("=", "").replace("^", "")
        out_path = cfg.DATA_DIR / "processed" / f"{safe}_{primary_tf}_features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        logger.info(
            f"Feature matrix saved → {out_path.name}  ({df.shape})"
        )