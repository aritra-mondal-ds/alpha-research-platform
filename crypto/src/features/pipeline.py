# crypto/src/features/pipeline.py
"""
Feature pipeline orchestrator.

Institutional Enhancements:
    - Dynamic Triple-Barrier Method labeling (volatility scaled)
    - Macro daily feature forward-fill with reindex(method='ffill')
    - Strict lookahead bias prevention
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
        symbol:        str   = cfg.SYMBOL,
        primary_tf:    str   = cfg.PRIMARY_TF,
        higher_tf:     str   = cfg.HIGHER_TF,
        horizon:       int   = 12,
        threshold:     float = 0.015,
        target_method: str   = "triple_barrier",
        save:          bool  = True,
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
        if target_method == "triple_barrier":
            logger.info(
                f"Building dynamic Triple-Barrier target [horizon={horizon}h, volatility scaled] ..."
            )
            df_1h["target"] = self.targets.build_triple_barrier(
                df_1h, pt_mult=2.0, sl_mult=1.5, max_holding=horizon
            )
        else:
            logger.info(
                f"Building fixed classification target "
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
        """
        daily_indicator_cols = [
            "ema_50", "ema_200", "rsi_14", "atr_pct",
            "bb_pct", "bb_width", "adx", "macd_hist",
            "realised_vol_24",
        ]
        available = [c for c in daily_indicator_cols if c in df_daily.columns]
        daily_sub = df_daily[available].add_prefix("d1_")

        daily_sub   = daily_sub.sort_index()
        df_out      = df_primary.copy()

        daily_reindexed = daily_sub.reindex(
            df_out.index, method="ffill"
        )

        for col in daily_reindexed.columns:
            df_out[col] = daily_reindexed[col].values

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
