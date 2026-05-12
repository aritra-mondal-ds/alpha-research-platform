# crypto/src/models/feature_selector.py
"""
Feature selection and stationarity analysis for trading models.

Core problem this solves:
    Absolute price-level features (EMA values, OBV, VWAP, BB bands)
    have completely different numerical ranges across market regimes.
    At BTC $9k, ema_50 ≈ 9000. At BTC $90k, ema_50 ≈ 90000.
    Tree-based models learn splits like "ema_50 > 40000" — which is
    really just "is this after 2021?" — a timestamp encoded as price.

    Solution: use ONLY dimensionless, scale-invariant features.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg


# ── Feature classification ────────────────────────────────────────────────────
NON_STATIONARY = [
    "ema_9", "ema_21", "ema_50", "ema_200",      # absolute EMA levels
    "macd", "macd_signal",                         # scale with price
    "bb_upper", "bb_mid", "bb_lower",              # absolute BB levels
    "kc_upper", "kc_lower",                        # absolute KC levels
    "atr_14",                                      # use atr_pct instead
    "vwap",                                        # use vwap_gap instead
    "obv",                                         # cumulative, always grows
    "volume_sma_20",                               # use volume_ratio instead
    "d1_ema_50", "d1_ema_200",                     # absolute daily EMAs
]

STATIONARY = [
    # EMA crossover flags (binary, dimensionless)
    "ema_cross_9_21", "ema_cross_21_50", "ema_cross_50_200",
    # Price distance from EMA (percent — scale-invariant)
    "price_vs_ema50", "price_vs_ema200",
    # MACD histogram (relative), cross flag
    "macd_hist", "macd_cross",
    # ADX — bounded [0,100]
    "adx", "adx_pos", "adx_neg",
    # RSI — bounded [0,100]
    "rsi_14", "rsi_7", "rsi_overbought", "rsi_oversold",
    # Stochastic — bounded [0,100]
    "stoch_k", "stoch_d",
    # Williams %R — bounded [-100,0]
    "williams_r",
    # Rate of Change (percentage)
    "roc_5", "roc_10",
    # Bollinger Band position & width (dimensionless)
    "bb_width", "bb_pct",
    # ATR as % of price (scale-invariant)
    "atr_pct",
    # Keltner Channel position (dimensionless)
    "kc_pct",
    # Volume ratio vs moving average (dimensionless)
    "volume_ratio",
    # VWAP gap as % (scale-invariant)
    "vwap_gap",
    # MFI — bounded [0,100]
    "mfi_14",
    # Returns (dimensionless)
    "log_return", "return_3", "return_5", "return_24",
    # Price anatomy ratios (dimensionless)
    "high_low_spread", "close_pct_range",
    # Realised volatility — scale-invariant
    "realised_vol_24", "realised_vol_168",
    # Daily macro — dimensionless versions only
    "d1_rsi_14", "d1_atr_pct", "d1_bb_pct", "d1_bb_width",
    "d1_adx", "d1_macd_hist", "d1_realised_vol_24",
    # Pattern flags (binary)
    "pat_doji", "pat_hammer", "pat_inv_hammer", "pat_shooting_star",
    "pat_marubozu_bull", "pat_marubozu_bear", "pat_spinning_top",
    "pat_engulf_bull", "pat_engulf_bear",
    "pat_harami_bull", "pat_harami_bear",
    "pat_piercing", "pat_dark_cloud",
    "pat_any_bull", "pat_any_bear",
]


def get_stationary_features(df: pd.DataFrame) -> list:
    """Return list of stationary features present in df."""
    available = set(df.columns)
    selected  = [f for f in STATIONARY if f in available]
    removed   = [f for f in NON_STATIONARY if f in available]
    logger.info(
        f"Feature selection: {len(selected)} stationary kept | "
        f"{len(removed)} non-stationary removed"
    )
    if removed:
        logger.info(f"Removed: {removed}")
    return selected


def analyse_stationarity(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Split dataset into thirds; measure how much feature means drift.
    Instability = (max_period_mean - min_period_mean) / abs_mean
    > 0.5 = likely non-stationary.
    """
    n     = len(df)
    third = n // 3
    parts = [df.iloc[:third], df.iloc[third:2*third], df.iloc[2*third:]]

    records = []
    for col in feature_cols:
        try:
            means = [p[col].mean() for p in parts]
            mean_range  = max(means) - min(means)
            mean_abs    = abs(np.mean(means)) + 1e-9
            instability = mean_range / mean_abs
            records.append({
                "feature":     col,
                "instability": round(instability, 4),
                "early_mean":  round(means[0], 4),
                "mid_mean":    round(means[1], 4),
                "late_mean":   round(means[2], 4),
                "verdict":     "✗ UNSTABLE" if instability > 0.5 else "✓ STABLE",
            })
        except Exception:
            pass

    return pd.DataFrame(records).sort_values("instability", ascending=False)


def print_feature_analysis(df: pd.DataFrame, feat_cols: list):
    """Print stationarity report for feature set."""
    stats = analyse_stationarity(df, feat_cols)
    sep   = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  FEATURE STATIONARITY ANALYSIS  ({len(feat_cols)} features)")
    print(f"  Instability > 0.5 = likely non-stationary (memorisation risk)")
    print(f"{'═'*72}")
    print(f"  {'Feature':<30} {'Instability':>12}  {'Early':>9}  {'Mid':>9}  {'Late':>9}  Verdict")
    print(sep)
    n_unstable = 0
    for _, row in stats.iterrows():
        print(
            f"  {row['feature']:<30} {row['instability']:>12.4f}  "
            f"{row['early_mean']:>9.4f}  {row['mid_mean']:>9.4f}  "
            f"{row['late_mean']:>9.4f}  {row['verdict']}"
        )
        if "UNSTABLE" in str(row["verdict"]):
            n_unstable += 1
    print(sep)
    print(f"  Unstable: {n_unstable}/{len(stats)} features")
    print(f"{'═'*72}\n")