# crypto/src/strategies/position_manager.py
"""
Signal → Position conversion.

Signals are instantaneous (fire once per bar).
Positions represent what you are currently holding (persist between bars).

Signal series:  +1 / 0 / -1  (sparse — mostly 0)
Position series: +1 / 0 / -1  (dense — reflects current holding)
"""

import pandas as pd
import numpy as np
from loguru import logger


def signals_to_positions(
    signals: pd.Series,
    mode: str = "long_only",
) -> pd.Series:
    """
    Convert raw entry/exit signals into a held-position series.

    Args:
        signals: Series of {-1, 0, +1} indexed by timestamp.
        mode:    "long_only"  — only long positions, -1 signal = exit
                 "long_short" — both long and short positions

    Returns:
        pd.Series of {-1, 0, +1} with same index:
            +1 = currently long
             0 = flat (no position)
            -1 = currently short

    Example (long_only):
        signals:   0  0  1  0  0 -1  0  1  0
        positions: 0  0  1  1  1  0  0  1  1
    """
    assert mode in ("long_only", "long_short"), f"Unknown mode: {mode}"
    assert signals.isin([-1, 0, 1]).all(), "Signals must be in {-1, 0, 1}"

    positions = pd.Series(0, index=signals.index, dtype=int)
    current   = 0

    for i, sig in enumerate(signals):
        if mode == "long_only":
            if sig == 1:
                current = 1    # enter long
            elif sig == -1:
                current = 0    # exit to flat (no shorting)
        elif mode == "long_short":
            if sig == 1:
                current = 1    # enter/switch to long
            elif sig == -1:
                current = -1   # enter/switch to short
        positions.iloc[i] = current

    _log_position_stats(positions, mode)
    return positions


def apply_position_sizing(
    positions: pd.Series,
    df: pd.DataFrame,
    risk_pct: float = 0.02,          # risk 2% of capital per trade
    atr_multiplier: float = 2.0,     # stop-loss = entry ± (atr_multiplier × ATR)
) -> pd.Series:
    """
    Scale position size based on ATR-derived risk.
    Returns a Series of position sizes in [0, 1] representing
    fraction of capital to deploy.

    Formula:
        stop_distance = atr_multiplier × ATR14
        position_size = risk_pct / (stop_distance / close)

    Capped at 1.0 (100% of capital) to prevent over-leveraging.
    """
    if "atr_14" not in df.columns:
        logger.warning(
            "atr_14 not found — using fixed 100% position size. "
            "Run TechnicalFeatures.compute() first."
        )
        return positions.abs().astype(float)

    atr          = df["atr_14"]
    close        = df["close"]
    stop_dist    = atr_multiplier * atr
    stop_pct     = stop_dist / close

    # Avoid division by zero on zero-atr rows
    raw_size = risk_pct / stop_pct.replace(0, np.nan)
    raw_size = raw_size.clip(upper=1.0).fillna(1.0)

    # Apply only where we have a position
    sized = positions.abs().astype(float) * raw_size
    sized = sized.clip(upper=1.0)

    logger.info(
        f"Position sizing: mean={sized[sized>0].mean():.2%}  "
        f"min={sized[sized>0].min():.2%}  max={sized[sized>0].max():.2%}"
    )
    return sized


def _log_position_stats(positions: pd.Series, mode: str):
    total  = len(positions)
    n_long = (positions ==  1).sum()
    n_flat = (positions ==  0).sum()
    n_short= (positions == -1).sum()
    pct_invested = n_long / total * 100

    logger.info(
        f"Positions [{mode}]: "
        f"LONG {n_long:,} ({n_long/total*100:.1f}%) | "
        f"FLAT {n_flat:,} ({n_flat/total*100:.1f}%) | "
        f"SHORT {n_short:,} ({n_short/total*100:.1f}%) | "
        f"Time invested: {pct_invested:.1f}%"
    )