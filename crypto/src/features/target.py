# crypto/src/features/target.py
"""
Target variable construction for supervised learning.

CRITICAL LOOKAHEAD RULE:
    Labels at time t use future prices df["close"].shift(-horizon).
    These columns must NEVER be used as input features.
    They are only valid as training targets (y), not features (X).

    The last `horizon` rows will have NaN targets and must be dropped
    before training. This is handled automatically in the pipeline.
"""

import numpy as np
import pandas as pd
from loguru import logger


# Signal class constants — consistent across all modules
LONG  =  1
HOLD  =  0
SHORT = -1


class TargetBuilder:

    def build_classification(
        self,
        df: pd.DataFrame,
        horizon: int   = 12,       # bars ahead (12h for 1h data)
        threshold: float = 0.015,  # 1.5% move required for a signal
    ) -> pd.Series:
        """
        3-class label for classification models (XGBoost, LSTM).

        Logic:
            future_return = close[t+horizon] / close[t] - 1

            +1 (LONG)  if future_return > +threshold
            -1 (SHORT) if future_return < -threshold
             0 (HOLD)  otherwise

        Args:
            df:        Feature DataFrame with 'close' column.
            horizon:   Number of bars to look ahead.
            threshold: Minimum price move to generate a directional signal.

        Returns:
            pd.Series of {-1, 0, 1}, NaN for last `horizon` rows.
        """
        future_return = df["close"].shift(-horizon) / df["close"] - 1

        conditions = [
            future_return >  threshold,
            future_return < -threshold,
        ]
        choices = [LONG, SHORT]

        raw = np.select(conditions, choices, default=HOLD).astype(float)
        target = pd.Series(raw, index=df.index, name=f"target_clf_{horizon}h")

        # Invalidate the last `horizon` rows — no valid future exists
        target.iloc[-horizon:] = np.nan

        # Log class distribution
        valid = target.dropna()
        counts = valid.value_counts().sort_index()
        total  = len(valid)
        logger.info(
            f"Target [{horizon}h | ±{threshold*100:.1f}%] — "
            f"LONG: {counts.get(LONG,0):,} ({counts.get(LONG,0)/total*100:.1f}%) | "
            f"HOLD: {counts.get(HOLD,0):,} ({counts.get(HOLD,0)/total*100:.1f}%) | "
            f"SHORT: {counts.get(SHORT,0):,} ({counts.get(SHORT,0)/total*100:.1f}%)"
        )

        return target

    def build_regression(
        self,
        df: pd.DataFrame,
        horizon: int = 12,
    ) -> pd.Series:
        """
        Continuous log-return over `horizon` bars.
        For regression models or reward shaping in RL.

        Returns:
            pd.Series of float, NaN for last `horizon` rows.
        """
        target = np.log(
            df["close"].shift(-horizon) / df["close"]
        ).rename(f"target_reg_{horizon}h")
        target.iloc[-horizon:] = np.nan
        logger.info(
            f"Regression target [{horizon}h] — "
            f"mean: {target.mean():.5f}, std: {target.std():.5f}"
        )
        return target

    def build_multi_horizon(
        self,
        df: pd.DataFrame,
        horizons: list = [6, 12, 24],
        threshold: float = 0.015,
    ) -> pd.DataFrame:
        """
        Build classification targets for multiple horizons simultaneously.
        Useful for multi-task learning or horizon comparison.

        Returns:
            DataFrame with one target column per horizon.
        """
        targets = {}
        for h in horizons:
            targets[f"target_clf_{h}h"] = self.build_classification(
                df, horizon=h, threshold=threshold
            )
        return pd.DataFrame(targets, index=df.index)

    def build_triple_barrier(
        self,
        df: pd.DataFrame,
        pt_mult: float = 2.0,
        sl_mult: float = 1.5,
        max_holding: int = 12,
        vol_col: str = "atr_pct",
    ) -> pd.Series:
        """
        Institutional Triple-Barrier Method (Marcos López de Prado).
        Dynamically scales upper (take profit) and lower (stop loss) barriers
        based on volatility (e.g. ATR % or rolling returns standard deviation).

        Labels:
            +1 (LONG) : Upper barrier touched first
            -1 (SHORT): Lower barrier touched first
             0 (HOLD) : Vertical barrier (max holding limit) reached before touch
        """
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        if vol_col in df.columns:
            vol = df[vol_col].values
        else:
            returns = np.log(df["close"] / df["close"].shift(1)).fillna(0)
            vol = returns.rolling(24).std().fillna(returns.std()).values
            vol = np.maximum(vol, 0.005)

        n = len(df)
        labels = np.zeros(n, dtype=float)

        for i in range(n - max_holding):
            p0 = close[i]
            v = vol[i]
            if np.isnan(v) or v <= 0:
                v = 0.01

            upper = p0 * (1.0 + pt_mult * v)
            lower = p0 * (1.0 - sl_mult * v)

            f_high = high[i + 1 : i + 1 + max_holding]
            f_low = low[i + 1 : i + 1 + max_holding]

            touch_upper = np.where(f_high >= upper)[0]
            touch_lower = np.where(f_low <= lower)[0]

            first_upper = touch_upper[0] if len(touch_upper) > 0 else 999999
            first_lower = touch_lower[0] if len(touch_lower) > 0 else 999999

            if first_upper < first_lower:
                labels[i] = LONG
            elif first_lower < first_upper:
                labels[i] = SHORT
            else:
                labels[i] = HOLD

        labels[-max_holding:] = np.nan
        target = pd.Series(labels, index=df.index, name=f"target_tb_{max_holding}h")

        valid = target.dropna()
        counts = valid.value_counts().sort_index()
        total = len(valid)
        if total > 0:
            logger.info(
                f"Triple-Barrier Target [{max_holding}h | pt={pt_mult}x | sl={sl_mult}x] — "
                f"LONG: {counts.get(LONG,0):,} ({counts.get(LONG,0)/total*100:.1f}%) | "
                f"HOLD: {counts.get(HOLD,0):,} ({counts.get(HOLD,0)/total*100:.1f}%) | "
                f"SHORT: {counts.get(SHORT,0):,} ({counts.get(SHORT,0)/total*100:.1f}%)"
            )
        return target


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_feature_target_split(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract X and y, dropping rows where target is NaN.
    Ensures features and target are perfectly aligned.
    """
    combined = df[feature_cols + [target_col]].dropna()
    X = combined[feature_cols]
    y = combined[target_col].astype(int)
    logger.info(
        f"X shape: {X.shape} | "
        f"y distribution: {y.value_counts().sort_index().to_dict()}"
    )
    return X, y