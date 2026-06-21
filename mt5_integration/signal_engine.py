# mt5_integration/signal_engine.py
"""
Signal engine — 1h XGBoost v2 strategy.

Routing:
    #BTCUSD → XGBoost v2 (1h), confidence_threshold=0.45
    AUDUSD  → EMA(50) + RSI(75) + EMA-50 daily filter, long_only
    XAUUSD  → XGBoost v2 (1h), confidence_threshold=0.0 (pure argmax)

Weekend closure guard (new):
    AUDUSD/XAUUSD return HOLD with strategy="market_closed" during the
    FX/Gold weekend (Fri 21:00 UTC -> Sun 21:00 UTC). Prevents acting on
    stale data MT5 keeps returning when no new bars are forming.
"""

import sys
import pickle
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

_REPO_ROOT  = Path(__file__).resolve().parents[1]
_CRYPTO_SRC = _REPO_ROOT / "crypto"
if str(_CRYPTO_SRC) not in sys.path:
    sys.path.append(str(_CRYPTO_SRC))

from mt5_config import ASSETS, AUDUSD_EMA_FAST, AUDUSD_RSI_THRESHOLD, WEEKEND_CLOSED_ASSETS


@dataclass
class Signal:
    asset:         str
    direction:     int    # -1 SHORT | 0 HOLD | 1 LONG
    confidence:    float
    strategy:      str
    atr_pct:       float
    current_price: float


def _is_fx_weekend_closed() -> bool:
    """
    True if the global FX/Gold market is currently closed.
    Closed window: Friday 21:00 UTC -> Sunday 21:00 UTC.
    weekday(): Monday=0 ... Sunday=6
    """
    now = datetime.now(timezone.utc)
    wd, hr = now.weekday(), now.hour

    if wd == 4 and hr >= 21:   # Friday from 21:00 UTC
        return True
    if wd == 5:                # all day Saturday
        return True
    if wd == 6 and hr < 21:    # Sunday before 21:00 UTC
        return True
    return False


class SignalEngine:

    def __init__(self):
        from src.features.technical import TechnicalFeatures
        from src.features.patterns  import ChartPatternFeatures
        self._tech     = TechnicalFeatures()
        self._patterns = ChartPatternFeatures()
        self._models   = {}
        self._load_models()

    def _load_models(self):
        for symbol, cfg in ASSETS.items():
            if cfg["strategy"] != "xgboost_v2" or cfg["model_path"] is None:
                continue
            model_dir = Path(cfg["model_path"])
            if not (model_dir / "model.json").exists():
                logger.warning(f"{symbol}: XGBoost v2 not found at {model_dir}.")
                continue
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(str(model_dir / "model.json"))
                with open(model_dir / "meta.pkl", "rb") as f:
                    meta = pickle.load(f)
                self._models[symbol] = {
                    "model":        model,
                    "feature_cols": meta["feature_cols"],
                }
                logger.info(
                    f"{symbol}: XGBoost v2 (1h) loaded ✓ "
                    f"({len(meta['feature_cols'])} features)"
                )
            except Exception as e:
                logger.error(f"{symbol}: model load failed — {e}")

        n_ml = len(self._models)
        n_rb = len(ASSETS) - n_ml
        logger.info(
            f"SignalEngine ready | XGBoost v2 (1h): {n_ml} | EMA+RSI: {n_rb}"
        )

    def generate(
        self, symbol: str, df_primary: pd.DataFrame, df_daily: pd.DataFrame
    ) -> Signal:
        cfg      = ASSETS[symbol]
        close_px = float(df_primary["close"].iloc[-1])

        # ── Weekend closure guard ─────────────────────────────────────────────
        if symbol in WEEKEND_CLOSED_ASSETS and _is_fx_weekend_closed():
            logger.debug(f"{symbol}: FX/Gold market closed (weekend) — HOLD.")
            return Signal(symbol, 0, 0.0, "market_closed", 0.0, close_px)

        df_feat = self._build_features(df_primary, df_daily)

        if len(df_feat) == 0:
            logger.warning(
                f"{symbol}: Empty features (need ≥250 bars, "
                f"got {len(df_primary)}). Returning HOLD."
            )
            return Signal(symbol, 0, 0.0, "insufficient_data", 0.0, close_px)

        atr_pct = float(df_feat["atr_pct"].iloc[-1]) \
                  if "atr_pct" in df_feat.columns else 0.0

        if cfg["strategy"] == "xgboost_v2" and symbol in self._models:
            direction, confidence = self._predict_xgboost(symbol, df_feat, cfg)
            strategy = "xgboost_v2"
        else:
            direction, confidence = self._predict_ema_rsi(symbol, df_feat)
            strategy = "ema_rsi"

        logger.info(
            f"{symbol}: "
            f"{'LONG' if direction==1 else 'SHORT' if direction==-1 else 'HOLD'}"
            f" | conf={confidence:.3f} | {strategy}"
            f" | px={close_px:.5f} | atr_pct={atr_pct:.5f}"
        )
        return Signal(symbol, direction, confidence, strategy, atr_pct, close_px)

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

    def _build_features(
        self, df_primary: pd.DataFrame, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        df = self._tech.compute(df_primary.copy())
        if len(df) == 0:
            return df
        df = self._patterns.compute(df)

        df_d = self._tech.compute(df_daily.copy())
        if len(df_d) > 0:
            cols  = ["ema_50","ema_200","rsi_14","atr_pct","bb_pct",
                     "bb_width","adx","macd_hist","realised_vol_24"]
            avail = [c for c in cols if c in df_d.columns]
            sub   = df_d[avail].add_prefix("d1_")
            ri    = sub.reindex(df.index, method="ffill")
            for col in ri.columns:
                df[col] = ri[col].values

        df.dropna(inplace=True)
        return df

    def _predict_xgboost(
        self, symbol: str, df_feat: pd.DataFrame, cfg: dict
    ) -> tuple[int, float]:
        m         = self._models[symbol]
        cols      = [f for f in m["feature_cols"] if f in df_feat.columns]
        X         = df_feat.iloc[[-1]][cols]
        proba     = m["model"].predict_proba(X)[0]
        raw       = int(np.argmax(proba))
        conf      = float(proba[raw])
        threshold = cfg["confidence_threshold"]

        if threshold > 0 and conf < threshold:
            raw = 1

        return {0: -1, 1: 0, 2: 1}[raw], conf

    def _predict_ema_rsi(
        self, symbol: str, df_feat: pd.DataFrame
    ) -> tuple[int, float]:
        from src.strategies.ema_rsi import EMACrossoverRSI

        strat = EMACrossoverRSI(
            AUDUSD_EMA_FAST, AUDUSD_RSI_THRESHOLD,
            use_ema50_filter=True,
            long_only=True,
        )
        signals   = strat.generate_signals(df_feat)
        raw       = int(signals.iloc[-1])
        # long_only fix: -1 means "exit any long", not "open a short".
        # Clamp -1 -> 0 (FLAT) so AUDUSD only ever goes LONG or stays flat.
        direction = max(0, raw)
        return direction, 0.0