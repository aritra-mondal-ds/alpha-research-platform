# mt5_integration/signal_engine.py
"""
Signal engine — loads trained models and generates live trading signals.

Strategy routing:
    BTCUSD  → XGBoost v2 (WFV 100% profitable folds, Sharpe 8.64)
    AUDUSD  → EMA+RSI    (rule-based; ML no edge on 2yr forex data)
    XAUUSD  → XGBoost v2 (pure argmax threshold)

Import note:
    All src.* imports are inside __init__() and methods, NOT at module level.
    This prevents crypto/src/features/pipeline.py from running its module-level
    `import config.settings as cfg` before sys.path is correctly configured.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────────
# Add crypto/ so that "src.*" resolves to crypto/src/
# Use append (not insert) so cwd stays first — mt5_config.py found before
# anything else, and crypto/config/ found correctly by pipeline.py
_REPO_ROOT  = Path(__file__).resolve().parents[1]
_CRYPTO_SRC = _REPO_ROOT / "crypto"
if str(_CRYPTO_SRC) not in sys.path:
    sys.path.append(str(_CRYPTO_SRC))

# mt5_config is safe — no conflict with crypto/config/ package
from mt5_config import ASSETS


@dataclass
class Signal:
    asset:         str    # "BTCUSD"
    direction:     int    # -1 SHORT | 0 HOLD | 1 LONG
    confidence:    float  # max class probability (0.0 for rule-based)
    strategy:      str
    atr_pct:       float  # current ATR% — used by RiskManager
    current_price: float


class SignalEngine:

    def __init__(self):
        # ── Lazy imports: src.* must NOT be at module level ───────────────────
        # pipeline.py, ema_rsi.py, multi_indicator.py all import
        # `config.settings` at module level. If imported before sys.path
        # is ready, Python finds mt5_integration/ as `config` (a file, not
        # a package) and crashes. Importing inside __init__ is safe because
        # sys.path is already configured by this point.
        from src.features.technical     import TechnicalFeatures
        from src.features.patterns      import ChartPatternFeatures

        self._tech     = TechnicalFeatures()
        self._patterns = ChartPatternFeatures()
        self._models   = {}
        self._load_all_models()

    def _load_all_models(self):
        for symbol, cfg in ASSETS.items():
            if cfg["strategy"] != "xgboost_v2" or cfg["model_path"] is None:
                continue
            model_dir = Path(cfg["model_path"])
            if not (model_dir / "model.json").exists():
                logger.warning(
                    f"{symbol}: XGBoost v2 not found at {model_dir}. "
                    f"Falling back to rule-based."
                )
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
                    f"{symbol}: XGBoost v2 loaded "
                    f"({len(meta['feature_cols'])} stationary features)"
                )
            except Exception as e:
                logger.error(f"{symbol}: Model load failed — {e}")

    def generate(
        self, symbol: str, df_1h: pd.DataFrame, df_1d: pd.DataFrame
    ) -> Signal:
        cfg      = ASSETS[symbol]
        df_feat  = self._build_features(df_1h, df_1d)
        close_px = float(df_1h["close"].iloc[-1])

        if len(df_feat) == 0:
            logger.warning(
                f"{symbol}: Empty feature matrix after warmup. "
                f"Need ≥250 bars, got {len(df_1h)}. Returning HOLD."
            )
            return Signal(symbol, 0, 0.0, "insufficient_data", 0.0, close_px)

        atr_pct = float(df_feat["atr_pct"].iloc[-1]) \
                  if "atr_pct" in df_feat.columns else 0.0

        if cfg["strategy"] == "xgboost_v2" and symbol in self._models:
            direction, confidence = self._predict_xgboost(symbol, df_feat, cfg)
            strategy = "xgboost_v2"
        else:
            direction, confidence = self._predict_rule_based(symbol, df_feat)
            strategy = cfg["strategy"]

        logger.info(
            f"{symbol}: {'LONG' if direction==1 else 'SHORT' if direction==-1 else 'HOLD'}"
            f" | conf={confidence:.3f} | strategy={strategy}"
            f" | px={close_px:.5f} | atr_pct={atr_pct:.4f}"
        )
        return Signal(symbol, direction, confidence, strategy, atr_pct, close_px)

    def generate_all(
        self, data: dict
    ) -> dict[str, Signal]:
        signals = {}
        for symbol, (df_1h, df_1d) in data.items():
            try:
                signals[symbol] = self.generate(symbol, df_1h, df_1d)
            except Exception as e:
                logger.error(f"{symbol}: Signal failed — {e}")
                import traceback; traceback.print_exc()
                px = float(df_1h["close"].iloc[-1]) if len(df_1h) > 0 else 0.0
                signals[symbol] = Signal(symbol, 0, 0.0, "error", 0.0, px)
        return signals

    # ── Feature building ──────────────────────────────────────────────────────
    def _build_features(
        self, df_1h: pd.DataFrame, df_1d: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replicates crypto/src/features/pipeline.py without importing it.
        Direct use of TechnicalFeatures + ChartPatternFeatures avoids the
        pipeline.py module-level `import config.settings` that causes the
        namespace crash when run from mt5_integration/.
        """
        # 1h indicators and patterns
        df = self._tech.compute(df_1h.copy())
        if len(df) == 0:
            return df
        df = self._patterns.compute(df)

        # 1d macro indicators — merged via reindex(ffill)
        # (same fix as pipeline.py for NYSE :30-minute offset timestamps)
        df_daily = self._tech.compute(df_1d.copy())
        if len(df_daily) > 0:
            daily_cols = [
                "ema_50", "ema_200", "rsi_14", "atr_pct",
                "bb_pct", "bb_width", "adx", "macd_hist", "realised_vol_24",
            ]
            available  = [c for c in daily_cols if c in df_daily.columns]
            daily_sub  = df_daily[available].add_prefix("d1_")
            daily_ri   = daily_sub.reindex(df.index, method="ffill")
            for col in daily_ri.columns:
                df[col] = daily_ri[col].values

        df.dropna(inplace=True)
        return df

    # ── XGBoost v2 inference ──────────────────────────────────────────────────
    def _predict_xgboost(
        self, symbol: str, df_feat: pd.DataFrame, cfg: dict
    ) -> tuple[int, float]:
        m         = self._models[symbol]
        cols      = [f for f in m["feature_cols"] if f in df_feat.columns]
        X         = df_feat.iloc[[-1]][cols]
        proba     = m["model"].predict_proba(X)[0]   # [P(SHORT), P(HOLD), P(LONG)]
        raw       = int(np.argmax(proba))
        conf      = float(proba[raw])
        threshold = cfg["confidence_threshold"]
        if threshold > 0 and conf < threshold:
            raw = 1                                  # force HOLD
        return {0: -1, 1: 0, 2: 1}[raw], conf

    # ── Rule-based inference ──────────────────────────────────────────────────
    def _predict_rule_based(
        self, symbol: str, df_feat: pd.DataFrame
    ) -> tuple[int, float]:
        # Lazy import — ema_rsi.py imports config.settings at module level
        from src.strategies.ema_rsi         import EMACrossoverRSI
        from src.strategies.multi_indicator import MultiIndicatorConfluence

        if symbol == "AUDUSD":
            strat = EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True)
        else:
            strat = MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)

        sigs = strat.generate_signals(df_feat)
        return int(sigs.iloc[-1]), 0.0