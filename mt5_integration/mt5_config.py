# mt5_integration/mt5_config.py
"""
1h XGBoost v2 strategy — enhanced after Jun 16-20 performance review.

Jun 16-20 results:
    #BTCUSD : +$1,771.58 combined, ~85% win rate over 28 trades — model healthy
    AUDUSD  : -$212.95 combined (mostly from pre-fix shorts) — long-only fix now active
    XAUUSD  : near breakeven, thin trade count

Changes this revision:
    1. WEEKEND_CLOSED_ASSETS: AUDUSD/XAUUSD now skip signal generation when
       the FX/Gold market is closed (Fri 21:00 UTC -> Sun 21:00 UTC). Prevents
       acting on stale Friday-close data that MT5 keeps returning all weekend.
       BTC is exempt (crypto trades 24/7).

    2. BTC risk_pct: 0.8% -> 1.0%. Sustained 75-95% win rate over 5 trading
       days supports a modest increase. Kept well short of aggressive sizing
       given the small sample size (28 trades) relative to a full edge-validation
       period.
"""

from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
CRYPTO_ROOT = REPO_ROOT / "crypto"

AUDUSD_EMA_FAST      = 50
AUDUSD_RSI_THRESHOLD = 75

# Assets that should not trade while their underlying market is closed.
# FX/Gold market: closes Friday ~21:00 UTC, reopens Sunday ~21:00 UTC.
WEEKEND_CLOSED_ASSETS = {"AUDUSD", "XAUUSD"}

ASSETS = {
    "#BTCUSD": {
        "display":              "BTC/USD",
        "mt5_symbol":           "#BTCUSD",
        "primary_tf":           "1h",
        "bars_needed":          300,
        "strategy":             "xgboost_v2",
        "model_path":           CRYPTO_ROOT / "models" / "xgboost_v2",
        "confidence_threshold": 0.45,
        "stop_atr_mult":        2.0,
        "tp_atr_mult":          4.0,
        "risk_pct":             0.010,   # was 0.008 — modest increase, strong WR track record
        "max_open_trades":      1,
    },
    "AUDUSD": {
        "display":              "AUD/USD",
        "mt5_symbol":           "AUDUSD",
        "primary_tf":           "1h",
        "bars_needed":          300,
        "strategy":             "ema_rsi",
        "model_path":           None,
        "confidence_threshold": 0.0,
        "stop_atr_mult":        2.0,
        "tp_atr_mult":          3.0,
        "risk_pct":             0.003,
        "max_open_trades":      1,
    },
    "XAUUSD": {
        "display":              "XAU/USD (Gold)",
        "mt5_symbol":           "XAUUSD",
        "primary_tf":           "1h",
        "bars_needed":          300,
        "strategy":             "xgboost_v2",
        "model_path":           REPO_ROOT / "commodities" / "models" / "xgboost_v2",
        "confidence_threshold": 0.0,
        "stop_atr_mult":        2.0,
        "tp_atr_mult":          3.5,
        "risk_pct":             0.004,
        "max_open_trades":      1,
    },
}

MAX_TOTAL_RISK_PCT = 0.03
MAX_DRAWDOWN_PCT   = 0.05
MIN_EQUITY         = 500.0

MAGIC_NUMBER    = 20260524
SLIPPAGE_POINTS = 20
ORDER_COMMENT   = "TradingBot_1h"

MT5_TIMEFRAME_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 16385, "4h": 16388, "1d": 16408,
}

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)