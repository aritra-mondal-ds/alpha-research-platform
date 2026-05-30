# mt5_integration/mt5_config.py
"""
Configuration for live MT5 trading.

Named mt5_config (not config) to avoid shadowing crypto/config/settings.py
when both directories are in sys.path simultaneously.

Edit symbol names to match your broker's Market Watch panel.
"""

from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
CRYPTO_ROOT = REPO_ROOT / "crypto"

ASSETS = {
    "BTCUSD": {
        "display":              "BTC/USDT",
        "mt5_symbol":           "#BTCUSD",       # also: "BTCUSDT", "Bitcoin", "#BTC"
        "bars_needed":          300,
        "strategy":             "xgboost_v2",
        "model_path":           CRYPTO_ROOT / "models" / "xgboost_v2",
        "confidence_threshold": 0.45,
        "stop_atr_mult":        2.0,
        "tp_atr_mult":          4.0,
        "risk_pct":             0.01,
        "max_open_trades":      1,
    },
    "AUDUSD": {
        "display":              "AUD/USD",
        "mt5_symbol":           "AUDUSD",
        "bars_needed":          300,
        "strategy":             "ema_rsi",       # ML has no edge on 2yr forex data
        "model_path":           None,
        "confidence_threshold": 0.0,
        "stop_atr_mult":        2.0,
        "tp_atr_mult":          3.0,
        "risk_pct":             0.005,
        "max_open_trades":      1,
    },
    "XAUUSD": {
        "display":              "XAU/USD (Gold)",
        "mt5_symbol":           "XAUUSD",        # also: "GOLD", "XAUUSDm"
        "bars_needed":          300,
        "strategy":             "xgboost_v2",
        "model_path":           REPO_ROOT / "commodities" / "models" / "xgboost_v2",
        "confidence_threshold": 0.0,             # pure argmax
        "stop_atr_mult":        2.0,
        "tp_atr_mult":          3.5,
        "risk_pct":             0.008,
        "max_open_trades":      1,
    },
}

MT5_TIMEFRAME_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 16385, "4h": 16388, "1d": 16408,
}

MAX_TOTAL_RISK_PCT = 0.03
MAX_DRAWDOWN_PCT   = 0.10
MIN_EQUITY         = 500.0

MAGIC_NUMBER     = 20260524
SLIPPAGE_POINTS  = 20
ORDER_COMMENT    = "TradingBot"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)