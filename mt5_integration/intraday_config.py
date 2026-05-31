"""
Configuration for intraday MT5 trading.

This version is designed for continuous monitoring with bar-confirmed
entries and active position management. It is intentionally more conservative
than the hourly trader because trading frequency is higher.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CRYPTO_ROOT = REPO_ROOT / "crypto"

INTRADAY_ASSETS = {
    "BTCUSD": {
        "display": "BTC/USDT",
        "mt5_symbol": "#BTCUSD",
        "signal_timeframe": "5m",
        "bars_needed": 400,
        "strategy": "xgboost_v2",
        "model_path": CRYPTO_ROOT / "models" / "xgboost_v2",
        "confidence_threshold": 0.50,
        "stop_atr_mult": 1.8,
        "tp_atr_mult": 2.8,
        "risk_pct": 0.004,
        "max_open_trades": 1,
        "cooldown_minutes": 0,
        "max_trades_per_day": 1000,
        "max_spread_points": None,
        "allow_long": True,
        "allow_short": True,
    },
    "AUDUSD": {
        "display": "AUD/USD",
        "mt5_symbol": "AUDUSD",
        "signal_timeframe": "5m",
        "bars_needed": 400,
        "strategy": "ema_rsi",
        "model_path": None,
        "confidence_threshold": 0.0,
        "stop_atr_mult": 1.6,
        "tp_atr_mult": 2.4,
        "risk_pct": 0.0025,
        "max_open_trades": 1,
        "cooldown_minutes": 0,
        "max_trades_per_day": 1000,
        "max_spread_points": 25,
        "allow_long": True,
        "allow_short": False,
    },
    "XAUUSD": {
        "display": "XAU/USD (Gold)",
        "mt5_symbol": "XAUUSD",
        "signal_timeframe": "5m",
        "bars_needed": 400,
        "strategy": "xgboost_v2",
        "model_path": REPO_ROOT / "commodities" / "models" / "xgboost_v2",
        "confidence_threshold": 0.10,
        "stop_atr_mult": 1.8,
        "tp_atr_mult": 2.5,
        "risk_pct": 0.003,
        "max_open_trades": 1,
        "cooldown_minutes": 0,
        "max_trades_per_day": 1000,
        "max_spread_points": 60,
        "allow_long": True,
        "allow_short": True,
    },
}

MT5_TIMEFRAME_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 16385,
    "4h": 16388,
    "1d": 16408,
}

POLL_INTERVAL_SECONDS = 10
HEARTBEAT_MINUTES = 5
USE_BAR_CLOSE_CONFIRMATION = True
REQUIRE_NEW_BAR_FOR_ENTRY = True
MANAGE_OPEN_POSITIONS_EVERY_LOOP = True

MAX_TOTAL_RISK_PCT = 0.10
MAX_DRAWDOWN_PCT = 0.05
MAX_DAILY_LOSS_PCT = None
MIN_EQUITY = 500.0
MAX_NEW_TRADES_PER_CYCLE = 5

ENABLE_TRAILING_STOP = False
TRAIL_AFTER_R_MULT = 1.0
TRAIL_TO_R_MULT = 0.25
ENABLE_BREAK_EVEN = False
BREAK_EVEN_AFTER_R_MULT = 1.0

MAGIC_NUMBER = 20260530
SLIPPAGE_POINTS = 20
ORDER_COMMENT = "TradingBot_Intraday"

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
