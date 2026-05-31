# mt5_integration/mt5_config.py
"""
High-frequency scalping configuration — 5-minute bars, 100+ trades/day target.

Key changes vs hourly config:
    - primary_tf: "1h" → "5m"     (288 bars/day vs 24)
    - bars_needed: 300 → 600      (EMA-200 warmup on 5m ≈ 50h data)
    - strategy: "ema_rsi" for all (XGBoost trained on 1h — not valid on 5m)
    - stop_atr_mult: 2.0 → 1.0    (tighter stops for scalping)
    - tp_atr_mult:   4.0 → 2.0    (2:1 R/R — quicker profit-taking)
    - risk_pct: 0.01 → 0.02       (2% per trade for higher returns)
    - max_open_trades: 1 → 3      (allow scaling / pyramiding)
    - EMA params: (50, 75 RSI) → (9, 40 RSI) (more signals on 5m)
"""

from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
CRYPTO_ROOT = REPO_ROOT / "crypto"

# ── Scalping EMA parameters (used by signal_engine) ──────────────────────────
# Shorter periods generate more crossover signals on 5m bars.
# RSI threshold 40 (vs 50) fires earlier in momentum moves.
EMA_FAST        = 9
EMA_SLOW        = 21
RSI_THRESHOLD   = 40

ASSETS = {
    "#BTCUSD": {
        "display":              "BTC/USD",
        "mt5_symbol":           "#BTCUSD",
        "primary_tf":           "5m",
        "bars_needed":          600,        # 600×5m = 50h — covers EMA-200 warmup
        "strategy":             "ema_rsi",  # ML model trained on 1h, not valid here
        "model_path":           None,
        "confidence_threshold": 0.0,
        "stop_atr_mult":        1.0,        # 1×ATR stop for tight scalping
        "tp_atr_mult":          2.0,        # 2×ATR TP → 2:1 R/R
        "risk_pct":             0.02,       # 2% per trade
        "max_open_trades":      3,
    },
    "AUDUSD": {
        "display":              "AUD/USD",
        "mt5_symbol":           "AUDUSD",
        "primary_tf":           "5m",
        "bars_needed":          600,
        "strategy":             "ema_rsi",
        "model_path":           None,
        "confidence_threshold": 0.0,
        "stop_atr_mult":        1.0,
        "tp_atr_mult":          2.0,
        "risk_pct":             0.015,      # 1.5% for forex
        "max_open_trades":      2,
    },
    "XAUUSD": {
        "display":              "XAU/USD (Gold)",
        "mt5_symbol":           "XAUUSD",
        "primary_tf":           "5m",
        "bars_needed":          600,
        "strategy":             "ema_rsi",
        "model_path":           None,
        "confidence_threshold": 0.0,
        "stop_atr_mult":        1.0,
        "tp_atr_mult":          2.0,
        "risk_pct":             0.02,       # 2% for Gold
        "max_open_trades":      2,
    },
}

# ── Portfolio-level risk limits ───────────────────────────────────────────────
MAX_TOTAL_RISK_PCT = 0.10   # allow up to 10% total open risk (scalping mode)
MAX_DRAWDOWN_PCT   = 0.15   # halt if session drawdown hits 15%
MIN_EQUITY         = 500.0

# ── MT5 execution ─────────────────────────────────────────────────────────────
MAGIC_NUMBER    = 20260524
SLIPPAGE_POINTS = 30        # slightly wider for 5m scalping
ORDER_COMMENT   = "TradingBot_5m"

MT5_TIMEFRAME_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 16385, "4h": 16388, "1d": 16408,
}

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)