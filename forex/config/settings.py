# forex/config/settings.py
"""
Forex module configuration — AUD/USD.
Mirrors crypto/config/settings.py structure exactly.
Data sourced from yfinance (no API key required).
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR   = BASE_DIR / "logs"

for _dir in [DATA_DIR / "raw", DATA_DIR / "processed",
             DATA_DIR / "splits", MODEL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Asset ─────────────────────────────────────────────────────────────────────
SYMBOL     = os.getenv("FOREX_SYMBOL",     "AUDUSD=X")  # Yahoo Finance ticker
DISPLAY    = "AUD/USD"                                   # human-readable name
EXCHANGE   = "forex"
PRIMARY_TF = os.getenv("PRIMARY_TF",       "1h")
HIGHER_TF  = os.getenv("HIGHER_TF",        "1d")
START_DATE = os.getenv("START_DATE",        "2018-01-01")

# ── Trading parameters ────────────────────────────────────────────────────────
# Forex: spread-based commission (~1 pip = 0.0001 for AUDUSD)
# At 1.0 lot ($100k notional), 1 pip = $10 profit/loss
INITIAL_CAPITAL   = float(os.getenv("INITIAL_CAPITAL",   "10000.0"))
COMMISSION        = float(os.getenv("COMMISSION",        "0.00010"))  # 1 pip spread
SLIPPAGE          = float(os.getenv("SLIPPAGE",          "0.00005"))  # 0.5 pip
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.95"))
RISK_PER_TRADE    = float(os.getenv("RISK_PER_TRADE",    "0.01"))     # 1% — tighter for FX

PIP_SIZE = 0.0001    # standard pip for AUDUSD

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ── Market hours ──────────────────────────────────────────────────────────────
# Forex: Mon 00:00 UTC → Fri 22:00 UTC (no weekend bars)
# yfinance returns weekend gaps — handled in loader validation
MARKET_HOURS_24_5 = True
