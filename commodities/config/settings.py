# commodities/config/settings.py
"""
Commodities module configuration — XAU/USD (Gold).

Data source: yfinance
Primary symbol: GLD  (SPDR Gold ETF — full hourly history, highly liquid)
Alternative:    GC=F (Gold Futures — limited hourly history via yfinance)

Why GLD instead of GC=F?
    GC=F (front-month futures) has unreliable hourly data through yfinance:
    - History often limited to < 60 days at 1h resolution
    - Futures roll creates price gaps
    GLD tracks Gold spot price with ~0.01% tracking error and has
    full hourly history since 2004 available through yfinance.

Price relationship:
    GLD price ≈ Gold spot price (XAU/USD) / 10
    e.g., Gold at $2,000/oz → GLD ≈ $200/share
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR  = Path(__file__).resolve().parent.parent   # .../commodities/
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR   = BASE_DIR / "logs"

for _dir in [DATA_DIR / "raw", DATA_DIR / "processed",
             DATA_DIR / "splits", MODEL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Asset ─────────────────────────────────────────────────────────────────────
SYMBOL     = os.getenv("COMMODITY_SYMBOL", "GLD")   # Gold ETF — reliable hourly data
DISPLAY    = "XAU/USD (via GLD)"
EXCHANGE   = "commodities"
PRIMARY_TF = os.getenv("PRIMARY_TF",       "1h")
HIGHER_TF  = os.getenv("HIGHER_TF",        "1d")
START_DATE = os.getenv("START_DATE",        "2018-01-01")

# ── Trading parameters ────────────────────────────────────────────────────────
# GLD commission: ~0.02% typical spread on ETF
INITIAL_CAPITAL   = float(os.getenv("INITIAL_CAPITAL",   "10000.0"))
COMMISSION        = float(os.getenv("COMMISSION",        "0.0002"))   # ~0.02%
SLIPPAGE          = float(os.getenv("SLIPPAGE",          "0.0001"))   # ~0.01%
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.90"))
RISK_PER_TRADE    = float(os.getenv("RISK_PER_TRADE",    "0.015"))    # 1.5%

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ── Market hours ──────────────────────────────────────────────────────────────
# GLD trades NYSE hours: Mon-Fri 09:30-16:00 ET (14:30-21:00 UTC)
# yfinance returns only trading hours — no weekend/overnight bars
MARKET_HOURS_NYSE = True