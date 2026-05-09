# crypto/config/settings.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env if present

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent   # .../crypto/
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR   = BASE_DIR / "logs"

# Create directories if they don't exist
for _dir in [DATA_DIR / "raw", DATA_DIR / "processed",
             DATA_DIR / "splits", MODEL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Exchange / Symbol ─────────────────────────────────────────────────────────
EXCHANGE    = os.getenv("EXCHANGE",    "binance")
SYMBOL      = os.getenv("SYMBOL",     "BTC/USDT")
PRIMARY_TF  = os.getenv("PRIMARY_TF", "1h")
HIGHER_TF   = os.getenv("HIGHER_TF",  "1d")
START_DATE  = os.getenv("START_DATE", "2020-01-01")

# ── Trading Parameters ────────────────────────────────────────────────────────
INITIAL_CAPITAL   = float(os.getenv("INITIAL_CAPITAL",   "10000.0"))
COMMISSION        = float(os.getenv("COMMISSION",        "0.001"))   # 0.1% Binance taker
SLIPPAGE          = float(os.getenv("SLIPPAGE",          "0.0005"))  # 0.05%
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.95"))
RISK_PER_TRADE    = float(os.getenv("RISK_PER_TRADE",    "0.02"))

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))