#!/usr/bin/env python3
# crypto/test_imports.py
"""
Run this to find exactly which module-level import crashes main.py.
    python test_imports.py 2>&1

Each import is tested independently. The one that fails shows the error.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("Python:", sys.version)
print("Testing module-level imports for crypto/main.py ...\n")

IMPORTS = [
    ("loguru",                                    "from loguru import logger"),
    ("pandas",                                    "import pandas as pd"),
    ("numpy",                                     "import numpy as np"),
    ("config.settings",                           "import config.settings as cfg"),
    ("src.utils.gpu_check",                       "from src.utils.gpu_check import verify_gpu"),
    ("src.data.fetcher",                          "from src.data.fetcher import OHLCVFetcher"),
    ("src.data.loader",                           "from src.data.loader import DataLoader, display_dataframe"),
    ("src.data.splitter",                         "from src.data.splitter import temporal_split"),
    ("src.features.pipeline",                     "from src.features.pipeline import FeaturePipeline"),
    ("src.strategies.ema_rsi",                    "from src.strategies.ema_rsi import EMACrossoverRSI"),
    ("src.strategies.multi_indicator",            "from src.strategies.multi_indicator import MultiIndicatorConfluence"),
    ("src.strategies.position_manager",           "from src.strategies.position_manager import signals_to_positions"),
    ("src.backtesting.simple_backtest",           "from src.backtesting.simple_backtest import SimpleBacktester"),
    ("src.models.feature_selector",               "from src.models.feature_selector import get_stationary_features"),
]

all_ok = True
for label, stmt in IMPORTS:
    sys.stdout.flush()
    try:
        exec(stmt, {"sys": sys})
        print(f"  ✓  {label}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗  {label}")
        print(f"     ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    sys.stdout.flush()

print()
if all_ok:
    print("All module-level imports OK.")
    print("The crash must be inside a function — run with 2>&1 to see the error.")
    print()
    print("Next: python main.py --mode compare 2>&1")
else:
    print("Fix the failing import above, then re-run.")

sys.stdout.flush()