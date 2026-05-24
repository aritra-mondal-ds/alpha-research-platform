#!/usr/bin/env python3
# crypto/diagnose.py
"""
Quick diagnostic — checks all imports required for --mode wfv and --mode report.
Run from crypto/ directory:
    python diagnose.py

If any import fails, it will show exactly what's missing.
Copy missing files from the project delivery outputs into place, then re-run.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS = [
    # (import_path, label)
    ("src.backtesting.walk_forward",  "WalkForwardValidator"),
    ("src.backtesting.metrics",       "check_overfitting, monte_carlo"),
    ("src.backtesting.simple_backtest","SimpleBacktester"),
    ("src.models.xgboost_model",      "XGBoostTradingModel"),
    ("src.models.xgboost_v2",         "XGBoostTradingModelV2"),
    ("src.models.feature_selector",   "get_stationary_features"),
    ("src.utils.experiment_tracker",  "ExperimentTracker"),
    ("src.utils.report_generator",    "generate_report"),
    ("src.features.pipeline",         "FeaturePipeline"),
    ("src.features.technical",        "TechnicalFeatures"),
    ("src.data.splitter",             "temporal_split"),
    ("src.data.loader",               "DataLoader"),
    ("src.strategies.ema_rsi",        "EMACrossoverRSI"),
    ("src.strategies.multi_indicator","MultiIndicatorConfluence"),
]

FILE_CHECKS = [
    ROOT / "src" / "backtesting" / "walk_forward.py",
    ROOT / "src" / "backtesting" / "metrics.py",
    ROOT / "src" / "backtesting" / "simple_backtest.py",
    ROOT / "src" / "models"      / "xgboost_model.py",
    ROOT / "src" / "models"      / "xgboost_v2.py",
    ROOT / "src" / "models"      / "feature_selector.py",
    ROOT / "src" / "utils"       / "experiment_tracker.py",
    ROOT / "src" / "utils"       / "report_generator.py",
    ROOT / "src" / "features"    / "pipeline.py",
    ROOT / "src" / "features"    / "technical.py",
    ROOT / "src" / "data"        / "splitter.py",
    ROOT / "src" / "data"        / "loader.py",
    ROOT / "models" / "xgboost"   / "model.json",
    ROOT / "models" / "xgboost_v2" / "model.json",
]

sep = "═" * 60
print(f"\n{sep}")
print(f"  CRYPTO MODULE DIAGNOSTIC")
print(f"  Working dir: {ROOT}")
print(sep)

# ── File existence checks ──────────────────────────────────────────────────
print(f"\n  FILE CHECKS:")
any_missing = False
for fpath in FILE_CHECKS:
    exists = fpath.exists()
    status = "✓" if exists else "✗ MISSING"
    rel    = fpath.relative_to(ROOT)
    print(f"  {'['+status+']':<12} {rel}")
    if not exists:
        any_missing = True

# ── Import checks ──────────────────────────────────────────────────────────
print(f"\n  IMPORT CHECKS:")
import_errors = []
for mod_path, label in CHECKS:
    try:
        __import__(mod_path)
        print(f"  [✓ OK      ] {mod_path:<40} ({label})")
    except Exception as e:
        print(f"  [✗ FAIL    ] {mod_path:<40} → {e}")
        import_errors.append((mod_path, str(e)))

# ── Model file checks ──────────────────────────────────────────────────────
print(f"\n  MODEL CHECKS:")
for name, path in [
    ("XGBoost v1", ROOT / "models" / "xgboost"    / "model.json"),
    ("XGBoost v2", ROOT / "models" / "xgboost_v2" / "model.json"),
    ("LSTM",       ROOT / "models" / "lstm"        / "model.pt"),
    ("PPO",        ROOT / "models" / "ppo"         / "model.zip"),
]:
    status = "✓ found" if path.exists() else "✗ not trained yet"
    print(f"  {name:<15} {status}")

# ── Feature matrix check ───────────────────────────────────────────────────
print(f"\n  DATA CHECKS:")
import config.settings as cfg
feat_path = cfg.DATA_DIR / "processed" / "BTC_USDT_1h_features.parquet"
if feat_path.exists():
    import pandas as pd
    df = pd.read_parquet(feat_path, engine="pyarrow")
    print(f"  Feature matrix   ✓  {len(df):,} rows | {df.shape[1]} cols")
    print(f"                      {df.index[0].date()} → {df.index[-1].date()}")
else:
    print(f"  Feature matrix   ✗ MISSING — run: python main.py --mode features")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{sep}")
if import_errors or any_missing:
    print(f"  RESULT: Issues found")
    if any_missing:
        print(f"  → Copy missing files from project delivery into the correct paths")
    if import_errors:
        print(f"  → Fix import errors shown above before running --mode wfv")
else:
    print(f"  RESULT: All checks passed ✓")
    print(f"  → Safe to run: python main.py --mode wfv --trials 20")
    print(f"  → Safe to run: python main.py --mode report")
print(f"{sep}\n")