#!/usr/bin/env python3
# Trading-Bot/main.py
"""
Multi-Asset Trading Bot — Top-Level Orchestrator

Usage:
    python main.py --mode status          # module health check
    python main.py --mode fetch-all       # fetch all asset data
    python main.py --mode portfolio       # inverse-vol capital allocation
    python main.py --mode compare-all     # cross-asset performance table
"""

import argparse
import sys
import os
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared"))

from loguru import logger


def _setup_logger():
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        str(log_dir / "portfolio.log"), rotation="20 MB",
        retention="14 days", level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}",
    )


# ── Module registry ───────────────────────────────────────────────────────────
# data_file: symbol.replace("/","_") only — matches BaseFetcher._parquet_path
MODULES = {
    "crypto": {
        "symbol":    "BTC/USDT",
        "data_file": "BTC_USDT_1h.parquet",
        "alt_files": ["BTC_USDT_1h.parquet"],
    },
    "forex": {
        "symbol":    "AUD/USD",
        "data_file": "AUDUSD=X_1h.parquet",
        "alt_files": ["AUDUSD=X_1h.parquet", "AUDUSDX_1h.parquet"],
    },
    "commodities": {
        "symbol":    "XAU/USD (GLD)",
        # GLD has no "/" so filename = "GLD_1h.parquet"
        "data_file": "GLD_1h.parquet",
        "alt_files": ["GLD_1h.parquet", "GC=F_1h.parquet", "GCF_1h.parquet"],
    },
}


def _data_exists(module: str) -> bool:
    """Return True if any recognised data file exists for this module."""
    raw_dir = ROOT / module / "data" / "raw"
    return any(
        (raw_dir / fname).exists()
        for fname in MODULES[module]["alt_files"]
    )


def _subprocess_env() -> dict:
    """Suppress TF/oneDNN DLL conflicts in subprocesses."""
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"]  = "3"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    env["ABSL_LOGTOSTDERR"]      = "0"
    return env


def _run_module(module: str, mode: str, extra_args: list = None) -> int:
    module_dir  = ROOT / module
    module_main = module_dir / "main.py"
    if not module_main.exists():
        logger.warning(f"{module}/main.py not found — skipping")
        return 1
    cmd    = [sys.executable, "main.py", "--mode", mode] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(module_dir), env=_subprocess_env())
    return result.returncode


def step_status():
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  TRADING-BOT — MULTI-ASSET MODULE STATUS")
    print(sep)
    print(f"\n  {'Module':<14} {'Symbol':<16} {'Raw Data':^10} "
          f"{'Features':^10} {'XGB v2':^10}")
    print(f"  {'─'*14} {'─'*16} {'─'*10} {'─'*10} {'─'*10}")

    for module, info in MODULES.items():
        mod_dir  = ROOT / module
        raw_dir  = mod_dir / "data" / "raw"
        raw_ok   = "✓" if any(
            (raw_dir / f).exists() for f in info["alt_files"]
        ) else "✗"
        feat_dir = mod_dir / "data" / "processed"
        feat_ok  = "✓" if feat_dir.exists() and any(feat_dir.glob("*.parquet")) else "✗"
        model_ok = "✓" if (mod_dir / "models" / "xgboost_v2" / "model.json").exists() else "✗"
        sym      = info["symbol"]
        print(f"  {module:<14} {sym:<16} "
              f"{'['+raw_ok+']':^10} {'['+feat_ok+']':^10} {'['+model_ok+']':^10}")

    print(f"\n{sep}")
    print(f"  Run a module:  cd <module> && python main.py --mode full")
    print(f"  Portfolio:     python main.py --mode portfolio")
    print(f"{sep}\n")


def step_fetch_all():
    """
    Fetch data for all modules.
    Skips modules where data already exists (avoids CUDA DLL subprocess crash).
    """
    logger.info("Fetching data for all asset modules ...")

    for module in MODULES:
        if _data_exists(module):
            logger.info(
                f"{module.capitalize():<14} data already exists — skipping ✓"
            )
            continue

        logger.info(f"Fetching {module} (no data found) ...")
        rc = _run_module(module, "fetch")
        if rc == 0:
            logger.info(f"{module.capitalize()} fetch complete ✓")
        else:
            logger.warning(
                f"{module} fetch exited with code {rc}.\n"
                f"  → Run manually: cd {module} && python main.py --mode fetch"
            )

    logger.info("Fetch-all summary:")
    for module in MODULES:
        status = "[✓] data found" if _data_exists(module) else "[✗] MISSING"
        logger.info(f"  {module:<14} {status}")


def step_portfolio():
    import pandas as pd
    from portfolio_manager import MultiAssetPortfolio

    logger.info("Computing multi-asset portfolio weights ...")
    prices = {}

    for module, info in MODULES.items():
        raw_dir    = ROOT / module / "data" / "raw"
        found_path = next(
            (raw_dir / f for f in info["alt_files"] if (raw_dir / f).exists()),
            None,
        )
        if found_path:
            df = pd.read_parquet(found_path, engine="pyarrow")
            prices[module] = df["close"].tail(720)
            logger.info(
                f"  {module:<14}: {len(prices[module])} bars  "
                f"({df.index[-1].date()})  ← {found_path.name}"
            )
        else:
            logger.warning(
                f"  {module:<14}: no data — run --mode fetch-all"
            )

    if not prices:
        logger.error("No data available. Run: python main.py --mode fetch-all")
        return None, None

    vols      = MultiAssetPortfolio.compute_volatilities(prices, window=168)
    total_cap = 30_000.0
    portfolio = MultiAssetPortfolio(
        assets=list(prices.keys()), total_capital=total_cap,
        scheme="inverse_vol", max_single=0.60, min_single=0.05,
    )
    weights     = portfolio.compute_weights(volatilities=vols)
    allocations = portfolio.allocate(weights)

    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  MULTI-ASSET PORTFOLIO  (${total_cap:,.0f} total capital)")
    print(f"  Scheme: inverse volatility  (lower vol → more capital)")
    print(f"  Assets: {', '.join(prices.keys())}")
    print(sep)
    print(f"  {'Asset':<16} {'Ann. Vol':>10} {'Weight':>10} {'Capital':>14}  Bar")
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*14}  {'─'*22}")
    for mod in prices:
        v, w, a = vols.get(mod, 0), weights.get(mod, 0), allocations.get(mod, 0)
        bar     = "█" * int(w * 30)
        print(f"  {mod:<16} {v*100:>9.2f}%  {w*100:>9.1f}%  ${a:>12,.2f}  {bar}")
    print(f"\n  Expected: BTC > Gold/FX volatility → crypto gets smallest allocation.")
    print(f"{sep}\n")
    return weights, allocations


def step_compare_all():
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  CROSS-ASSET PERFORMANCE COMPARISON  —  Test Sets")
    print(sep)

    metrics_found = {}
    for module in MODULES:
        log_dir = ROOT / module / "logs" / "runs"
        if log_dir.exists():
            runs = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if runs:
                with open(runs[-1]) as f:
                    metrics_found[module] = json.load(f).get("metrics", {})

    if not metrics_found:
        print(f"\n  No results yet. Run each module pipeline first:")
        for m in MODULES:
            print(f"    cd {m:<14} && python main.py --mode compare")
        print(f"\n  Then: python main.py --mode compare-all")
        print(f"{sep}\n")
        return

    rows = [
        ("Sharpe Ratio",   "test_sharpe",           "{:>+.3f}"),
        ("Total Return %", "test_total_return_pct",  "{:>+.2f}%"),
        ("Max Drawdown %", "test_max_drawdown_pct",  "{:>.2f}%"),
        ("Win Rate %",     "test_win_rate_pct",      "{:>.1f}%"),
        ("Profit Factor",  "test_profit_factor",     "{:>.3f}"),
        ("Total Trades",   "test_total_trades",      "{:>}"),
        ("Buy & Hold %",   "test_buy_hold_pct",      "{:>+.2f}%"),
    ]
    names = list(metrics_found.keys())
    cw    = 18
    print(f"\n  {'Metric':<26}" + "".join(f"{n:>{cw}}" for n in names))
    print(f"  {'─'*26}" + "─"*(cw*len(names)))
    for label, key, fmt in rows:
        row = f"  {label:<26}"
        for mod in names:
            val = metrics_found[mod].get(key, None)
            try:    row += f"{fmt.format(float(val)):>{cw}}"
            except: row += f"{'N/A':>{cw}}"
        print(row)
    print(f"\n  Targets: Sharpe > 1.5  |  Profit Factor > 1.5  |  Win Rate > 50%")
    print(f"{sep}\n")


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="Trading-Bot Multi-Asset Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["status","fetch-all","portfolio","compare-all"],
        default="status",
    )
    args = parser.parse_args()
    logger.info(f"Trading-Bot  |  mode={args.mode}")

    {
        "status":      step_status,
        "fetch-all":   step_fetch_all,
        "portfolio":   step_portfolio,
        "compare-all": step_compare_all,
    }[args.mode]()


if __name__ == "__main__":
    main()