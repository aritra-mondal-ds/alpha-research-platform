# crypto/main.py
"""
CryptoTradingBot — Entry Point
==============================
Usage:
    python main.py --mode fetch       # GPU check + download BTC/USDT 1h + 1d
    python main.py --mode load        # GPU check + load & display raw data
    python main.py --mode features    # GPU check + build feature matrix + display
    python main.py --mode strategy    # GPU check + run rule-based strategies + backtest
    python main.py --mode full        # fetch → features → strategy (default)

Flags:
    --force   Force re-download / re-compute even if files already exist
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger
import config.settings as cfg
from src.utils.gpu_check              import verify_gpu
from src.data.fetcher                 import OHLCVFetcher
from src.data.loader                  import DataLoader, display_dataframe
from src.data.splitter                import temporal_split
from src.features.pipeline            import FeaturePipeline
from src.strategies.ema_rsi           import EMACrossoverRSI
from src.strategies.multi_indicator   import MultiIndicatorConfluence
from src.strategies.position_manager  import signals_to_positions
from src.backtesting.simple_backtest  import SimpleBacktester


# ── Logger ────────────────────────────────────────────────────────────────────
def _setup_logger():
    log_path = cfg.LOG_DIR / "main.log"
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(log_path),
        rotation="20 MB",
        retention="14 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    )


# ── Steps ─────────────────────────────────────────────────────────────────────
def step_gpu_check():
    logger.info("STEP 1 — GPU Check")
    verify_gpu(verbose=True)


def step_fetch(force=False):
    logger.info("STEP 2 — Data Fetch")
    fetcher = OHLCVFetcher(exchange_id=cfg.EXCHANGE)
    fetcher.fetch(cfg.SYMBOL, cfg.PRIMARY_TF, cfg.START_DATE, force=force)
    fetcher.fetch(cfg.SYMBOL, cfg.HIGHER_TF,  cfg.START_DATE, force=force)


def step_load_and_display():
    logger.info("STEP 3 — Load & Display Raw Data")
    loader = DataLoader()
    df_primary = loader.load(cfg.SYMBOL, cfg.PRIMARY_TF)
    display_dataframe(df_primary, f"{cfg.SYMBOL} | {cfg.PRIMARY_TF} | Primary")
    df_daily   = loader.load(cfg.SYMBOL, cfg.HIGHER_TF)
    display_dataframe(df_daily,   f"{cfg.SYMBOL} | {cfg.HIGHER_TF} | Macro")
    return df_primary, df_daily


def step_features(force=False):
    logger.info("STEP 3 — Feature Engineering")
    import pandas as pd

    processed_path = (
        cfg.DATA_DIR / "processed" /
        f"{cfg.SYMBOL.replace('/', '_')}_{cfg.PRIMARY_TF}_features.parquet"
    )

    if processed_path.exists() and not force:
        logger.info(f"Loading existing feature matrix → {processed_path.name}")
        df = pd.read_parquet(processed_path, engine="pyarrow")
    else:
        pipeline = FeaturePipeline()
        df = pipeline.run(
            symbol=cfg.SYMBOL, primary_tf=cfg.PRIMARY_TF,
            higher_tf=cfg.HIGHER_TF, horizon=12, threshold=0.015, save=True,
        )

    _display_features(df)
    return df


def step_strategy(force=False):
    import pandas as pd

    logger.info("STEP 4 — Rule-Based Strategy + Backtest")

    # ── Load feature matrix ───────────────────────────────────────────────────
    processed_path = (
        cfg.DATA_DIR / "processed" /
        f"{cfg.SYMBOL.replace('/', '_')}_{cfg.PRIMARY_TF}_features.parquet"
    )
    if not processed_path.exists():
        logger.info("Feature matrix not found — building it first ...")
        step_features(force=force)

    df = pd.read_parquet(processed_path, engine="pyarrow")

    # ── Temporal split ────────────────────────────────────────────────────────
    logger.info("Splitting data — 70% train | 15% val | 15% test ...")
    split = temporal_split(df, train_ratio=0.70, val_ratio=0.15)

    # Evaluate on TEST set only — never tune on test
    df_test = split.test

    backtester = SimpleBacktester(
        initial_capital = cfg.INITIAL_CAPITAL,
        commission      = cfg.COMMISSION,
        slippage        = cfg.SLIPPAGE,
    )

    results = {}

    # ── Strategy 1: EMA Crossover + RSI ──────────────────────────────────────
    logger.info("Running Strategy 1: EMA Crossover + RSI ...")
    strat1     = EMACrossoverRSI(
        rsi_long_threshold = 50.0,
        rsi_exit_ob        = 75.0,
        use_ema50_filter   = True,
        long_only          = True,
    )
    signals1   = strat1.generate_signals(df_test)
    positions1 = signals_to_positions(signals1, mode="long_only")
    results["EMA+RSI"] = backtester.run(df_test, positions1, "EMA Crossover + RSI")

    # ── Strategy 2: Multi-Indicator Confluence ────────────────────────────────
    logger.info("Running Strategy 2: Multi-Indicator Confluence ...")
    strat2     = MultiIndicatorConfluence(
        rsi_low      = 45.0,
        rsi_high     = 70.0,
        volume_surge = 1.2,
        long_only    = True,
    )
    signals2   = strat2.generate_signals(df_test)
    positions2 = signals_to_positions(signals2, mode="long_only")
    results["Confluence"] = backtester.run(
        df_test, positions2, "Multi-Indicator Confluence"
    )

    # ── Side-by-side comparison ───────────────────────────────────────────────
    _display_strategy_comparison(results)

    return results


# ── Display helpers ───────────────────────────────────────────────────────────
def _display_features(df):
    import pandas as pd
    sep       = "═" * 65
    target    = "target"
    feat_cols = [c for c in df.columns
                 if c not in {"open","high","low","close","volume",target}]

    print(f"\n{sep}")
    print(f"  FEATURE MATRIX SUMMARY")
    print(sep)
    print(f"  Total rows      : {len(df):,}")
    print(f"  Feature columns : {len(feat_cols)}")
    print(f"  Date from       : {df.index[0]}")
    print(f"  Date to         : {df.index[-1]}")

    groups = {
        "Trend":         [c for c in feat_cols if any(k in c for k in ["ema","macd","adx","cross"])],
        "Momentum":      [c for c in feat_cols if any(k in c for k in ["rsi","stoch","williams","roc"])],
        "Volatility":    [c for c in feat_cols if any(k in c for k in ["bb_","atr","kc_","vol"])],
        "Volume":        [c for c in feat_cols if any(k in c for k in ["volume","obv","vwap","mfi"])],
        "Price-Derived": [c for c in feat_cols if any(k in c for k in ["return","spread","range","log"])],
        "Macro (1d)":    [c for c in feat_cols if c.startswith("d1_")],
        "Patterns":      [c for c in feat_cols if c.startswith("pat_")],
    }
    print(f"\n  [ Feature Groups ]\n")
    for group, cols in groups.items():
        print(f"  {group:<18}: {len(cols):>3} cols")

    if target in df.columns:
        print(f"\n  [ Target Distribution ]\n")
        counts = df[target].value_counts().sort_index()
        total  = len(df)
        labels = {-1:"SHORT", 0:"HOLD", 1:"LONG"}
        for val, cnt in counts.items():
            bar = "█" * int(cnt/total*40)
            print(f"  {labels.get(val,val):>6} ({int(val):>2}): "
                  f"{cnt:>6,} ({cnt/total*100:>5.1f}%)  {bar}")

    nan_counts = df[feat_cols].isnull().sum()
    print(f"\n  [ NaN Check ] : "
          f"{'No NaN values ✓' if not nan_counts.any() else 'WARNING — NaNs detected'}")
    print(f"{sep}\n")


def _display_strategy_comparison(results: dict):
    sep = "═" * 75
    metrics = [
        ("Total Return %",   "total_return_pct",  "{:>+.2f}%"),
        ("Ann. Return %",    "ann_return_pct",     "{:>+.2f}%"),
        ("Buy & Hold %",     "buy_hold_pct",       "{:>+.2f}%"),
        ("Sharpe Ratio",     "sharpe",             "{:>.3f}"),
        ("Sortino Ratio",    "sortino",            "{:>.3f}"),
        ("Calmar Ratio",     "calmar",             "{:>.3f}"),
        ("Max Drawdown %",   "max_drawdown_pct",   "{:>.2f}%"),
        ("Total Trades",     "total_trades",       "{:>}"),
        ("Win Rate %",       "win_rate_pct",       "{:>.1f}%"),
        ("Profit Factor",    "profit_factor",      "{:>.3f}"),
        ("Avg Win %",        "avg_win_pct",        "{:>+.2f}%"),
        ("Avg Loss %",       "avg_loss_pct",       "{:>+.2f}%"),
        ("Expectancy %",     "expectancy_pct",     "{:>+.4f}%"),
        ("Time in Market %", "time_in_market_pct", "{:>.1f}%"),
    ]

    names     = list(results.keys())
    col_width = 22

    print(f"\n{sep}")
    print(f"  STRATEGY COMPARISON  —  Test Set Only")
    print(f"  (Sharpe > 1.5 and Profit Factor > 1.5 are baseline targets)")
    print(sep)
    header = f"  {'Metric':<28}" + "".join(f"{n:>{col_width}}" for n in names)
    print(header)
    print(f"  {'─'*28}" + "─" * (col_width * len(names)))

    for label, key, fmt in metrics:
        row = f"  {label:<28}"
        for name in names:
            val = results[name].metrics.get(key, 0)
            try:
                formatted = fmt.format(val)
            except Exception:
                formatted = str(val)
            row += f"{formatted:>{col_width}}"
        print(row)

    print(f"{sep}\n")


# ── Modes ─────────────────────────────────────────────────────────────────────
def run_fetch(force):
    step_gpu_check(); step_fetch(force)
    logger.info("Fetch complete.")

def run_load(force):
    step_gpu_check(); step_load_and_display()

def run_features(force):
    step_gpu_check(); step_features(force)

def run_strategy(force):
    step_gpu_check(); step_strategy(force)

def run_full(force):
    step_gpu_check()
    step_fetch(force)
    step_features(force)
    step_strategy(force)
    logger.info("Full pipeline complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="CryptoTradingBot")
    parser.add_argument(
        "--mode",
        choices=["fetch","load","features","strategy","full"],
        default="full",
    )
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    logger.info(
        f"CryptoTradingBot  |  mode={args.mode}  |  "
        f"symbol={cfg.SYMBOL}  |  force={args.force}"
    )
    dispatch = {
        "fetch":    run_fetch,
        "load":     run_load,
        "features": run_features,
        "strategy": run_strategy,
        "full":     run_full,
    }
    dispatch[args.mode](args.force)


if __name__ == "__main__":
    main()