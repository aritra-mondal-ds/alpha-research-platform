# commodities/main.py
"""
Commodities Trading Bot — XAU/USD (Gold via GLD ETF)
Usage:
    python main.py --mode fetch       # download GLD 1h + 1d
    python main.py --mode load        # display raw data
    python main.py --mode features    # build feature matrix
    python main.py --mode strategy    # rule-based backtest
    python main.py --mode train-v2    # XGBoost v2 (anti-overfit)
    python main.py --mode compare     # compare all models + save JSON log
    python main.py --mode full        # end-to-end

Commodities-specific tuning:
    target threshold    : ±0.50%  (GLD typical 12h move: 0.3-0.8%)
    confidence threshold: 0.00    (pure argmax — same reason as forex:
                                   limited data means low max probabilities)
"""

import sys
import json
import argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = MODULE_ROOT.parent
CRYPTO_ROOT = REPO_ROOT / "crypto"
SHARED_DIR  = REPO_ROOT / "shared"

for p in [str(CRYPTO_ROOT), str(SHARED_DIR)]:
    if p not in sys.path:
        sys.path.append(p)

sys.path.insert(0, str(MODULE_ROOT))

import config.settings as cfg
from fetcher import CommoditiesFetcher

from loguru import logger
import pandas as pd

from src.data.loader              import DataLoader, display_dataframe
from src.data.splitter            import temporal_split
from src.features.pipeline        import FeaturePipeline
from src.strategies.ema_rsi       import EMACrossoverRSI
from src.strategies.multi_indicator  import MultiIndicatorConfluence
from src.strategies.position_manager import signals_to_positions
from src.backtesting.simple_backtest  import SimpleBacktester


# ── Commodities-specific constants ────────────────────────────────────────────
GOLD_TARGET_THRESHOLD    = 0.005   # ±0.50%
GOLD_CONFIDENCE_THRESHOLD = 0.0    # pure argmax


def _setup_logger():
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(cfg.LOG_DIR / "main.log"), rotation="20 MB",
        retention="14 days", level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}",
    )


def _safe_symbol(sym: str) -> str:
    return sym.replace("/", "_").replace("=", "").replace("^", "")


def _load_feature_matrix(force=False) -> pd.DataFrame:
    path = (
        cfg.DATA_DIR / "processed" /
        f"{_safe_symbol(cfg.SYMBOL)}_{cfg.PRIMARY_TF}_features.parquet"
    )
    if path.exists() and not force:
        df = pd.read_parquet(path, engine="pyarrow")
        if len(df) == 0:
            logger.warning("Cached feature matrix is empty — recomputing.")
            force = True
        else:
            logger.info(f"Loading feature matrix → {path.name}")
            return df

    return FeaturePipeline().run(
        symbol=cfg.SYMBOL, primary_tf=cfg.PRIMARY_TF,
        higher_tf=cfg.HIGHER_TF, horizon=12,
        threshold=GOLD_TARGET_THRESHOLD, save=True,
    )


def _get_feat_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in {"open","high","low","close","volume","target"}]


def _save_results_json(results: dict, split):
    """Save best model metrics to logs/runs/ for compare-all."""
    log_dir = cfg.LOG_DIR / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)

    best_name = max(results, key=lambda k: results[k].metrics.get("sharpe", -99))
    best      = results[best_name]

    payload = {
        "asset":    cfg.DISPLAY,
        "strategy": best_name,
        "metrics": {
            f"test_{k}": v
            for k, v in best.metrics.items()
            if isinstance(v, (int, float))
        },
    }

    from datetime import datetime
    fname = log_dir / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved → {fname.name}  (best: {best_name})")


# ── Steps ─────────────────────────────────────────────────────────────────────
def step_fetch(force=False):
    logger.info(f"Fetching {cfg.DISPLAY} ...")
    fetcher = CommoditiesFetcher(cfg.DATA_DIR)
    fetcher.fetch(cfg.SYMBOL, cfg.PRIMARY_TF, cfg.START_DATE, force=force)
    fetcher.fetch(cfg.SYMBOL, cfg.HIGHER_TF,  cfg.START_DATE, force=force)


def step_load():
    loader = DataLoader()
    display_dataframe(loader.load(cfg.SYMBOL, cfg.PRIMARY_TF),
                      f"{cfg.DISPLAY} | {cfg.PRIMARY_TF}")
    display_dataframe(loader.load(cfg.SYMBOL, cfg.HIGHER_TF),
                      f"{cfg.DISPLAY} | {cfg.HIGHER_TF}")


def step_features(force=False):
    df = _load_feature_matrix(force)
    fc = _get_feat_cols(df)
    logger.info(
        f"Feature matrix: {df.shape} | {len(fc)} features | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    if "target" in df.columns:
        counts = df["target"].value_counts().sort_index()
        total  = len(df)
        labels = {-1:"SHORT", 0:"HOLD", 1:"LONG"}
        for val, cnt in counts.items():
            bar = "█" * int(cnt/total*30)
            logger.info(
                f"  {labels.get(val,val):>6}: {cnt:>6,} ({cnt/total*100:>5.1f}%)  {bar}"
            )
    return df


def step_strategy(force=False):
    logger.info(f"Rule-Based Strategies — {cfg.DISPLAY}")
    df    = _load_feature_matrix(force)
    split = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    bt    = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    results = {}
    for name, strat in [
        ("EMA+RSI",    EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True)),
        ("Confluence", MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)),
    ]:
        results[name] = bt.run(
            split.test, signals_to_positions(strat.generate_signals(split.test)), name
        )
    _display_comparison(results)
    return results


def step_train_v2(force=False, n_trials=50):
    from src.models.xgboost_v2 import XGBoostTradingModelV2

    logger.info(f"XGBoost v2 Training — {cfg.DISPLAY}")
    logger.info(
        f"Gold tuning: target=±{GOLD_TARGET_THRESHOLD*100:.2f}%  "
        f"confidence_threshold={GOLD_CONFIDENCE_THRESHOLD} (pure argmax)"
    )

    df    = _load_feature_matrix(force)
    split = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    fc    = _get_feat_cols(df)

    model           = XGBoostTradingModelV2()
    model.threshold = GOLD_CONFIDENCE_THRESHOLD

    model.train(
        split.train[fc], split.train["target"].astype(int),
        split.val[fc],   split.val["target"].astype(int),
        n_trials=n_trials,
    )
    model.evaluate(split.test[fc], split.test["target"].astype(int), "Test")
    model.feature_importance(top_n=15)

    X_te  = split.test[model.feature_cols]
    sigs  = pd.Series(
        model.predict(X_te, threshold=GOLD_CONFIDENCE_THRESHOLD),
        index=split.test.index,
    )

    n_long  = (sigs == 1).sum()
    n_short = (sigs == -1).sum()
    n_hold  = (sigs == 0).sum()
    logger.info(
        f"Signal breakdown — LONG: {n_long} | HOLD: {n_hold} | SHORT: {n_short}"
    )

    bt  = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    res = bt.run(split.test, signals_to_positions(sigs), "XGBoost v2")
    _display_comparison({"XGBoost v2": res})
    return model, res


def step_compare(force=False):
    from src.models.xgboost_model import XGBoostTradingModel
    from src.models.xgboost_v2   import XGBoostTradingModelV2

    logger.info(f"Model Comparison — {cfg.DISPLAY}")
    df    = _load_feature_matrix(force)
    split = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    fc    = _get_feat_cols(df)
    bt    = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    results = {}

    for name, loader_fn in [
        ("XGBoost v1", lambda: XGBoostTradingModel().load()),
        ("XGBoost v2", lambda: XGBoostTradingModelV2().load()),
    ]:
        try:
            m    = loader_fn()
            x_in = split.test[m.feature_cols] if (
                hasattr(m,"feature_cols") and m.feature_cols
                and set(m.feature_cols).issubset(split.test.columns)
            ) else split.test[fc]
            sigs = pd.Series(
                m.predict(x_in, threshold=GOLD_CONFIDENCE_THRESHOLD),
                index=split.test.index,
            )
            n_l = (sigs == 1).sum()
            n_h = (sigs == 0).sum()
            n_s = (sigs == -1).sum()
            logger.info(
                f"{name} signals — LONG: {n_l} | HOLD: {n_h} | SHORT: {n_s}"
            )
            results[name] = bt.run(split.test, signals_to_positions(sigs), name)
        except FileNotFoundError:
            logger.warning(f"{name} not found — run --mode train-v2")

    for name, strat in [
        ("EMA+RSI",    EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True)),
        ("Confluence", MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)),
    ]:
        results[name] = bt.run(
            split.test, signals_to_positions(strat.generate_signals(split.test)), name
        )

    _display_comparison(results)

    # Save to logs/runs/ for compare-all
    _save_results_json(results, split)

    return results


def _display_comparison(results: dict):
    if not results:
        return
    metrics = [
        ("Total Return %",   "total_return_pct",  "{:>+.2f}%"),
        ("Sharpe Ratio",     "sharpe",             "{:>.3f}"),
        ("Sortino Ratio",    "sortino",            "{:>.3f}"),
        ("Max Drawdown %",   "max_drawdown_pct",   "{:>.2f}%"),
        ("Win Rate %",       "win_rate_pct",        "{:>.1f}%"),
        ("Profit Factor",    "profit_factor",       "{:>.3f}"),
        ("Total Trades",     "total_trades",        "{:>}"),
        ("Buy & Hold %",     "buy_hold_pct",        "{:>+.2f}%"),
    ]
    names = list(results.keys())
    cw    = 18
    sep   = "═" * (28 + cw * len(names))
    print(f"\n{sep}")
    print(f"  {cfg.DISPLAY} — STRATEGY COMPARISON (Test Set)")
    print(f"  Target: ±{GOLD_TARGET_THRESHOLD*100:.2f}%  |  "
          f"Confidence threshold: {GOLD_CONFIDENCE_THRESHOLD}")
    print(sep)
    print(f"  {'Metric':<26}" + "".join(f"{n:>{cw}}" for n in names))
    print(f"  {'─'*26}" + "─"*(cw*len(names)))
    for label, key, fmt in metrics:
        row = f"  {label:<26}"
        for name in names:
            val = results[name].metrics.get(key, 0)
            try:    row += f"{fmt.format(val):>{cw}}"
            except: row += f"{str(val):>{cw}}"
        print(row)
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description=f"Commodities Bot — {cfg.DISPLAY}")
    parser.add_argument(
        "--mode",
        choices=["fetch","load","features","strategy","train","train-v2","compare","full"],
        default="full",
    )
    parser.add_argument("--force",     action="store_true", default=False)
    parser.add_argument("--no-optuna", action="store_true", default=False)
    parser.add_argument("--trials",    type=int, default=50)
    args = parser.parse_args()

    logger.info(f"Commodities Bot  |  mode={args.mode}  |  symbol={cfg.DISPLAY}")

    if args.mode == "fetch":
        step_fetch(args.force)
    elif args.mode == "load":
        step_load()
    elif args.mode == "features":
        step_features(args.force)
    elif args.mode == "strategy":
        step_strategy(args.force)
    elif args.mode in ("train", "train-v2"):
        step_train_v2(args.force, args.trials)
    elif args.mode == "compare":
        step_compare(args.force)
    elif args.mode == "full":
        step_fetch(args.force)
        step_features(args.force)
        step_strategy(args.force)
        step_train_v2(args.force, args.trials)
        step_compare(args.force)


if __name__ == "__main__":
    main()