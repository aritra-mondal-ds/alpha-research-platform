# crypto/main.py
"""
CryptoTradingBot — Entry Point
==============================
Usage:
    python main.py --mode fetch
    python main.py --mode load
    python main.py --mode features
    python main.py --mode strategy
    python main.py --mode train
    python main.py --mode train-v2
    python main.py --mode train-lstm
    python main.py --mode train-rl
    python main.py --mode optimize
    python main.py --mode compare
    python main.py --mode wfv
    python main.py --mode evaluate
    python main.py --mode report
    python main.py --mode full

Flags:
    --force       Re-compute even if files exist
    --no-optuna   Skip Optuna (faster)
    --trials N    Optuna trials (default 20)
    --timesteps N PPO timesteps (default 500000)

Lazy import strategy:
    ALL heavy libraries (PyTorch, XGBoost, SB3, WalkForwardValidator)
    are imported INSIDE their functions only.
    This prevents Windows CUDA DLL crash (0xC0000005) on startup
    for lightweight modes like fetch, features, strategy.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger
import pandas as pd
import numpy as np
import config.settings as cfg

# ── Lightweight module-level imports only ────────────────────────────────────
from src.data.fetcher                 import OHLCVFetcher
from src.data.loader                  import DataLoader, display_dataframe
from src.data.splitter                import temporal_split
from src.features.pipeline            import FeaturePipeline
from src.strategies.ema_rsi           import EMACrossoverRSI
from src.strategies.multi_indicator   import MultiIndicatorConfluence
from src.strategies.position_manager  import signals_to_positions
from src.backtesting.simple_backtest  import SimpleBacktester
from src.models.feature_selector      import get_stationary_features, print_feature_analysis

# ALL of these are now LAZY (inside functions):
# WalkForwardValidator, check_overfitting, monte_carlo_simulation,
# print_wfv_comparison, XGBoostTradingModel, XGBoostTradingModelV2,
# LSTMTradingModel, PPOTradingAgent, generate_report, ExperimentTracker


# ── Logger ────────────────────────────────────────────────────────────────────
def _setup_logger():
    log_path = cfg.LOG_DIR / "main.log"
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(log_path), rotation="20 MB", retention="14 days", level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}",
    )


# ── Shared helpers ────────────────────────────────────────────────────────────
def step_gpu_check():
    # Lazy — torch/CUDA must NOT be imported at module level on Windows.
    # Importing torch before argparse runs causes silent 0xC0000005 crash.
    from src.utils.gpu_check import verify_gpu
    logger.info("STEP 1 — GPU Check")
    verify_gpu(verbose=True)


def step_fetch(force=False):
    logger.info("STEP — Data Fetch")
    f = OHLCVFetcher(exchange_id=cfg.EXCHANGE)
    f.fetch(cfg.SYMBOL, cfg.PRIMARY_TF, cfg.START_DATE, force=force)
    f.fetch(cfg.SYMBOL, cfg.HIGHER_TF,  cfg.START_DATE, force=force)


def _load_feature_matrix(force=False) -> pd.DataFrame:
    path = (
        cfg.DATA_DIR / "processed" /
        f"{cfg.SYMBOL.replace('/', '_')}_{cfg.PRIMARY_TF}_features.parquet"
    )
    if path.exists() and not force:
        logger.info(f"Loading feature matrix → {path.name}")
        return pd.read_parquet(path, engine="pyarrow")
    return FeaturePipeline().run(
        symbol=cfg.SYMBOL, primary_tf=cfg.PRIMARY_TF,
        higher_tf=cfg.HIGHER_TF, horizon=12, threshold=0.015, save=True,
    )


def _get_feat_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in {"open","high","low","close","volume","target"}]


def _get_split_and_arrays(force=False):
    df        = _load_feature_matrix(force)
    split     = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    feat_cols = _get_feat_cols(df)
    return (
        split, feat_cols,
        split.train[feat_cols], split.train["target"].astype(int),
        split.val[feat_cols],   split.val["target"].astype(int),
        split.test[feat_cols],  split.test["target"].astype(int),
    )


def _save_results_json(results: dict):
    """Save best model metrics to logs/runs/ so compare-all can read them."""
    log_dir = cfg.LOG_DIR / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(results, key=lambda k: results[k].metrics.get("sharpe", -99))
    payload   = {
        "asset":    cfg.SYMBOL,
        "strategy": best_name,
        "metrics": {
            f"test_{k}": v
            for k, v in results[best_name].metrics.items()
            if isinstance(v, (int, float))
        },
    }
    fname = log_dir / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved → {fname.name}  (best: {best_name})")


# ── Pipeline steps ────────────────────────────────────────────────────────────
def step_load_and_display():
    loader = DataLoader()
    display_dataframe(loader.load(cfg.SYMBOL, cfg.PRIMARY_TF),
                      f"{cfg.SYMBOL} | {cfg.PRIMARY_TF} | Primary")
    display_dataframe(loader.load(cfg.SYMBOL, cfg.HIGHER_TF),
                      f"{cfg.SYMBOL} | {cfg.HIGHER_TF} | Macro")


def step_features(force=False):
    df = _load_feature_matrix(force)
    _display_features(df)
    return df


def step_strategy(force=False):
    logger.info("STEP — Rule-Based Strategies")
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
    _display_strategy_comparison(results)
    return results


def step_train(force=False, n_trials=50, use_optuna=True):
    from src.models.xgboost_model import XGBoostTradingModel, LABEL_MAP
    import xgboost as xgb

    logger.info("STEP — XGBoost v1 Training")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)

    model = XGBoostTradingModel()
    if not use_optuna:
        model.feature_cols = feat_cols
        y_tr_enc = y_tr.map(LABEL_MAP).values.astype(int)
        y_vl_enc = y_vl.map(LABEL_MAP).values.astype(int)
        p = {
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", "tree_method": "hist",
            "device": "cuda", "seed": cfg.RANDOM_SEED, "verbosity": 0,
        }
        model.best_params = p
        model.model = xgb.XGBClassifier(**p, early_stopping_rounds=30)
        w = model._compute_weights(y_tr_enc)
        model.model.fit(X_tr, y_tr_enc, sample_weight=w,
                        eval_set=[(X_vl, y_vl_enc)], verbose=100)
        model.save()
    else:
        model.train(X_tr, y_tr, X_vl, y_vl, n_trials=n_trials)

    model.evaluate(X_te, y_te, "Test")
    model.feature_importance(top_n=20)
    bt  = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    res = bt.run(split.test,
                 signals_to_positions(pd.Series(model.predict(X_te), index=split.test.index)),
                 "XGBoost v1")
    _display_strategy_comparison({"XGBoost v1": res})
    return model, res


def step_train_v2(force=False, n_trials=50, use_optuna=True):
    from src.models.xgboost_v2 import XGBoostTradingModelV2

    logger.info("STEP — XGBoost v2 Training (Anti-Overfit)")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)

    model_v2 = XGBoostTradingModelV2()
    if not use_optuna:
        import xgboost as xgb
        from src.models.xgboost_v2 import LABEL_MAP
        stationary = get_stationary_features(pd.concat([X_tr, X_vl]))
        model_v2.feature_cols = stationary
        X_tr_s = X_tr[stationary]; X_vl_s = X_vl[stationary]
        y_tr_enc = y_tr.map(LABEL_MAP).values.astype(int)
        y_vl_enc = y_vl.map(LABEL_MAP).values.astype(int)
        p = {
            "n_estimators": 300, "max_depth": 5, "max_leaves": 32,
            "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 2.0, "reg_lambda": 5.0, "min_child_weight": 20,
            "gamma": 3.0, "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", "tree_method": "hist",
            "device": "cuda", "seed": cfg.RANDOM_SEED, "verbosity": 0,
        }
        model_v2.best_params = p
        model_v2.model = xgb.XGBClassifier(**p, early_stopping_rounds=20)
        w = model_v2._compute_weights(y_tr_enc)
        model_v2.model.fit(X_tr_s, y_tr_enc, sample_weight=w,
                           eval_set=[(X_vl_s, y_vl_enc)], verbose=50)
        model_v2._print_overfit_check(X_tr_s, y_tr_enc, X_vl_s, y_vl_enc)
        model_v2.save()
    else:
        model_v2.train(X_tr, y_tr, X_vl, y_vl, n_trials=n_trials)

    model_v2.evaluate(X_te, y_te, "Test")
    model_v2.feature_importance(top_n=20)
    X_te_s = X_te[model_v2.feature_cols]
    bt     = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    res    = bt.run(split.test,
                    signals_to_positions(pd.Series(model_v2.predict(X_te_s), index=split.test.index)),
                    "XGBoost v2")
    _display_strategy_comparison({"XGBoost v2": res})
    return model_v2, res


def step_train_lstm(force=False):
    from src.models.lstm_model import LSTMTradingModel

    logger.info("STEP — LSTM Training")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    lstm = LSTMTradingModel(
        seq_len=60, hidden_size=128, num_layers=2,
        dropout=0.3, batch_size=512, max_epochs=60, lr=1e-3, patience=10,
    )
    lstm.train(X_tr, y_tr, X_vl, y_vl)
    lstm.evaluate(X_te, y_te, "Test")
    bt      = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    sigs    = pd.Series(lstm.predict(X_te), index=split.test.index)
    res     = bt.run(split.test, signals_to_positions(sigs), "LSTM")
    _display_strategy_comparison({"LSTM": res})
    return lstm, res


def step_train_rl(force=False, total_timesteps=500_000):
    from src.models.rl.ppo_trainer import PPOTradingAgent

    logger.info("STEP — PPO Reinforcement Learning")
    df        = _load_feature_matrix(force)
    split     = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    feat_cols = _get_feat_cols(df)
    agent     = PPOTradingAgent()
    agent.train(
        df_train=split.train.copy(), df_val=split.val.copy(),
        feature_cols=feat_cols, total_timesteps=total_timesteps,
        n_envs=4, window=30,
    )
    return agent, agent.evaluate(split.test.copy(), feat_cols, window=30)


def step_optimize(force=False):
    from src.models.xgboost_model import XGBoostTradingModel
    from src.models.xgboost_v2   import XGBoostTradingModelV2
    from src.backtesting.metrics  import check_overfitting, monte_carlo_simulation

    logger.info("STEP — Optimization Analysis")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    df = _load_feature_matrix(force)
    print_feature_analysis(df, feat_cols)

    bt = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    results_tr, results_te = {}, {}
    for name, loader_fn, xtr, xte in [
        ("v1", lambda: XGBoostTradingModel().load(),  X_tr, X_te),
        ("v2", lambda: XGBoostTradingModelV2().load(), None, None),
    ]:
        try:
            m    = loader_fn()
            _xtr = X_tr[m.feature_cols] if hasattr(m,"feature_cols") and m.feature_cols and set(m.feature_cols).issubset(X_tr.columns) else xtr
            _xte = X_te[m.feature_cols] if hasattr(m,"feature_cols") and m.feature_cols and set(m.feature_cols).issubset(X_te.columns) else xte
            if _xtr is not None:
                results_tr[name] = bt.run(split.train,
                    signals_to_positions(pd.Series(m.predict(_xtr), index=split.train.index)),
                    f"{name} Train")
            if _xte is not None:
                results_te[name] = bt.run(split.test,
                    signals_to_positions(pd.Series(m.predict(_xte), index=split.test.index)),
                    f"{name} Test")
            if name in results_tr and name in results_te:
                check_overfitting(results_tr[name].metrics, results_te[name].metrics)
        except FileNotFoundError:
            logger.warning(f"XGBoost {name} not found")

    if results_tr: _display_strategy_comparison(results_tr)
    if results_te: _display_strategy_comparison(results_te)

    best = max(results_te, key=lambda k: results_te[k].metrics.get("sharpe",-99),
               ) if results_te else None
    if best and len(results_te[best].trades) >= 5:
        monte_carlo_simulation(results_te[best].trades["pnl_pct"],
                               cfg.INITIAL_CAPITAL, 1000)


def step_wfv(force=False, n_trials=20):
    # ── Lazy imports — ALL heavy imports inside this function ─────────────────
    from src.backtesting.walk_forward import WalkForwardValidator
    from src.backtesting.metrics      import print_wfv_comparison

    logger.info("STEP — Walk-Forward Validation")
    df        = _load_feature_matrix(force)
    feat_cols = _get_feat_cols(df)

    wfv = WalkForwardValidator(
        train_window_h=17_520, test_window_h=4_380,
        step_h=4_380, mode="expanding",
    )
    reports = {}

    logger.info("Running XGBoost WFV (re-trains each fold) ...")
    try:
        reports["XGBoost"] = wfv.run_xgboost(df, feat_cols, n_trials=n_trials)
        logger.info("XGBoost WFV complete ✓")
    except Exception as e:
        logger.error(f"XGBoost WFV failed: {e}")
        import traceback; traceback.print_exc()

    logger.info("Running Confluence WFV ...")
    try:
        reports["Confluence"] = wfv.run_rule_based(
            df, MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True),
            name="Confluence",
        )
        logger.info("Confluence WFV complete ✓")
    except Exception as e:
        logger.error(f"Confluence WFV failed: {e}")

    logger.info("Running EMA+RSI WFV ...")
    try:
        reports["EMA+RSI"] = wfv.run_rule_based(
            df, EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True),
            name="EMA+RSI",
        )
        logger.info("EMA+RSI WFV complete ✓")
    except Exception as e:
        logger.error(f"EMA+RSI WFV failed: {e}")

    if reports:
        print_wfv_comparison(reports)
    return reports


def step_evaluate(force=False):
    from src.models.xgboost_model import XGBoostTradingModel
    from src.backtesting.metrics  import check_overfitting, monte_carlo_simulation

    logger.info("STEP — Evaluate: Overfitting Check + Monte Carlo")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    try:
        model = XGBoostTradingModel().load()
    except FileNotFoundError:
        logger.error("XGBoost model not found — run --mode train first")
        return

    bt         = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    train_sigs = pd.Series(model.predict(X_tr), index=split.train.index)
    test_sigs  = pd.Series(model.predict(X_te), index=split.test.index)
    train_res  = bt.run(split.train, signals_to_positions(train_sigs), "Train")
    test_res   = bt.run(split.test,  signals_to_positions(test_sigs),  "Test")

    check_overfitting(train_res.metrics, test_res.metrics, threshold=0.5)

    if len(test_res.trades) >= 5:
        monte_carlo_simulation(test_res.trades["pnl_pct"], cfg.INITIAL_CAPITAL, 1000)

    _display_strategy_comparison({"XGBoost Train": train_res, "XGBoost Test": test_res})
    return train_res, test_res


def step_compare(force=False):
    from src.models.xgboost_model import XGBoostTradingModel
    from src.models.xgboost_v2   import XGBoostTradingModelV2

    logger.info("STEP — Full Model Comparison")
    split, feat_cols, _, _, _, _, X_te, y_te = _get_split_and_arrays(force)
    df_test = split.test.copy()
    bt      = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    results = {}

    # ── ML models (XGBoost only — LSTM import skipped to avoid CUDA crash) ───
    for name, loader_fn in [
        ("XGBoost v1", lambda: XGBoostTradingModel().load()),
        ("XGBoost v2", lambda: XGBoostTradingModelV2().load()),
    ]:
        try:
            m    = loader_fn()
            x_in = X_te[m.feature_cols] if (
                hasattr(m,"feature_cols") and m.feature_cols
                and set(m.feature_cols).issubset(X_te.columns)
            ) else X_te
            sigs = pd.Series(m.predict(x_in), index=df_test.index)
            results[name] = bt.run(df_test, signals_to_positions(sigs), name)
            logger.info(f"{name} loaded ✓")
        except FileNotFoundError:
            logger.warning(f"{name} not found")
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    # NOTE: LSTM deliberately excluded — importing torch in this process context
    # triggers an uncatchable Windows 0xC0000005 CUDA crash (not a Python exception).
    # LSTM results available separately via: python main.py --mode train-lstm

    # ── Rule-based ────────────────────────────────────────────────────────────
    for name, strat in [
        ("EMA+RSI",    EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True)),
        ("Confluence", MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)),
    ]:
        results[name] = bt.run(
            df_test, signals_to_positions(strat.generate_signals(df_test)), name
        )

    if results:
        _display_strategy_comparison(results)
        _save_results_json(results)

    return results


def step_report(force=False):
    from src.utils.report_generator import generate_report
    from src.backtesting.metrics    import check_overfitting, monte_carlo_simulation

    logger.info("STEP — Generating Evaluation Report")
    results     = step_compare(force=False)
    eval_output = step_evaluate(force=False)

    train_res = eval_output[0] if eval_output and len(eval_output) > 0 else None
    test_res  = eval_output[1] if eval_output and len(eval_output) > 1 else None

    overfit_checks = {}
    if train_res and test_res:
        overfit_checks["XGBoost v1"] = check_overfitting(
            train_res.metrics, test_res.metrics, threshold=0.5
        )

    report_path = generate_report(
        results        = results,
        overfit_checks = overfit_checks,
        model_meta     = {
            "XGBoost v1 features": "73 (all)",
            "XGBoost v2 features": "56 (stationary only)",
        },
    )
    print(f"\n  Report saved → {report_path}\n")


# ── Display helpers ───────────────────────────────────────────────────────────
def _display_features(df):
    sep       = "═" * 65
    feat_cols = [c for c in df.columns
                 if c not in {"open","high","low","close","volume","target"}]
    print(f"\n{sep}")
    print(f"  FEATURE MATRIX  |  {len(df):,} rows  |  {len(feat_cols)} features")
    print(f"  {df.index[0]}  →  {df.index[-1]}")
    if "target" in df.columns:
        counts = df["target"].value_counts().sort_index()
        total  = len(df)
        labels = {-1:"SHORT", 0:"HOLD", 1:"LONG"}
        for val, cnt in counts.items():
            bar = "█" * int(cnt/total*40)
            print(f"  {labels.get(val,val):>6} ({int(val):>2}): "
                  f"{cnt:>6,} ({cnt/total*100:>5.1f}%)  {bar}")
    print(f"{sep}\n")


def _display_strategy_comparison(results: dict):
    if not results:
        return
    metrics = [
        ("Total Return %",   "total_return_pct",  "{:>+.2f}%"),
        ("Ann. Return %",    "ann_return_pct",     "{:>+.2f}%"),
        ("Buy & Hold %",     "buy_hold_pct",       "{:>+.2f}%"),
        ("Sharpe Ratio",     "sharpe",             "{:>.3f}"),
        ("Sortino Ratio",    "sortino",            "{:>.3f}"),
        ("Max Drawdown %",   "max_drawdown_pct",   "{:>.2f}%"),
        ("Total Trades",     "total_trades",       "{:>}"),
        ("Win Rate %",       "win_rate_pct",       "{:>.1f}%"),
        ("Profit Factor",    "profit_factor",      "{:>.3f}"),
        ("Expectancy %",     "expectancy_pct",     "{:>+.4f}%"),
        ("Time in Market %", "time_in_market_pct", "{:>.1f}%"),
    ]
    names = list(results.keys())
    cw    = 20
    sep   = "═" * (30 + cw * len(names))
    print(f"\n{sep}")
    print(f"  STRATEGY COMPARISON  —  Test Set")
    print(sep)
    print(f"  {'Metric':<28}" + "".join(f"{n:>{cw}}" for n in names))
    print(f"  {'─'*28}" + "─"*(cw*len(names)))
    for label, key, fmt in metrics:
        row = f"  {label:<28}"
        for name in names:
            val = results[name].metrics.get(key, 0)
            try:    row += f"{fmt.format(val):>{cw}}"
            except: row += f"{str(val):>{cw}}"
        print(row)
    print(f"{sep}\n")


# ── Mode runners ──────────────────────────────────────────────────────────────
def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="CryptoTradingBot")
    parser.add_argument(
        "--mode",
        choices=[
            "fetch","load","features","strategy",
            "train","train-v2","train-lstm","train-rl",
            "optimize","compare","wfv","evaluate","report","full",
        ],
        default="full",
    )
    parser.add_argument("--force",      action="store_true", default=False)
    parser.add_argument("--no-optuna",  action="store_true", default=False)
    parser.add_argument("--trials",     type=int, default=20)
    parser.add_argument("--timesteps",  type=int, default=500_000)
    args = parser.parse_args()

    logger.info(
        f"CryptoTradingBot  |  mode={args.mode}  |  "
        f"symbol={cfg.SYMBOL}  |  force={args.force}"
    )

    # GPU check (torch/CUDA) only for modes that actually train models.
    # Importing torch in a fresh process causes a Windows CUDA DLL conflict
    # (0xC0000005) when PyTorch + TensorFlow both try to claim CUDA at startup.
    # Modes that only run inference or backtesting skip the GPU check entirely.
    dispatch = {
        "fetch":      lambda: step_fetch(args.force),
        "load":       lambda: step_load_and_display(),
        "features":   lambda: step_features(args.force),
        "strategy":   lambda: step_strategy(args.force),
        "train":      lambda: (step_gpu_check(), step_train(args.force, args.trials, not args.no_optuna)),
        "train-v2":   lambda: (step_gpu_check(), step_train_v2(args.force, args.trials, not args.no_optuna)),
        "train-lstm": lambda: (step_gpu_check(), step_train_lstm(args.force)),
        "train-rl":   lambda: (step_gpu_check(), step_train_rl(args.force, args.timesteps)),
        "optimize":   lambda: step_optimize(args.force),
        "compare":    lambda: step_compare(args.force),
        "wfv":        lambda: step_wfv(args.force, args.trials),
        "evaluate":   lambda: step_evaluate(args.force),
        "report":     lambda: step_report(args.force),
        "full": lambda: (
            step_fetch(args.force),
            step_gpu_check(),
            step_features(args.force),
            step_strategy(args.force),
            step_train(args.force, args.trials, not args.no_optuna),
            step_train_v2(args.force, args.trials, not args.no_optuna),
            step_optimize(args.force),
            step_wfv(args.force, args.trials),
            step_evaluate(args.force),
            step_compare(args.force),
            step_report(args.force),
        ),
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()