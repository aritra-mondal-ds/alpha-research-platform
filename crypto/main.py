# crypto/main.py
"""
CryptoTradingBot — Entry Point
==============================
Usage:
    python main.py --mode fetch        # download BTC/USDT 1h + 1d
    python main.py --mode load         # display raw data
    python main.py --mode features     # build feature matrix
    python main.py --mode strategy     # rule-based strategies + backtest
    python main.py --mode train        # train XGBoost v1
    python main.py --mode train-v2     # train XGBoost v2 (anti-overfit)
    python main.py --mode train-lstm   # train LSTM
    python main.py --mode train-rl     # train PPO RL agent
    python main.py --mode optimize     # stationarity analysis + v1 vs v2
    python main.py --mode compare      # compare all saved models
    python main.py --mode wfv          # walk-forward validation
    python main.py --mode evaluate     # overfitting check + Monte Carlo
    python main.py --mode report       # generate full markdown report
    python main.py --mode full         # end-to-end pipeline

Flags:
    --force       Re-download / re-compute even if files exist
    --no-optuna   Skip Optuna, use default params (faster)
    --trials N    Optuna trials (default 20 for WFV/v2, 50 for train)
    --timesteps N PPO total timesteps (default: 500000)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger
import pandas as pd
import numpy as np
import config.settings as cfg

from src.utils.gpu_check              import verify_gpu
from src.utils.experiment_tracker     import (
    ExperimentTracker,
    log_xgboost_run,
    log_wfv_run,
    log_comparison_run,
    MLFLOW_AVAILABLE,
)
from src.utils.report_generator       import generate_report
from src.data.fetcher                 import OHLCVFetcher
from src.data.loader                  import DataLoader, display_dataframe
from src.data.splitter                import temporal_split
from src.features.pipeline            import FeaturePipeline
from src.strategies.ema_rsi           import EMACrossoverRSI
from src.strategies.multi_indicator   import MultiIndicatorConfluence
from src.strategies.position_manager  import signals_to_positions
from src.backtesting.simple_backtest  import SimpleBacktester
from src.backtesting.walk_forward     import WalkForwardValidator
from src.backtesting.metrics          import (
    check_overfitting,
    monte_carlo_simulation,
    print_wfv_comparison,
)
from src.models.xgboost_model         import XGBoostTradingModel
from src.models.xgboost_v2            import XGBoostTradingModelV2
from src.models.lstm_model            import LSTMTradingModel
from src.models.rl.ppo_trainer        import PPOTradingAgent
from src.models.feature_selector      import (
    get_stationary_features,
    print_feature_analysis,
)


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
    logger.info("STEP 1 — GPU Check")
    verify_gpu(verbose=True)


def step_fetch(force=False):
    logger.info("STEP 2 — Data Fetch")
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
    omit = {"open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns if c not in omit]


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

    with ExperimentTracker("rule_based", "rule_based_strategies") as t:
        t.log_system_info()
        for name, result in results.items():
            t.log_backtest_result(result, prefix=name.replace("+","plus").lower())

    return results


def step_train(force=False, n_trials=50, use_optuna=True):
    logger.info("STEP — XGBoost v1 Training")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    logger.info(
        f"X_train: {X_tr.shape} | X_val: {X_vl.shape} | X_test: {X_te.shape}"
    )
    model = XGBoostTradingModel()
    if not use_optuna:
        import xgboost as xgb
        from src.models.xgboost_model import LABEL_MAP
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
        w_tr = model._compute_weights(y_tr_enc)
        w_vl = model._compute_weights(y_vl_enc)
        model.model.fit(
            X_tr, y_tr_enc, sample_weight=w_tr,
            eval_set=[(X_vl, y_vl_enc)],
            sample_weight_eval_set=[w_vl], verbose=100,
        )
        model._print_eval(X_vl, y_vl_enc, "Validation")
        model.save()
    else:
        model.train(X_tr, y_tr, X_vl, y_vl, n_trials=n_trials)

    model.evaluate(X_te, y_te, "Test")
    model.feature_importance(top_n=20)

    df_test    = split.test.copy()
    bt         = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    train_sigs = pd.Series(model.predict(X_tr), index=split.train.index)
    test_sigs  = pd.Series(model.predict(X_te), index=df_test.index)
    train_res  = bt.run(split.train, signals_to_positions(train_sigs), "XGBoost v1 (Train)")
    test_res   = bt.run(df_test,     signals_to_positions(test_sigs),  "XGBoost v1 (Test)")
    _display_strategy_comparison({"XGBoost v1 Train": train_res, "XGBoost v1 Test": test_res})

    log_xgboost_run(model, train_res, test_res, split, version="v1")
    return model, test_res


def step_train_v2(force=False, n_trials=50, use_optuna=True):
    logger.info("STEP — XGBoost v2 Training (Anti-Overfit)")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    logger.info(
        f"X_train: {X_tr.shape} | X_val: {X_vl.shape} | X_test: {X_te.shape}"
    )
    model_v2 = XGBoostTradingModelV2()
    if not use_optuna:
        import xgboost as xgb
        from src.models.xgboost_v2 import LABEL_MAP
        all_feat_df     = pd.concat([X_tr, X_vl], axis=0)
        stationary      = get_stationary_features(all_feat_df)
        model_v2.feature_cols = stationary
        X_tr_s = X_tr[stationary]
        X_vl_s = X_vl[stationary]
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
        w_tr = model_v2._compute_weights(y_tr_enc)
        w_vl = model_v2._compute_weights(y_vl_enc)
        model_v2.model.fit(
            X_tr_s, y_tr_enc, sample_weight=w_tr,
            eval_set=[(X_vl_s, y_vl_enc)],
            sample_weight_eval_set=[w_vl], verbose=50,
        )
        model_v2._print_eval(X_vl_s, y_vl_enc, "Validation")
        model_v2._print_overfit_check(X_tr_s, y_tr_enc, X_vl_s, y_vl_enc)
        model_v2.save()
    else:
        model_v2.train(X_tr, y_tr, X_vl, y_vl, n_trials=n_trials)

    model_v2.evaluate(X_te, y_te, "Test")
    model_v2.feature_importance(top_n=20)

    df_test    = split.test.copy()
    bt         = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    X_te_stat  = X_te[model_v2.feature_cols]
    X_tr_stat  = X_tr[model_v2.feature_cols]
    train_sigs = pd.Series(model_v2.predict(X_tr_stat), index=split.train.index)
    test_sigs  = pd.Series(model_v2.predict(X_te_stat), index=df_test.index)
    train_res  = bt.run(split.train, signals_to_positions(train_sigs), "v2 Train")
    test_res   = bt.run(df_test,     signals_to_positions(test_sigs),  "v2 Test")

    results = {"XGBoost v2": test_res}
    try:
        m1    = XGBoostTradingModel().load()
        v1_sigs = pd.Series(m1.predict(X_te), index=df_test.index)
        results["XGBoost v1"] = bt.run(
            df_test, signals_to_positions(v1_sigs), "XGBoost v1"
        )
    except FileNotFoundError:
        pass
    results["Confluence"] = bt.run(
        df_test,
        signals_to_positions(
            MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)
            .generate_signals(df_test)
        ),
        "Confluence",
    )
    _display_strategy_comparison(results)

    log_xgboost_run(model_v2, train_res, test_res, split, version="v2")
    return model_v2, test_res


def step_train_lstm(force=False):
    logger.info("STEP — LSTM Training")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    lstm = LSTMTradingModel(
        seq_len=60, hidden_size=128, num_layers=2,
        dropout=0.3, batch_size=512, max_epochs=60, lr=1e-3, patience=10,
    )
    lstm.train(X_tr, y_tr, X_vl, y_vl)
    lstm.evaluate(X_te, y_te, "Test")
    df_test  = split.test.copy()
    sigs     = pd.Series(lstm.predict(X_te), index=df_test.index)
    bt       = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    lstm_res = bt.run(df_test, signals_to_positions(sigs), "LSTM")
    rb_res   = bt.run(
        df_test,
        signals_to_positions(
            MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)
            .generate_signals(df_test)
        ),
        "Confluence",
    )
    _display_strategy_comparison({"LSTM": lstm_res, "Confluence": rb_res})

    with ExperimentTracker("lstm", "lstm_training") as t:
        t.log_system_info()
        t.log_params({"seq_len": 60, "hidden_size": 128, "num_layers": 2})
        t.log_backtest_result(lstm_res, prefix="test")

    return lstm, lstm_res


def step_train_rl(force=False, total_timesteps=500_000):
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
    metrics = agent.evaluate(split.test.copy(), feat_cols, window=30)

    with ExperimentTracker("ppo", "ppo_training") as t:
        t.log_system_info()
        t.log_params({"total_timesteps": total_timesteps, "n_envs": 4, "window": 30})
        t.log_metrics({k: v for k, v in metrics.items()
                       if isinstance(v, (int, float))})

    return agent, metrics


def step_optimize(force=False):
    logger.info("STEP — Model Optimization Analysis")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    df        = _load_feature_matrix(force)
    stationary = get_stationary_features(df)
    print_feature_analysis(df, feat_cols)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  FEATURE REDUCTION SUMMARY")
    print(sep)
    removed = [f for f in feat_cols if f not in stationary]
    print(f"  All features        : {len(feat_cols)}")
    print(f"  Stationary (kept)   : {len(stationary)}")
    print(f"  Non-stationary (rm) : {len(removed)}")
    print(f"\n  Removed features:")
    for f in removed:
        print(f"    ✗  {f}")
    print(f"{sep}\n")

    bt = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    results_train, results_test, overfit_checks = {}, {}, {}

    for model_name, loader_fn, x_tr, x_te in [
        ("XGBoost v1", lambda: XGBoostTradingModel().load(), X_tr, X_te),
        ("XGBoost v2", lambda: XGBoostTradingModelV2().load(), None, None),
    ]:
        try:
            m = loader_fn()
            _x_tr = X_tr[m.feature_cols] if hasattr(m, 'feature_cols') and m.feature_cols and X_tr is not None and set(m.feature_cols).issubset(X_tr.columns) else x_tr
            _x_te = X_te[m.feature_cols] if hasattr(m, 'feature_cols') and m.feature_cols and X_te is not None and set(m.feature_cols).issubset(X_te.columns) else x_te

            if _x_tr is not None:
                tr_sigs = pd.Series(m.predict(_x_tr), index=split.train.index)
                results_train[model_name] = bt.run(
                    split.train, signals_to_positions(tr_sigs), f"{model_name} Train"
                )

            if _x_te is not None:
                te_sigs = pd.Series(m.predict(_x_te), index=split.test.index)
                results_test[model_name] = bt.run(
                    split.test, signals_to_positions(te_sigs), f"{model_name} Test"
                )

            if model_name in results_train and model_name in results_test:
                overfit_checks[model_name] = check_overfitting(
                    results_train[model_name].metrics,
                    results_test[model_name].metrics,
                    threshold=0.5,
                )
            logger.info(f"{model_name} evaluated ✓")
        except FileNotFoundError:
            logger.warning(f"{model_name} not found")

    if results_train:
        print(f"\n{'═'*60}\n  IN-SAMPLE COMPARISON (Train)\n{'═'*60}")
        _display_strategy_comparison(results_train)
    if results_test:
        print(f"\n{'═'*60}\n  OUT-OF-SAMPLE COMPARISON (Test)\n{'═'*60}")
        _display_strategy_comparison(results_test)

    best = max(results_test.keys(),
               key=lambda k: results_test[k].metrics.get("sharpe", -99),
               ) if results_test else None
    if best and len(results_test[best].trades) >= 5:
        monte_carlo_simulation(
            results_test[best].trades["pnl_pct"],
            cfg.INITIAL_CAPITAL, 1000,
        )


def step_wfv(force=False, n_trials=20):
    logger.info("STEP — Walk-Forward Validation")
    df        = _load_feature_matrix(force)
    feat_cols = _get_feat_cols(df)

    wfv = WalkForwardValidator(
        train_window_h=17_520, test_window_h=4_380,
        step_h=4_380, mode="expanding",
    )
    reports = {}

    logger.info("XGBoost WFV ...")
    reports["XGBoost"] = wfv.run_xgboost(df, feat_cols, n_trials=n_trials)
    log_wfv_run(reports["XGBoost"], "XGBoost")

    logger.info("Confluence WFV ...")
    reports["Confluence"] = wfv.run_rule_based(
        df,
        MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True),
        name="Confluence",
    )
    log_wfv_run(reports["Confluence"], "Confluence")

    logger.info("EMA+RSI WFV ...")
    reports["EMA+RSI"] = wfv.run_rule_based(
        df,
        EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True),
        name="EMA+RSI",
    )
    log_wfv_run(reports["EMA+RSI"], "EMA+RSI")

    print_wfv_comparison(reports)
    return reports


def step_evaluate(force=False):
    logger.info("STEP — Evaluation: Overfitting Check + Monte Carlo")
    split, feat_cols, X_tr, y_tr, X_vl, y_vl, X_te, y_te = \
        _get_split_and_arrays(force)
    try:
        model = XGBoostTradingModel().load()
    except FileNotFoundError:
        logger.error("XGBoost model not found — run --mode train first")
        return None, None

    bt         = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    train_sigs = pd.Series(model.predict(X_tr), index=split.train.index)
    test_sigs  = pd.Series(model.predict(X_te), index=split.test.index)
    train_res  = bt.run(split.train, signals_to_positions(train_sigs), "XGBoost (Train)")
    test_res   = bt.run(split.test,  signals_to_positions(test_sigs),  "XGBoost (Test)")

    check_overfitting(train_res.metrics, test_res.metrics, threshold=0.5)

    mc = None
    if len(test_res.trades) >= 5:
        mc = monte_carlo_simulation(
            test_res.trades["pnl_pct"], cfg.INITIAL_CAPITAL,
            n_simulations=1000, seed=cfg.RANDOM_SEED,
        )
    else:
        logger.warning(f"Only {len(test_res.trades)} trades — Monte Carlo skipped (need ≥ 5)")

    _display_strategy_comparison({
        "XGBoost (Train)": train_res,
        "XGBoost (Test)":  test_res,
    })
    return train_res, test_res, mc


def step_compare(force=False):
    logger.info("STEP — Full Model Comparison")
    split, feat_cols, _, _, _, _, X_te, y_te = _get_split_and_arrays(force)
    df_test = split.test.copy()
    bt      = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    results = {}

    for model_name, loader_fn in [
        ("XGBoost v1", lambda: XGBoostTradingModel().load()),
        ("XGBoost v2", lambda: XGBoostTradingModelV2().load()),
        ("LSTM",       lambda: LSTMTradingModel().load()),
    ]:
        try:
            m    = loader_fn()
            x_in = X_te[m.feature_cols] if (
                hasattr(m, "feature_cols") and m.feature_cols
                and set(m.feature_cols).issubset(X_te.columns)
            ) else X_te
            raw  = m.predict(x_in)
            results[model_name] = bt.run(
                df_test, signals_to_positions(pd.Series(raw, index=df_test.index)),
                model_name,
            )
            logger.info(f"{model_name} loaded ✓")
        except FileNotFoundError:
            logger.warning(f"{model_name} not found")

    for name, strat in [
        ("EMA+RSI",    EMACrossoverRSI(50, 75, use_ema50_filter=True, long_only=True)),
        ("Confluence", MultiIndicatorConfluence(45, 70, volume_surge=1.2, long_only=True)),
    ]:
        results[name] = bt.run(
            df_test, signals_to_positions(strat.generate_signals(df_test)), name
        )

    if results:
        _display_strategy_comparison(results)
        log_comparison_run(results)
    return results


def step_report(force=False):
    """Generate the final markdown evaluation report."""
    logger.info("STEP — Generating Evaluation Report")

    # Collect whatever results are available
    results      = step_compare(force=False)
    eval_output  = step_evaluate(force=False)

    train_res = eval_output[0] if eval_output and len(eval_output) > 0 else None
    test_res  = eval_output[1] if eval_output and len(eval_output) > 1 else None
    mc        = eval_output[2] if eval_output and len(eval_output) > 2 else None

    overfit_checks = {}
    if train_res and test_res:
        overfit_checks["XGBoost v1"] = check_overfitting(
            train_res.metrics, test_res.metrics, threshold=0.5
        )

    mc_results = {}
    if mc:
        mc_results["XGBoost v1"] = mc

    report_path = generate_report(
        results        = results,
        wfv_reports    = None,    # populate after running --mode wfv
        mc_results     = mc_results,
        overfit_checks = overfit_checks,
        model_meta     = {
            "XGBoost v1 features": "73 (all)",
            "XGBoost v2 features": "56 (stationary only)",
            "Next step":           "Run --mode wfv to validate across regimes",
        },
    )

    print(f"\n  Report saved → {report_path}")
    print(f"  Open in any markdown viewer or VS Code.\n")


# ── Display helpers ───────────────────────────────────────────────────────────
def _display_features(df):
    sep       = "═" * 65
    target    = "target"
    feat_cols = [c for c in df.columns
                 if c not in {"open","high","low","close","volume",target}]
    print(f"\n{sep}")
    print(f"  FEATURE MATRIX  |  {len(df):,} rows  |  {len(feat_cols)} features")
    print(f"  {df.index[0]}  →  {df.index[-1]}")
    groups = {
        "Trend":         [c for c in feat_cols if any(k in c for k in ["ema","macd","adx","cross"])],
        "Momentum":      [c for c in feat_cols if any(k in c for k in ["rsi","stoch","williams","roc"])],
        "Volatility":    [c for c in feat_cols if any(k in c for k in ["bb_","atr","kc_","vol"])],
        "Volume":        [c for c in feat_cols if any(k in c for k in ["volume","obv","vwap","mfi"])],
        "Price-Derived": [c for c in feat_cols if any(k in c for k in ["return","spread","range","log"])],
        "Macro (1d)":    [c for c in feat_cols if c.startswith("d1_")],
        "Patterns":      [c for c in feat_cols if c.startswith("pat_")],
    }
    for g, cols in groups.items():
        print(f"  {g:<18}: {len(cols):>3} cols")
    if target in df.columns:
        counts = df[target].value_counts().sort_index()
        total  = len(df)
        labels = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
        print(f"\n  [ Target Distribution ]")
        for val, cnt in counts.items():
            bar = "█" * int(cnt/total*40)
            print(f"  {labels.get(val,val):>6} ({int(val):>2}): "
                  f"{cnt:>6,} ({cnt/total*100:>5.1f}%)  {bar}")
    nan_ok = not df[feat_cols].isnull().any().any()
    print(f"\n  [ NaN Check ] : "
          f"{'No NaN values ✓' if nan_ok else 'WARNING — NaNs detected'}")
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
    names = list(results.keys())
    cw    = 22
    sep   = "═" * (30 + cw * len(names))
    print(f"\n{sep}")
    print(f"  STRATEGY COMPARISON  —  Test Set")
    print(f"  Targets: Sharpe > 1.5  |  Profit Factor > 1.5  |  Win Rate > 50%")
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
def run_fetch(args):     step_gpu_check(); step_fetch(args.force)
def run_load(args):      step_gpu_check(); step_load_and_display()
def run_features(args):  step_gpu_check(); step_features(args.force)
def run_strategy(args):  step_gpu_check(); step_strategy(args.force)
def run_compare(args):   step_gpu_check(); step_compare(args.force)
def run_wfv(args):       step_gpu_check(); step_wfv(args.force, args.trials)
def run_evaluate(args):  step_gpu_check(); step_evaluate(args.force)
def run_optimize(args):  step_gpu_check(); step_optimize(args.force)
def run_report(args):    step_gpu_check(); step_report(args.force)

def run_train(args):
    step_gpu_check()
    step_train(args.force, args.trials, not args.no_optuna)

def run_train_v2(args):
    step_gpu_check()
    step_train_v2(args.force, args.trials, not args.no_optuna)

def run_train_lstm(args):
    step_gpu_check()
    step_train_lstm(args.force)

def run_train_rl(args):
    step_gpu_check()
    step_train_rl(args.force, args.timesteps)

def run_full(args):
    step_gpu_check()
    step_fetch(args.force)
    step_features(args.force)
    step_strategy(args.force)
    step_train(args.force, args.trials, not args.no_optuna)
    step_train_v2(args.force, args.trials, not args.no_optuna)
    step_train_lstm(args.force)
    step_train_rl(args.force, args.timesteps)
    step_optimize(args.force)
    step_wfv(args.force, args.trials)
    step_evaluate(args.force)
    step_compare(args.force)
    step_report(args.force)
    logger.info("Full pipeline complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
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
    if MLFLOW_AVAILABLE:
        logger.info(f"MLflow tracking enabled → {cfg.BASE_DIR / 'mlflow.db'}")
    else:
        logger.warning("MLflow not available — install with: pip install mlflow==2.13.0")

    dispatch = {
        "fetch":      run_fetch,
        "load":       run_load,
        "features":   run_features,
        "strategy":   run_strategy,
        "train":      run_train,
        "train-v2":   run_train_v2,
        "train-lstm": run_train_lstm,
        "train-rl":   run_train_rl,
        "optimize":   run_optimize,
        "compare":    run_compare,
        "wfv":        run_wfv,
        "evaluate":   run_evaluate,
        "report":     run_report,
        "full":       run_full,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()