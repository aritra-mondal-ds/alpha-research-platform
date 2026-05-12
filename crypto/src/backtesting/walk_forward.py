# crypto/src/backtesting/walk_forward.py
"""
Walk-Forward Validation (WFV) for time-series trading strategies.

Why WFV instead of a single train/test split:
    A single split tells you performance on ONE market regime.
    WFV rolls the training window forward, testing on unseen data
    at each step, producing N out-of-sample performance samples.
    This reveals whether performance is consistent or regime-specific.

Two modes:
    Expanding window:  train grows each fold (more data over time)
    Rolling window:    train stays fixed size (recent data emphasis)

For XGBoost — this is the primary use case.
For rule-based strategies — no retraining needed, just evaluate each fold.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg


@dataclass
class WFVFold:
    """Results for a single walk-forward fold."""
    fold:        int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    n_train:     int
    n_test:      int
    metrics:     dict = field(default_factory=dict)

    def __str__(self):
        m = self.metrics
        return (
            f"Fold {self.fold:>2} | "
            f"Train: {self.train_start.date()} → {self.train_end.date()} "
            f"({self.n_train:,}) | "
            f"Test: {self.test_start.date()} → {self.test_end.date()} "
            f"({self.n_test:,}) | "
            f"Sharpe: {m.get('sharpe', 0):>+.3f} | "
            f"Return: {m.get('total_return_pct', 0):>+.1f}% | "
            f"Trades: {m.get('total_trades', 0):>3}"
        )


@dataclass
class WFVReport:
    """Aggregated walk-forward validation results."""
    strategy_name: str
    folds:         List[WFVFold]
    mode:          str   # "expanding" or "rolling"

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean([f.metrics.get("sharpe", 0) for f in self.folds]))

    @property
    def mean_return(self) -> float:
        return float(np.mean([f.metrics.get("total_return_pct", 0) for f in self.folds]))

    @property
    def pct_profitable_folds(self) -> float:
        wins = sum(1 for f in self.folds if f.metrics.get("total_return_pct", 0) > 0)
        return wins / len(self.folds) * 100

    @property
    def consistency_score(self) -> float:
        """
        Sharpe of fold-level returns.
        High = consistent performance across regimes.
        Low  = lucky on one regime, fails on others.
        """
        fold_returns = [f.metrics.get("total_return_pct", 0) for f in self.folds]
        if len(fold_returns) < 2:
            return 0.0
        m = np.mean(fold_returns)
        s = np.std(fold_returns)
        return float(m / s) if s > 0 else 0.0

    def summary(self) -> str:
        sep  = "─" * 65
        sep2 = "═" * 65
        lines = [
            f"\n{sep2}",
            f"  WALK-FORWARD VALIDATION — {self.strategy_name}",
            f"  Mode: {self.mode.upper()} | Folds: {len(self.folds)}",
            sep2,
        ]
        for fold in self.folds:
            lines.append(f"  {fold}")
        lines += [
            sep,
            f"  {'Mean Sharpe':<35} {self.mean_sharpe:>+.4f}",
            f"  {'Mean Return %':<35} {self.mean_return:>+.2f}%",
            f"  {'Profitable Folds':<35} {self.pct_profitable_folds:>.1f}%",
            f"  {'Consistency Score (Sharpe of folds)':<35} {self.consistency_score:>+.4f}",
            sep,
            f"  Interpretation:",
            f"  {'Mean Sharpe > 1.0':<35} {'✓ PASS' if self.mean_sharpe > 1.0 else '✗ FAIL'}",
            f"  {'Profitable Folds > 60%':<35} {'✓ PASS' if self.pct_profitable_folds > 60 else '✗ FAIL'}",
            f"  {'Consistency Score > 0.5':<35} {'✓ PASS' if self.consistency_score > 0.5 else '✗ FAIL'}",
            sep2 + "\n",
        ]
        return "\n".join(lines)


# ── Walk-Forward Runner ───────────────────────────────────────────────────────
class WalkForwardValidator:
    """
    Runs walk-forward validation for any strategy or ML model.

    Args:
        train_window_h: Training window in hours (default: 17520 = 2 years)
        test_window_h:  Test window in hours    (default: 4380 = 6 months)
        step_h:         Roll-forward step size  (default: 4380 = 6 months)
        mode:           "expanding" or "rolling"
    """

    def __init__(
        self,
        train_window_h: int = 17_520,    # 2 years of hourly bars
        test_window_h:  int =  4_380,    # 6 months
        step_h:         int =  4_380,    # roll forward 6 months each fold
        mode:           str = "expanding",
    ):
        assert mode in ("expanding", "rolling"), \
            "mode must be 'expanding' or 'rolling'"
        self.train_window_h = train_window_h
        self.test_window_h  = test_window_h
        self.step_h         = step_h
        self.mode           = mode

    def run_xgboost(
        self,
        df:           pd.DataFrame,
        feat_cols:    list,
        n_trials:     int = 20,    # fewer trials per fold for speed
    ) -> WFVReport:
        """
        Walk-forward validation for XGBoost.
        Re-trains model from scratch on each fold's training data.
        """
        from src.models.xgboost_model import XGBoostTradingModel
        from src.backtesting.simple_backtest import SimpleBacktester

        bt     = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
        folds  = self._make_folds(df)
        logger.info(
            f"XGBoost WFV — {len(folds)} folds | "
            f"mode: {self.mode} | Optuna trials per fold: {n_trials}"
        )

        results = []
        for i, (train_df, test_df) in enumerate(folds, 1):
            logger.info(f"Fold {i}/{len(folds)} — training XGBoost ...")

            X_tr = train_df[feat_cols]
            y_tr = train_df["target"].astype(int)
            # Use last 20% of training as validation for early stopping
            val_split = int(len(X_tr) * 0.8)
            X_vl, y_vl = X_tr.iloc[val_split:], y_tr.iloc[val_split:]
            X_tr, y_tr = X_tr.iloc[:val_split], y_tr.iloc[:val_split]

            model = XGBoostTradingModel()
            model.train(X_tr, y_tr, X_vl, y_vl,
                        n_trials=n_trials, use_gpu=True)

            X_te = test_df[feat_cols]
            sigs = pd.Series(
                model.predict(X_te), index=test_df.index, name="signal"
            )
            from src.strategies.position_manager import signals_to_positions
            positions = signals_to_positions(sigs, mode="long_only")
            result    = bt.run(test_df, positions, f"XGBoost Fold {i}")

            fold = WFVFold(
                fold        = i,
                train_start = train_df.index[0],
                train_end   = train_df.index[-1],
                test_start  = test_df.index[0],
                test_end    = test_df.index[-1],
                n_train     = len(train_df),
                n_test      = len(test_df),
                metrics     = result.metrics,
            )
            results.append(fold)
            logger.info(str(fold))

        report = WFVReport("XGBoost", results, self.mode)
        print(report.summary())
        return report

    def run_rule_based(
        self,
        df:       pd.DataFrame,
        strategy,
        name:     str = "Strategy",
    ) -> WFVReport:
        """
        Walk-forward validation for rule-based strategies.
        No retraining — just evaluate on each test fold.
        """
        from src.backtesting.simple_backtest import SimpleBacktester
        from src.strategies.position_manager import signals_to_positions

        bt    = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
        folds = self._make_folds(df)
        logger.info(f"{name} WFV — {len(folds)} folds (no retraining)")

        results = []
        for i, (train_df, test_df) in enumerate(folds, 1):
            sigs      = strategy.generate_signals(test_df)
            positions = signals_to_positions(sigs, mode="long_only")
            result    = bt.run(test_df, positions, f"{name} Fold {i}")
            fold = WFVFold(
                fold        = i,
                train_start = train_df.index[0],
                train_end   = train_df.index[-1],
                test_start  = test_df.index[0],
                test_end    = test_df.index[-1],
                n_train     = len(train_df),
                n_test      = len(test_df),
                metrics     = result.metrics,
            )
            results.append(fold)
            logger.info(str(fold))

        report = WFVReport(name, results, self.mode)
        print(report.summary())
        return report

    # ─────────────────────────────────────────────────────────────────────────
    def _make_folds(
        self, df: pd.DataFrame
    ) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate (train_df, test_df) pairs for all folds.
        """
        folds  = []
        start  = 0
        n      = len(df)

        while start + self.train_window_h + self.test_window_h <= n:
            train_end  = start + self.train_window_h
            test_end   = train_end + self.test_window_h

            if self.mode == "expanding":
                train_df = df.iloc[:train_end].copy()     # grows each fold
            else:
                train_df = df.iloc[start:train_end].copy()  # fixed size

            test_df = df.iloc[train_end:test_end].copy()

            if len(test_df) >= 100:   # skip folds with too little test data
                folds.append((train_df, test_df))

            start += self.step_h

        if not folds:
            raise ValueError(
                f"No folds generated. Dataset has {n} rows but needs at least "
                f"{self.train_window_h + self.test_window_h} rows per fold. "
                f"Try reducing train_window_h or test_window_h."
            )

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds