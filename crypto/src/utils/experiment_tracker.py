# crypto/src/utils/experiment_tracker.py
"""
MLflow experiment tracking for all trading model runs.

Tracks:
    - Model hyperparameters
    - Backtest performance metrics
    - Feature importance
    - Overfitting ratios
    - Walk-forward fold results
    - Artifacts: model files, config, feature lists

Setup (run once before first use):
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 --port 5000

Then open http://localhost:5000 in your browser.
"""

import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg

try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — run: pip install mlflow==2.13.0")


# ── Experiment names ──────────────────────────────────────────────────────────
EXPERIMENTS = {
    "xgboost_v1":  "crypto_xgboost_v1",
    "xgboost_v2":  "crypto_xgboost_v2",
    "lstm":        "crypto_lstm",
    "ppo":         "crypto_ppo",
    "rule_based":  "crypto_rule_based",
    "wfv":         "crypto_walk_forward",
    "comparison":  "crypto_model_comparison",
}

TRACKING_URI = f"sqlite:///{cfg.BASE_DIR / 'mlflow.db'}"


def _get_git_commit() -> str:
    """Get current git commit hash for reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cfg.BASE_DIR,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _setup_mlflow(experiment_key: str) -> Optional[str]:
    """Initialise MLflow and return experiment ID."""
    if not MLFLOW_AVAILABLE:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    exp_name = EXPERIMENTS.get(experiment_key, experiment_key)
    exp      = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            exp_name,
            artifact_location=str(cfg.BASE_DIR / "mlruns" / exp_name),
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(exp_name)
    return exp_id


# ── Logging helpers ───────────────────────────────────────────────────────────
class ExperimentTracker:
    """
    Context manager for MLflow run tracking.
    Falls back to local JSON logging if MLflow is unavailable.
    """

    def __init__(self, experiment_key: str, run_name: str):
        self.experiment_key = experiment_key
        self.run_name       = run_name
        self.run            = None
        self._local_log     = {}

    def __enter__(self):
        if MLFLOW_AVAILABLE:
            _setup_mlflow(self.experiment_key)
            self.run = mlflow.start_run(run_name=self.run_name)
            logger.info(
                f"MLflow run started: {self.run_name} "
                f"(id: {self.run.info.run_id[:8]})"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if MLFLOW_AVAILABLE and self.run:
            mlflow.end_run()
        if self._local_log:
            self._save_local()

    def log_params(self, params: dict):
        """Log hyperparameters."""
        if MLFLOW_AVAILABLE:
            mlflow.log_params(params)
        self._local_log["params"] = params

    def log_metrics(self, metrics: dict, step: int = None):
        """Log backtest performance metrics."""
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)
        self._local_log.setdefault("metrics", {}).update(metrics)

    def log_artifact(self, path: str):
        """Log a file as an artifact."""
        if MLFLOW_AVAILABLE and Path(path).exists():
            mlflow.log_artifact(path)

    def log_model_xgboost(self, model, name: str = "model"):
        """Log XGBoost model."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.xgboost.log_model(model, name)
            except Exception as e:
                logger.warning(f"Could not log XGBoost model: {e}")

    def log_feature_importance(self, importance_df: pd.DataFrame):
        """Log feature importance as a CSV artifact."""
        if importance_df is None or importance_df.empty:
            return
        path = cfg.BASE_DIR / "logs" / f"feature_importance_{self.run_name}.csv"
        importance_df.to_csv(path, index=False)
        self.log_artifact(str(path))

    def log_backtest_result(self, result, prefix: str = ""):
        """Log all backtest metrics from a BacktestResult object."""
        if result is None:
            return
        prefix = f"{prefix}_" if prefix else ""
        metrics = {f"{prefix}{k}": v for k, v in result.metrics.items()
                   if isinstance(v, (int, float))}
        self.log_metrics(metrics)

    def log_system_info(self):
        """Log hardware and software context."""
        import torch
        params = {
            "python_version":   sys.version.split()[0],
            "torch_version":    torch.__version__,
            "cuda_available":   torch.cuda.is_available(),
            "gpu_name":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "symbol":           cfg.SYMBOL,
            "primary_tf":       cfg.PRIMARY_TF,
            "random_seed":      cfg.RANDOM_SEED,
            "commission":       cfg.COMMISSION,
            "slippage":         cfg.SLIPPAGE,
            "initial_capital":  cfg.INITIAL_CAPITAL,
            "git_commit":       _get_git_commit(),
        }
        self.log_params(params)

    def _save_local(self):
        """Fallback: save run data to JSON when MLflow unavailable."""
        log_dir = cfg.BASE_DIR / "logs" / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.run_name}.json"
        with open(path, "w") as f:
            json.dump(self._local_log, f, indent=2, default=str)
        logger.info(f"Run logged locally → {path}")


# ── Convenience wrappers ──────────────────────────────────────────────────────
def log_xgboost_run(
    model,
    train_result,
    test_result,
    split,
    version: str = "v1",
):
    """Log a complete XGBoost training run."""
    with ExperimentTracker(f"xgboost_{version}", f"xgb_{version}_{cfg.SYMBOL}") as t:
        t.log_system_info()
        t.log_params({
            "model_version":   version,
            "n_features":      len(model.feature_cols) if model.feature_cols else "unknown",
            "train_start":     str(split.train.index[0].date()),
            "train_end":       str(split.train.index[-1].date()),
            "test_start":      str(split.test.index[0].date()),
            "test_end":        str(split.test.index[-1].date()),
            **(model.best_params or {}),
        })
        t.log_backtest_result(train_result, prefix="train")
        t.log_backtest_result(test_result,  prefix="test")

        if train_result and test_result:
            # Compute train/test ratios for quick overfit monitoring
            for metric in ["sharpe", "win_rate_pct", "profit_factor"]:
                tr  = train_result.metrics.get(metric, 0)
                te  = test_result.metrics.get(metric, 0)
                ratio = te / (tr + 1e-9)
                t.log_metrics({f"ratio_{metric}": ratio})

        t.log_model_xgboost(model.model, name=f"xgboost_{version}")
        t.log_artifact(str(cfg.MODEL_DIR / f"xgboost_{version}" / "model.json"))
        t.log_artifact(str(cfg.MODEL_DIR / f"xgboost_{version}" / "best_params.json"))

    logger.info(f"XGBoost {version} run logged to MLflow ✓")


def log_wfv_run(report, strategy_name: str):
    """Log a walk-forward validation report."""
    with ExperimentTracker("wfv", f"wfv_{strategy_name}") as t:
        t.log_system_info()
        t.log_params({
            "strategy":      strategy_name,
            "n_folds":       len(report.folds),
            "wfv_mode":      report.mode,
        })
        t.log_metrics({
            "mean_sharpe":              report.mean_sharpe,
            "mean_return_pct":          report.mean_return,
            "pct_profitable_folds":     report.pct_profitable_folds,
            "consistency_score":        report.consistency_score,
        })
        for fold in report.folds:
            step = fold.fold
            t.log_metrics({
                f"fold_{step}_sharpe":       fold.metrics.get("sharpe", 0),
                f"fold_{step}_return_pct":   fold.metrics.get("total_return_pct", 0),
                f"fold_{step}_win_rate":     fold.metrics.get("win_rate_pct", 0),
            }, step=step)
    logger.info(f"WFV run for {strategy_name} logged ✓")


def log_comparison_run(results: dict):
    """Log a model comparison run with all strategies."""
    with ExperimentTracker("comparison", f"comparison_{cfg.SYMBOL}") as t:
        t.log_system_info()
        for name, result in results.items():
            safe_name = name.lower().replace(" ", "_").replace("+", "plus")
            t.log_backtest_result(result, prefix=safe_name)
    logger.info("Comparison run logged to MLflow ✓")