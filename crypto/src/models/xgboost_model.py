# crypto/src/models/xgboost_model.py
"""
XGBoost 3-class classifier for BTC/USDT trading signals.

Classes:
    -1 → SHORT / EXIT
     0 → HOLD
    +1 → LONG

Key design decisions:
    - GPU acceleration via device='cuda' (RTX 3060 12GB)
    - Class imbalance handled via inverse-frequency sample weights
    - Optuna tunes on VALIDATION set only — test set never touched
    - SHAP for reliable feature importance
    - Confidence threshold filters low-conviction signals before backtesting
    - Model saved to disk for reproducibility
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import xgboost as xgb
import optuna
from sklearn.metrics import classification_report, confusion_matrix, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg


# ── Label encoding ────────────────────────────────────────────────────────────
# XGBoost multi:softprob requires classes {0, 1, 2}
LABEL_MAP     = {-1: 0,  0: 1,  1: 2}
LABEL_MAP_INV = { 0: -1, 1: 0,  2: 1}
LABEL_NAMES   = { 0: "SHORT", 1: "HOLD", 2: "LONG"}


class XGBoostTradingModel:

    def __init__(self):
        self.model        = None
        self.best_params  = None
        self.feature_cols = None
        self.threshold    = 0.45   # min prob to act — below this → HOLD

    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        X_train:  pd.DataFrame,
        y_train:  pd.Series,
        X_val:    pd.DataFrame,
        y_val:    pd.Series,
        n_trials: int  = 50,
        use_gpu:  bool = True,
    ) -> "XGBoostTradingModel":
        """
        Optuna hyperparameter search → final model training.
        Optimises weighted F1 on validation set.
        """
        self.feature_cols = list(X_train.columns)

        y_tr = y_train.map(LABEL_MAP).values.astype(int)
        y_vl = y_val.map(LABEL_MAP).values.astype(int)

        w_tr = self._compute_weights(y_tr)
        w_vl = self._compute_weights(y_vl)

        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    use_gpu = False
            except Exception:
                use_gpu = False

        device = "cuda" if use_gpu else "cpu"
        logger.info(f"XGBoost — device: {device}")
        logger.info(
            f"Train: {len(X_train):,} | Val: {len(X_val):,} | "
            f"Features: {len(self.feature_cols)}"
        )

        counts = pd.Series(y_tr).value_counts().sort_index()
        logger.info(
            f"Train class dist → "
            f"SHORT: {counts.get(0,0):,} | "
            f"HOLD: {counts.get(1,0):,} | "
            f"LONG: {counts.get(2,0):,}"
        )

        # ── Optuna search ─────────────────────────────────────────────────────
        logger.info(f"Optuna search — {n_trials} trials ...")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("lr", 1e-3, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
                "objective":        "multi:softprob",
                "num_class":        3,
                "eval_metric":      "mlogloss",
                "tree_method":      "hist",
                "device":           device,
                "seed":             cfg.RANDOM_SEED,
                "verbosity":        0,
            }
            clf = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            clf.fit(
                X_train, y_tr,
                sample_weight          = w_tr,
                eval_set               = [(X_val, y_vl)],
                sample_weight_eval_set = [w_vl],
                verbose                = False,
            )
            preds = clf.predict(X_val)
            return f1_score(y_vl, preds, average="weighted", zero_division=0)

        study = optuna.create_study(
            direction  = "maximize",
            study_name = "xgb_btcusdt",
            sampler    = optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        # Rename 'lr' back to 'learning_rate' (Optuna key was shortened)
        if "lr" in self.best_params:
            self.best_params["learning_rate"] = self.best_params.pop("lr")

        logger.info(f"Best val weighted-F1: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # ── Final model ───────────────────────────────────────────────────────
        logger.info("Training final model ...")
        final_params = {
            **self.best_params,
            "objective":   "multi:softprob",
            "num_class":   3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device":      device,
            "seed":        cfg.RANDOM_SEED,
            "verbosity":   0,
        }
        self.model = xgb.XGBClassifier(**final_params, early_stopping_rounds=30)
        self.model.fit(
            X_train, y_tr,
            sample_weight          = w_tr,
            eval_set               = [(X_val, y_vl)],
            sample_weight_eval_set = [w_vl],
            verbose                = 100,
        )

        self._print_eval(X_val, y_vl, label="Validation")
        self.save()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    def predict(
        self,
        X:         pd.DataFrame,
        threshold: float = None,
    ) -> np.ndarray:
        """
        Predict {-1, 0, 1} signals with confidence threshold.
        Bars where max class probability < threshold → forced to HOLD (0).
        """
        assert self.model is not None, "Call .train() or .load() first."
        thresh = threshold if threshold is not None else self.threshold
        proba  = self.predict_proba(X)
        raw    = np.argmax(proba, axis=1)       # 0/1/2
        conf   = proba.max(axis=1)
        raw[conf < thresh] = 1                  # low confidence → HOLD
        return np.vectorize(LABEL_MAP_INV.get)(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 3) softmax probabilities [SHORT, HOLD, LONG]."""
        assert self.model is not None
        return self.model.predict_proba(X)

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self, X: pd.DataFrame, y: pd.Series, label: str = "Test"):
        """Evaluate model on any split. Call with test set after training."""
        y_enc = y.map(LABEL_MAP).values.astype(int)
        self._print_eval(X, y_enc, label=label)

    def _print_eval(self, X, y_enc, label: str = ""):
        preds = self.model.predict(X)
        sep   = "─" * 54
        print(f"\n{'═'*54}")
        print(f"  XGBoost Evaluation — {label}")
        print(f"{'═'*54}")
        print(classification_report(
            y_enc, preds,
            target_names=["SHORT", "HOLD", "LONG"],
            zero_division=0,
        ))
        cm = confusion_matrix(y_enc, preds)
        print(f"  Confusion Matrix:")
        print(sep)
        print(f"  {'':14} {'Pred SHORT':>12} {'Pred HOLD':>12} {'Pred LONG':>12}")
        for i, row_label in enumerate(["Act SHORT", "Act HOLD", "Act LONG"]):
            print(f"  {row_label:<14} {cm[i,0]:>12,} {cm[i,1]:>12,} {cm[i,2]:>12,}")
        print(sep)
        f1 = f1_score(y_enc, preds, average="weighted", zero_division=0)
        print(f"  Weighted F1: {f1:.4f}")
        print(f"{'═'*54}\n")

    # ─────────────────────────────────────────────────────────────────────────
    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """XGBoost gain importance — fast, approximate."""
        assert self.model is not None
        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)

        print(f"\n  Top {top_n} Features — Gain Importance")
        print("─" * 48)
        for _, row in imp.iterrows():
            bar = "█" * int(row["importance"] * 400)
            print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")
        print("─" * 48)
        return imp

    def shap_importance(
        self, X_sample: pd.DataFrame, top_n: int = 20
    ) -> pd.DataFrame:
        """SHAP-based importance — more reliable, slower."""
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — pip install shap")
            return self.feature_importance(top_n)

        logger.info(f"Computing SHAP on {len(X_sample):,} samples ...")
        explainer = shap.TreeExplainer(self.model)
        sv        = explainer.shap_values(X_sample)
        mean_shap = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 \
                    else np.abs(sv).mean(axis=0)

        imp = pd.DataFrame({
            "feature":   self.feature_cols,
            "shap_mean": mean_shap,
        }).sort_values("shap_mean", ascending=False).head(top_n)

        print(f"\n  Top {top_n} Features — SHAP Mean |value|")
        print("─" * 52)
        for _, row in imp.iterrows():
            bar = "█" * int(row["shap_mean"] * 600)
            print(f"  {row['feature']:<30} {row['shap_mean']:.4f}  {bar}")
        print("─" * 52)
        return imp

    # ─────────────────────────────────────────────────────────────────────────
    def save(self):
        out_dir = cfg.MODEL_DIR / "xgboost"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out_dir / "model.json"))
        with open(out_dir / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        with open(out_dir / "meta.json", "w") as f:
            json.dump({
                "feature_cols": list(self.feature_cols) if self.feature_cols is not None else [],
                "threshold": float(self.threshold),
            }, f, indent=2)
        with open(out_dir / "meta.pkl", "wb") as f:
            pickle.dump({
                "feature_cols": self.feature_cols,
                "threshold":    self.threshold,
            }, f)
        logger.info(f"XGBoost model saved → {out_dir}")

    def load(self) -> "XGBoostTradingModel":
        out_dir = cfg.MODEL_DIR / "xgboost"
        if not (out_dir / "model.json").exists():
            raise FileNotFoundError(f"No model at {out_dir}. Train first.")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(out_dir / "model.json"))
        with open(out_dir / "best_params.json") as f:
            self.best_params = json.load(f)
        with open(out_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]
        self.threshold    = meta["threshold"]
        logger.info(f"XGBoost model loaded ← {out_dir}")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_weights(y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse-frequency sample weights to combat class imbalance.
        HOLD bars are downweighted; LONG/SHORT bars get higher weight.
        Normalised to mean=1 so learning rate scale is preserved.
        """
        counts  = np.bincount(y_encoded, minlength=3).astype(float)
        total   = counts.sum()
        inv_freq = total / (3 * counts + 1e-9)
        weights  = inv_freq[y_encoded]
        return weights / weights.mean()