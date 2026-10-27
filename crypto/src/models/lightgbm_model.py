# crypto/src/models/lightgbm_model.py
"""
LightGBM 3-class classifier for quantitative trading signals.

Classes:
    -1 → SHORT / EXIT
     0 → HOLD
    +1 → LONG

Key Design Principles:
    - High-efficiency CPU training with histogram binning (safe on all platforms)
    - Anti-overfitting regularization: L1/L2 penalties, feature fraction, leaf constraints
    - Class imbalance handling via inverse-frequency sample weights
    - Optuna hyperparameter optimization strictly on validation splits
    - Seamless metadata serialization (features, scaling, thresholds) for live execution
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

import lightgbm as lgb
import optuna
from sklearn.metrics import classification_report, confusion_matrix, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg
from src.models.feature_selector import get_stationary_features

# Label encoding for multi-class classification
LABEL_MAP     = {-1: 0,  0: 1,  1: 2}
LABEL_MAP_INV = { 0: -1, 1: 0,  2: 1}
LABEL_NAMES   = { 0: "SHORT", 1: "HOLD", 2: "LONG"}


class LightGBMTradingModel:
    """
    Institutional LightGBM classifier with stationary feature selection
    and CPU-optimized hyperparameter search.
    """

    VERSION = "1.0"

    def __init__(self, use_stationary_only: bool = True):
        self.model               = None
        self.best_params         = None
        self.feature_cols        = None
        self.threshold           = 0.40  # minimum conviction threshold
        self.use_stationary_only = use_stationary_only

    def train(
        self,
        X_train:  pd.DataFrame,
        y_train:  pd.Series,
        X_val:    pd.DataFrame,
        y_val:    pd.Series,
        n_trials: int = 50,
    ) -> "LightGBMTradingModel":
        """
        Optuna search optimizing validation weighted F1 with early stopping.
        """
        if self.use_stationary_only:
            all_feats = pd.concat([X_train, X_val], axis=0)
            self.feature_cols = get_stationary_features(all_feats)
        else:
            self.feature_cols = list(X_train.columns)

        X_tr = X_train[self.feature_cols]
        X_vl = X_val[self.feature_cols]

        y_tr = y_train.map(LABEL_MAP).values.astype(int)
        y_vl = y_val.map(LABEL_MAP).values.astype(int)

        w_tr = self._compute_weights(y_tr)
        w_vl = self._compute_weights(y_vl)

        logger.info(
            f"LightGBM Training | features: {len(self.feature_cols)} | "
            f"train: {len(X_tr):,} | val: {len(X_vl):,}"
        )

        counts = pd.Series(y_tr).value_counts().sort_index()
        logger.info(
            f"Train distribution → SHORT: {counts.get(0,0):,} | "
            f"HOLD: {counts.get(1,0):,} | LONG: {counts.get(2,0):,}"
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective":         "multiclass",
                "num_class":         3,
                "metric":            "multi_logloss",
                "boosting_type":     "gbdt",
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "learning_rate":     trial.suggest_float("lr", 0.01, 0.2, log=True),
                "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
                "max_depth":         trial.suggest_int("max_depth", 3, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state":      cfg.RANDOM_SEED,
                "n_jobs":            -1,
                "verbose":           -1,
            }

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_vl, y_vl)],
                eval_sample_weight=[w_vl],
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            )
            preds = clf.predict(X_vl)
            return f1_score(y_vl, preds, average="weighted", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params

        logger.info(f"Optuna Best Val F1: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Train final model with best params
        final_params = {
            "objective":         "multiclass",
            "num_class":         3,
            "metric":            "multi_logloss",
            "boosting_type":     "gbdt",
            "n_estimators":      self.best_params["n_estimators"],
            "learning_rate":     self.best_params["lr"],
            "num_leaves":        self.best_params["num_leaves"],
            "max_depth":         self.best_params["max_depth"],
            "min_child_samples": self.best_params["min_child_samples"],
            "subsample":         self.best_params["subsample"],
            "colsample_bytree":  self.best_params["colsample_bytree"],
            "reg_alpha":         self.best_params["reg_alpha"],
            "reg_lambda":        self.best_params["reg_lambda"],
            "random_state":      cfg.RANDOM_SEED,
            "n_jobs":            -1,
            "verbose":           -1,
        }

        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_vl, y_vl)],
            eval_sample_weight=[w_vl],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        self._print_eval(X_vl, y_vl, label="Validation")
        self.save()
        return self

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Generate {-1, 0, 1} trading signals applying minimum confidence threshold.
        """
        assert self.model is not None, "Call .train() or .load() first."
        thresh = threshold if threshold is not None else self.threshold
        X_in   = X[self.feature_cols] if set(self.feature_cols).issubset(X.columns) else X
        proba  = self.predict_proba(X_in)
        raw    = np.argmax(proba, axis=1)
        conf   = proba.max(axis=1)
        raw[conf < thresh] = 1  # filter low confidence to HOLD
        return np.vectorize(LABEL_MAP_INV.get)(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 3) probabilities [SHORT, HOLD, LONG]."""
        assert self.model is not None, "Call .train() or .load() first."
        X_in = X[self.feature_cols] if set(self.feature_cols).issubset(X.columns) else X
        return self.model.predict_proba(X_in)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, label: str = "Test"):
        y_enc = y.map(LABEL_MAP).values.astype(int)
        X_in  = X[self.feature_cols] if set(self.feature_cols).issubset(X.columns) else X
        self._print_eval(X_in, y_enc, label=label)

    def _print_eval(self, X, y_enc, label: str = ""):
        preds = self.model.predict(X)
        sep   = "─" * 54
        print(f"\n{'═'*54}")
        print(f"  LightGBM Evaluation — {label}")
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

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        assert self.model is not None
        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)

        print(f"\n  Top {top_n} Features — LightGBM Split Importance")
        print("─" * 48)
        for _, row in imp.iterrows():
            bar = "█" * int(row["importance"] / (imp["importance"].max() + 1e-9) * 30)
            print(f"  {row['feature']:<30} {row['importance']:>6}  {bar}")
        print("─" * 48)
        return imp

    def save(self):
        out_dir = cfg.MODEL_DIR / "lightgbm"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(out_dir / "model.txt"))
        with open(out_dir / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        with open(out_dir / "meta.json", "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "threshold":    self.threshold,
                "version":      self.VERSION,
            }, f, indent=2)
        logger.info(f"LightGBM model and metadata saved → {out_dir}")

    def load(self) -> "LightGBMTradingModel":
        out_dir = cfg.MODEL_DIR / "lightgbm"
        model_file = out_dir / "model.txt"
        meta_file  = out_dir / "meta.json"
        if not model_file.exists():
            raise FileNotFoundError(f"No model found at {model_file}. Train first.")

        booster = lgb.Booster(model_file=str(model_file))
        self.model = lgb.LGBMClassifier()
        self.model._Booster = booster
        self.model.fitted_ = True

        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            self.feature_cols = meta.get("feature_cols", [])
            self.threshold    = meta.get("threshold", 0.40)

        if (out_dir / "best_params.json").exists():
            with open(out_dir / "best_params.json") as f:
                self.best_params = json.load(f)

        logger.info(f"LightGBM model loaded ← {out_dir} ({len(self.feature_cols)} features)")
        return self

    @staticmethod
    def _compute_weights(y_encoded: np.ndarray) -> np.ndarray:
        counts   = np.bincount(y_encoded, minlength=3).astype(float)
        total    = counts.sum()
        inv_freq = total / (3 * counts + 1e-9)
        weights  = inv_freq[y_encoded]
        return weights / weights.mean()

