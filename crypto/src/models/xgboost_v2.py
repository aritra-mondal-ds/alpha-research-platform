# crypto/src/models/xgboost_v2.py
"""
XGBoost v2 — Anti-Overfitting Edition.

Changes from v1:
    1. STATIONARY features only (no absolute price levels)
    2. Tighter regularisation search space in Optuna
    3. max_depth capped at 6 (was 8)
    4. Temporal validation split (last 20% of train, not random)
    5. max_leaves constraint added (limits tree complexity)
    6. Overfitting guard: if train F1 / val F1 > 1.5, add penalty
    7. Early stopping patience reduced to 20 (was 30)

Expected outcome:
    - Train return: should drop from billions% to something reasonable
    - Test return: may drop slightly but gap to train should narrow
    - Train/Test Sharpe ratio: should move toward 1.0
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
from src.models.feature_selector import get_stationary_features

LABEL_MAP     = {-1: 0,  0: 1,  1: 2}
LABEL_MAP_INV = { 0: -1, 1: 0,  2: 1}


class XGBoostTradingModelV2:
    """
    Anti-overfitting XGBoost. Uses stationary features only.
    Tighter regularisation. Temporal validation split.
    """

    VERSION = "2.0"

    def __init__(self):
        self.model        = None
        self.best_params  = None
        self.feature_cols = None
        self.threshold    = 0.45

    def train(
        self,
        X_train:  pd.DataFrame,
        y_train:  pd.Series,
        X_val:    pd.DataFrame,
        y_val:    pd.Series,
        n_trials: int  = 50,
        use_gpu:  bool = False,
    ) -> "XGBoostTradingModelV2":

        # ── Step 1: Filter to stationary features only ────────────────────────
        all_features   = pd.concat([X_train, X_val], axis=0)
        stationary     = get_stationary_features(all_features)
        self.feature_cols = stationary

        X_tr = X_train[stationary]
        X_vl = X_val[stationary]

        y_tr = y_train.map(LABEL_MAP).values.astype(int)
        y_vl = y_val.map(LABEL_MAP).values.astype(int)

        w_tr = self._compute_weights(y_tr)
        w_vl = self._compute_weights(y_vl)
        device = "cuda" if use_gpu else "cpu"











        logger.info(
            f"XGBoost v2 | device: {device} | "
            f"stationary features: {len(stationary)} | "
            f"train: {len(X_tr):,} | val: {len(X_vl):,}"
        )

        # Log class distribution
        counts = pd.Series(y_tr).value_counts().sort_index()
        logger.info(
            f"Train dist → SHORT: {counts.get(0,0):,} | "
            f"HOLD: {counts.get(1,0):,} | LONG: {counts.get(2,0):,}"
        )

        # ── Step 2: Optuna search — tighter search space ──────────────────────
        logger.info(f"Optuna search — {n_trials} trials (anti-overfit constraints) ...")

        def objective(trial: optuna.Trial) -> float:
            params = {
                # Fewer trees — cap at 600 (was 1000)
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                # Shallower trees — cap at 6 (was 8)
                "max_depth":        trial.suggest_int("max_depth", 3, 6),
                # Limit leaves — hard cap on tree complexity
                "max_leaves":       trial.suggest_int("max_leaves", 8, 64),
                "learning_rate":    trial.suggest_float("lr", 1e-3, 0.1, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 0.85),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
                # Stronger L1/L2 regularisation
                "reg_alpha":        trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
                # Higher min_child_weight prevents small-sample splits
                "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
                "gamma":            trial.suggest_float("gamma", 1.0, 10.0),
                "objective":        "multi:softprob",
                "num_class":        3,
                "eval_metric":      "mlogloss",
                "tree_method":      "hist",
                "device":           device,
                "seed":             cfg.RANDOM_SEED,
                "verbosity":        0,
            }
            clf = xgb.XGBClassifier(**params, early_stopping_rounds=20)
            clf.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_set=[(X_vl, y_vl)],
                sample_weight_eval_set=[w_vl],
                verbose=False,
            )

            val_preds  = clf.predict(X_vl)
            val_f1     = f1_score(y_vl, val_preds, average="weighted", zero_division=0)

            # Overfitting penalty: if train F1 >> val F1, penalise
            train_preds = clf.predict(X_tr)
            train_f1    = f1_score(y_tr, train_preds, average="weighted", zero_division=0)
            overfit_gap = max(0, train_f1 - val_f1 - 0.15)   # allow 15% gap
            penalised   = val_f1 - 0.5 * overfit_gap

            return penalised

        study = optuna.create_study(
            direction  = "maximize",
            study_name = "xgb_v2_btcusdt",
            sampler    = optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        if "lr" in self.best_params:
            self.best_params["learning_rate"] = self.best_params.pop("lr")

        logger.info(f"Best penalised val F1: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # ── Step 3: Train final model ─────────────────────────────────────────
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
        self.model = xgb.XGBClassifier(**final_params, early_stopping_rounds=20)
        self.model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_vl, y_vl)],
            sample_weight_eval_set=[w_vl],
            verbose=50,
        )

        self._print_eval(X_vl, y_vl, label="Validation")
        self._print_overfit_check(X_tr, y_tr, X_vl, y_vl)
        self.save()
        return self

    def _print_overfit_check(self, X_tr, y_tr, X_vl, y_vl):
        """Quick in-training overfitting diagnostic."""
        train_f1 = f1_score(
            y_tr, self.model.predict(X_tr), average="weighted", zero_division=0
        )
        val_f1 = f1_score(
            y_vl, self.model.predict(X_vl), average="weighted", zero_division=0
        )
        gap    = train_f1 - val_f1
        status = "✓ OK" if gap < 0.15 else ("⚠ WARN" if gap < 0.25 else "✗ OVERFIT")
        sep    = "─" * 40
        print(f"\n{sep}")
        print(f"  Overfit Check (F1 gap)")
        print(sep)
        print(f"  Train F1 : {train_f1:.4f}")
        print(f"  Val   F1 : {val_f1:.4f}")
        print(f"  Gap      : {gap:.4f}  →  {status}")
        print(f"{sep}\n")

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        assert self.model is not None
        thresh = threshold if threshold is not None else self.threshold
        X_sel  = X[self.feature_cols] if set(self.feature_cols).issubset(X.columns) \
                 else X
        proba  = self.model.predict_proba(X_sel)
        raw    = np.argmax(proba, axis=1)
        raw[proba.max(axis=1) < thresh] = 1
        return np.vectorize(LABEL_MAP_INV.get)(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        X_sel = X[self.feature_cols] if set(self.feature_cols).issubset(X.columns) \
                else X
        return self.model.predict_proba(X_sel)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, label: str = "Test"):
        X_sel = X[self.feature_cols] if set(self.feature_cols).issubset(X.columns) \
                else X
        y_enc = y.map(LABEL_MAP).values.astype(int)
        self._print_eval(X_sel, y_enc, label=label)

    def _print_eval(self, X, y_enc, label: str = ""):
        preds = self.model.predict(X)
        sep   = "─" * 54
        print(f"\n{'═'*54}")
        print(f"  XGBoost v2 Evaluation — {label}")
        print(f"{'═'*54}")
        print(classification_report(
            y_enc, preds,
            target_names=["SHORT", "HOLD", "LONG"], zero_division=0,
        ))
        cm = confusion_matrix(y_enc, preds)
        print(f"  Confusion Matrix:")
        print(sep)
        print(f"  {'':14} {'Pred SHORT':>12} {'Pred HOLD':>12} {'Pred LONG':>12}")
        for i, rl in enumerate(["Act SHORT", "Act HOLD", "Act LONG"]):
            print(f"  {rl:<14} {cm[i,0]:>12,} {cm[i,1]:>12,} {cm[i,2]:>12,}")
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
        print(f"\n  Top {top_n} Features — Gain (v2, stationary only)")
        print("─" * 48)
        for _, row in imp.iterrows():
            bar = "█" * int(row["importance"] * 400)
            print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")
        print("─" * 48)
        return imp

    def save(self):
        out_dir = cfg.MODEL_DIR / "xgboost_v2"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out_dir / "model.json"))
        with open(out_dir / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        with open(out_dir / "meta.json", "w") as f:
            json.dump({
                "feature_cols": list(self.feature_cols) if self.feature_cols is not None else [],
                "threshold": float(self.threshold),
                "version": self.VERSION,
            }, f, indent=2)
        with open(out_dir / "meta.pkl", "wb") as f:
            pickle.dump({
                "feature_cols": self.feature_cols,
                "threshold":    self.threshold,
                "version":      self.VERSION,
            }, f)
        logger.info(f"XGBoost v2 saved → {out_dir}")

    def load(self) -> "XGBoostTradingModelV2":
        out_dir = cfg.MODEL_DIR / "xgboost_v2"
        if not (out_dir / "model.json").exists():
            raise FileNotFoundError(f"No v2 model at {out_dir}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(out_dir / "model.json"))
        with open(out_dir / "best_params.json") as f:
            self.best_params = json.load(f)
        with open(out_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]
        self.threshold    = meta["threshold"]
        logger.info(f"XGBoost v2 loaded ← {out_dir}")
        return self

    @staticmethod
    def _compute_weights(y_encoded: np.ndarray) -> np.ndarray:
        counts   = np.bincount(y_encoded, minlength=3).astype(float)
        total    = counts.sum()
        inv_freq = total / (3 * counts + 1e-9)
        weights  = inv_freq[y_encoded]
        return weights / weights.mean()