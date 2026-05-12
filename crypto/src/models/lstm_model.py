# crypto/src/models/lstm_model.py
"""
LSTM sequence classifier for BTC/USDT trading signals.

Architecture:
    Input:  (batch, seq_len, n_features)  — sliding window of 60 bars
    LSTM:   2 stacked layers, hidden_size=128, dropout=0.3
    Head:   FC(128→64) → ReLU → Dropout → FC(64→3)
    Output: softmax over {SHORT, HOLD, LONG}

Key differences vs XGBoost:
    - Captures temporal dependencies across 60 bars (2.5 days of 1h data)
    - Learns sequential patterns: e.g. "RSI rising + EMA converging over time"
    - More data-hungry — benefits from the full 35k training rows
    - GPU acceleration via CUDA (RTX 3060)

Lookahead contract:
    Each window [t-59 .. t] predicts the label at t.
    Labels were built with shift(-12) in target.py — safe.
    Scaler is fit on train only, applied to val/test.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label encoding — same convention as XGBoost model
LABEL_MAP     = {-1: 0,  0: 1,  1: 2}
LABEL_MAP_INV = { 0: -1, 1: 0,  2: 1}


# ── Dataset ───────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    Sliding window dataset.
    Window at position i = features[i : i+seq_len]
    Label at position i = y[i+seq_len-1]  (label for last bar in window)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 60,
    ):
        self.seq_len = seq_len
        # Pre-allocate tensors for speed
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],   # (seq_len, n_features)
            self.y[idx + self.seq_len - 1],      # scalar label for last bar
        )


# ── Model architecture ────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Stacked LSTM → fully-connected classifier.

    Regularisation:
        - Dropout between LSTM layers (inter-layer)
        - Dropout before FC head (output regularisation)
        - Weight decay (AdamW) during training

    Batch norm on input helps with non-stationary financial data.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        num_classes: int   = 3,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.input_bn = nn.BatchNorm1d(input_size)   # normalise each feature

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        batch, seq, feats = x.shape

        # Apply BatchNorm per feature across (batch × seq) dimension
        x_2d = x.reshape(batch * seq, feats)
        x_2d = self.input_bn(x_2d)
        x    = x_2d.reshape(batch, seq, feats)

        out, _ = self.lstm(x)        # (batch, seq, hidden)
        last   = out[:, -1, :]       # take last timestep
        return self.head(last)       # (batch, num_classes)


# ── Trainer ───────────────────────────────────────────────────────────────────
class LSTMTradingModel:

    def __init__(
        self,
        seq_len:     int   = 60,     # 60 hourly bars = 2.5 days context
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        batch_size:  int   = 512,    # RTX 3060 12GB handles this comfortably
        max_epochs:  int   = 60,
        lr:          float = 1e-3,
        patience:    int   = 10,     # early stopping patience
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.lr          = lr
        self.patience    = patience

        self.model        = None
        self.scaler       = None
        self.feature_cols = None
        self.threshold    = 0.45
        self.input_size   = None

    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> "LSTMTradingModel":
        """
        Train LSTM with early stopping on validation loss.
        Scaler is fit on train only.
        """
        self.feature_cols = list(X_train.columns)
        self.input_size   = len(self.feature_cols)

        # ── Encode labels ─────────────────────────────────────────────────────
        y_tr = y_train.map(LABEL_MAP).values.astype(int)
        y_vl = y_val.map(LABEL_MAP).values.astype(int)

        # ── Scale features — fit on TRAIN only ───────────────────────────────
        self.scaler = StandardScaler()
        X_tr_sc = self.scaler.fit_transform(X_train.values).astype(np.float32)
        X_vl_sc = self.scaler.transform(X_val.values).astype(np.float32)

        # ── Class weights for loss function ───────────────────────────────────
        counts      = np.bincount(y_tr, minlength=3).astype(float)
        total       = counts.sum()
        class_wts   = torch.tensor(
            total / (3 * counts + 1e-9), dtype=torch.float32
        ).to(DEVICE)

        # ── Datasets & loaders ────────────────────────────────────────────────
        train_ds = SequenceDataset(X_tr_sc, y_tr, self.seq_len)
        val_ds   = SequenceDataset(X_vl_sc, y_vl, self.seq_len)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=False,          # time-series — DO NOT shuffle
            pin_memory=True,
            num_workers=0,          # Windows: keep 0
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=0,
        )

        logger.info(
            f"LSTM training — device: {DEVICE} | "
            f"seq_len: {self.seq_len} | batch: {self.batch_size} | "
            f"train sequences: {len(train_ds):,} | val sequences: {len(val_ds):,}"
        )

        # ── Model, optimizer, scheduler ───────────────────────────────────────
        self.model = LSTMClassifier(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            num_classes = 3,
            dropout     = self.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss(weight=class_wts)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5
        )

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        patience_cnt  = 0
        best_path     = cfg.MODEL_DIR / "lstm" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training — max {self.max_epochs} epochs ...")
        sep = "─" * 60

        print(f"\n{sep}")
        print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
              f"{'Val F1':>8}  {'LR':>10}")
        print(sep)

        for epoch in range(1, self.max_epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_losses, all_preds, all_labels = [], [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    logits = self.model(Xb)
                    loss   = criterion(logits, yb)
                    val_losses.append(loss.item())
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            val_f1     = f1_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"  {epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
                  f"{val_f1:>8.4f}  {current_lr:>10.2e}")

            scheduler.step(val_loss)

            # ── Early stopping ─────────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt  = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        print(sep)

        # ── Load best checkpoint ──────────────────────────────────────────────
        self.model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        logger.info(f"Best val loss: {best_val_loss:.4f}")

        # ── Evaluate on validation ────────────────────────────────────────────
        self._print_eval(val_ds, "Validation")
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
        Note: first seq_len-1 bars cannot be predicted (no full window).
        Returns array aligned to X index; leading bars filled with 0 (HOLD).
        """
        assert self.model is not None, "Call .train() or .load() first."
        thresh = threshold if threshold is not None else self.threshold

        proba  = self.predict_proba(X)           # shape: (n_valid, 3)
        raw    = np.argmax(proba, axis=1)
        conf   = proba.max(axis=1)
        raw[conf < thresh] = 1                   # low confidence → HOLD

        signals_valid = np.vectorize(LABEL_MAP_INV.get)(raw)

        # Pad leading bars with HOLD (no window available)
        pad     = np.zeros(self.seq_len - 1, dtype=int)
        signals = np.concatenate([pad, signals_valid])
        return signals

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n_valid_samples, 3) softmax probabilities."""
        assert self.model is not None and self.scaler is not None
        X_sc = self.scaler.transform(X.values).astype(np.float32)
        ds   = SequenceDataset(X_sc, np.zeros(len(X_sc), dtype=int), self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size,
                            shuffle=False, num_workers=0)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for Xb, _ in loader:
                logits = self.model(Xb.to(DEVICE))
                probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
        return np.array(probs)

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self, X: pd.DataFrame, y: pd.Series, label: str = "Test"):
        assert self.scaler is not None
        X_sc = self.scaler.transform(X.values).astype(np.float32)
        y_enc = y.map(LABEL_MAP).values.astype(int)
        ds    = SequenceDataset(X_sc, y_enc, self.seq_len)
        self._print_eval(ds, label)

    def _print_eval(self, ds: SequenceDataset, label: str):
        loader = DataLoader(ds, batch_size=self.batch_size,
                            shuffle=False, num_workers=0)
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                logits = self.model(Xb.to(DEVICE))
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                labels.extend(yb.numpy())

        sep = "─" * 54
        print(f"\n{'═'*54}")
        print(f"  LSTM Evaluation — {label}")
        print(f"{'═'*54}")
        print(classification_report(
            labels, preds,
            target_names=["SHORT", "HOLD", "LONG"],
            zero_division=0,
        ))
        cm = confusion_matrix(labels, preds)
        print(f"  Confusion Matrix:")
        print(sep)
        print(f"  {'':14} {'Pred SHORT':>12} {'Pred HOLD':>12} {'Pred LONG':>12}")
        for i, row_label in enumerate(["Act SHORT", "Act HOLD", "Act LONG"]):
            print(f"  {row_label:<14} {cm[i,0]:>12,} {cm[i,1]:>12,} {cm[i,2]:>12,}")
        print(sep)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        print(f"  Weighted F1: {f1:.4f}")
        print(f"{'═'*54}\n")

    # ─────────────────────────────────────────────────────────────────────────
    def save(self):
        out_dir = cfg.MODEL_DIR / "lstm"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(out_dir / "meta.pkl", "wb") as f:
            pickle.dump({
                "feature_cols": self.feature_cols,
                "seq_len":      self.seq_len,
                "hidden_size":  self.hidden_size,
                "num_layers":   self.num_layers,
                "dropout":      self.dropout,
                "batch_size":   self.batch_size,
                "threshold":    self.threshold,
                "input_size":   self.input_size,
            }, f)
        logger.info(f"LSTM model saved → {out_dir}")

    def load(self) -> "LSTMTradingModel":
        out_dir = cfg.MODEL_DIR / "lstm"
        if not (out_dir / "model.pt").exists():
            raise FileNotFoundError(f"No LSTM model at {out_dir}. Train first.")
        with open(out_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]
        self.seq_len      = meta["seq_len"]
        self.hidden_size  = meta["hidden_size"]
        self.num_layers   = meta["num_layers"]
        self.dropout      = meta["dropout"]
        self.batch_size   = meta["batch_size"]
        self.threshold    = meta["threshold"]
        self.input_size   = meta["input_size"]

        self.model = LSTMClassifier(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout,
        ).to(DEVICE)
        self.model.load_state_dict(
            torch.load(out_dir / "model.pt", map_location=DEVICE)
        )
        self.model.eval()
        with open(out_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"LSTM model loaded ← {out_dir}")
        return self