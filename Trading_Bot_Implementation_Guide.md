# Trading-Bot: Production-Grade Implementation Guide

**Target:** CryptoTradingBot (BTC/USD) → Multi-Asset System  
**Stack:** Python 3.11 | Windows | i5-12400F | RTX 3060 8GB | 16GB RAM  
**Last Updated:** 2025

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Phase 1 — Data Engineering](#2-phase-1--data-engineering)
3. [Phase 2 — Feature Engineering](#3-phase-2--feature-engineering)
4. [Phase 3 — Strategy Design (Rule-Based)](#4-phase-3--strategy-design-rule-based)
5. [Phase 4 — Machine Learning Models (XGBoost + LSTM)](#5-phase-4--machine-learning-models-xgboost--lstm)
6. [Phase 5 — Reinforcement Learning (PPO)](#6-phase-5--reinforcement-learning-ppo)
7. [Phase 6 — Backtesting Engine](#7-phase-6--backtesting-engine)
8. [Phase 7 — Evaluation & Optimization](#8-phase-7--evaluation--optimization)
9. [Phase 8 — Experiment Tracking & Logging](#9-phase-8--experiment-tracking--logging)
10. [Phase 9 — Scaling to Multi-Asset](#10-phase-9--scaling-to-multi-asset)
11. [Code Architecture Reference](#11-code-architecture-reference)
12. [Best Practices Checklist](#12-best-practices-checklist)

---

## 1. Environment Setup

### 1.1 Python Environment

```bash
# Create and activate venv
python -m venv .venv
.venv\Scripts\activate

# Pin Python version
python --version  # Target: 3.11.x
```

### 1.2 `requirements.txt` (crypto module)

```text
# Data
pandas==2.2.2
numpy==1.26.4
pyarrow==15.0.2
requests==2.31.0
ccxt==4.3.x            # Crypto exchange API (Binance, Bybit, etc.)

# Technical Analysis
ta==0.11.0             # pip install ta
pandas-ta==0.3.14b     # Extended indicators
scipy==1.13.0

# ML
xgboost==2.0.3
scikit-learn==1.5.0
optuna==3.6.1          # Hyperparameter optimization
shap==0.45.0           # Feature importance

# Deep Learning / LSTM
torch==2.3.0+cu121     # CUDA 12.1 for RTX 3060
torchvision==0.18.0+cu121

# Reinforcement Learning
stable-baselines3==2.3.2
gymnasium==0.29.1

# Backtesting
vectorbt==0.26.2        # Fast, vectorized backtesting
# OR: backtrader==1.9.78 (event-driven, heavier)

# Experiment Tracking
mlflow==2.13.0

# Visualization
matplotlib==3.9.0
plotly==5.22.0

# Utilities
python-dotenv==1.0.1
pydantic==2.7.1        # Config validation
loguru==0.7.2          # Structured logging
pytest==8.2.0
```

### 1.3 GPU Verification

```python
# crypto/src/utils/gpu_check.py
import torch

def verify_gpu():
    assert torch.cuda.is_available(), "CUDA not found. Check driver + torch+cu121 install."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # RTX 3060 has 8GB VRAM — keep batch sizes moderate (256–1024)

verify_gpu()
```

### 1.4 Configuration Layer

```python
# crypto/config/settings.py
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"

    # Data
    SYMBOL: str = "BTC/USDT"
    EXCHANGE: str = "binance"
    PRIMARY_TF: str = "1h"          # Primary timeframe
    HIGHER_TF: str = "1d"           # Macro context
    START_DATE: str = "2020-01-01"

    # Trading
    INITIAL_CAPITAL: float = 10_000.0
    COMMISSION: float = 0.001       # 0.1% per trade (Binance taker)
    SLIPPAGE: float = 0.0005        # 0.05% slippage estimate
    MAX_POSITION_SIZE: float = 0.95  # Max 95% of capital per position
    RISK_PER_TRADE: float = 0.02    # 2% portfolio risk per trade

    # Reproducibility
    RANDOM_SEED: int = 42

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 2. Phase 1 — Data Engineering

**Objective:** Build a clean, timestamped, multi-timeframe OHLCV dataset stored in an efficient format.

### 2.1 Module: `src/data/fetcher.py`

```python
import ccxt
import pandas as pd
from pathlib import Path
from crypto.config.settings import settings
from loguru import logger

class OHLCVFetcher:
    """Fetches OHLCV data from exchange via CCXT. Rate-limit safe."""

    def __init__(self, exchange_id: str = settings.EXCHANGE):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch(
        self,
        symbol: str = settings.SYMBOL,
        timeframe: str = settings.PRIMARY_TF,
        since: str = settings.START_DATE,
        save: bool = True,
    ) -> pd.DataFrame:
        since_ms = self.exchange.parse8601(f"{since}T00:00:00Z")
        all_ohlcv = []
        logger.info(f"Fetching {symbol} [{timeframe}] from {since}...")

        while True:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000
            )
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since_ms = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]  # Remove duplicates
        df.sort_index(inplace=True)

        if save:
            self._save(df, symbol, timeframe)

        return df

    def _save(self, df: pd.DataFrame, symbol: str, timeframe: str):
        safe_symbol = symbol.replace("/", "_")
        path = settings.DATA_DIR / "raw" / f"{safe_symbol}_{timeframe}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {len(df)} rows → {path}")
```

**Why Parquet?**  Columnar storage, compressed (~5x vs CSV), preserves dtypes, fast pandas/PyArrow reads. Use `snappy` compression for best read speed. Reserve CSV for human-readable exports only.

### 2.2 Module: `src/data/loader.py`

```python
import pandas as pd
from crypto.config.settings import settings

class DataLoader:
    """Loads, validates, and optionally resamples persisted data."""

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        safe_symbol = symbol.replace("/", "_")
        path = settings.DATA_DIR / "raw" / f"{safe_symbol}_{timeframe}.parquet"
        df = pd.read_parquet(path, engine="pyarrow")
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame):
        required = ["open", "high", "low", "close", "volume"]
        assert all(c in df.columns for c in required), "Missing OHLCV columns"
        assert df.index.is_monotonic_increasing, "Index not sorted"
        assert not df.isnull().any().any(), f"NaN values found: {df.isnull().sum()}"
        assert (df["high"] >= df["low"]).all(), "High < Low detected"

    def resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Downsample to higher timeframe. E.g., '4h', '1d'."""
        return df.resample(rule).agg({
            "open": "first", "high": "max",
            "low": "min",    "close": "last",
            "volume": "sum"
        }).dropna()
```

### 2.3 Data Directory Structure

```
data/
├── raw/
│   ├── BTC_USDT_1h.parquet
│   └── BTC_USDT_1d.parquet
├── processed/
│   └── BTC_USDT_1h_features.parquet
└── splits/
    ├── train.parquet
    ├── val.parquet
    └── test.parquet
```

### 2.4 Expected Outputs

| Output | Description |
|---|---|
| `raw/BTC_USDT_1h.parquet` | ~35,000 rows of raw OHLCV (2020–present) |
| `raw/BTC_USDT_1d.parquet` | ~1,800 rows for macro context |
| Validation logs | Zero NaNs, sorted index, OHLCV integrity |

---

## 3. Phase 2 — Feature Engineering

**Objective:** Compute technical indicators as ML-ready features, avoiding lookahead bias.

> **Golden Rule of Lookahead Prevention:** Every feature at time `t` must be computable using only data up to and including bar `t`. Never shift a feature backward. Validate with `assert (df.feature.shift(1) != df.feature).any()` type checks.

### 3.1 Module: `src/features/technical.py`

```python
import pandas as pd
import pandas_ta as pta
import numpy as np

class TechnicalFeatures:
    """
    Computes indicators strictly in-order.
    All values at index t use only [0..t] data.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- Trend ---
        out["ema_9"]   = pta.ema(df["close"], length=9)
        out["ema_21"]  = pta.ema(df["close"], length=21)
        out["ema_50"]  = pta.ema(df["close"], length=50)
        out["ema_200"] = pta.ema(df["close"], length=200)

        # EMA crossover signal (current bar only, no future)
        out["ema_cross_9_21"] = (out["ema_9"] > out["ema_21"]).astype(int)

        # --- Momentum ---
        macd = pta.macd(df["close"], fast=12, slow=26, signal=9)
        out["macd"]        = macd["MACD_12_26_9"]
        out["macd_signal"] = macd["MACDs_12_26_9"]
        out["macd_hist"]   = macd["MACDh_12_26_9"]

        out["rsi_14"] = pta.rsi(df["close"], length=14)
        out["rsi_7"]  = pta.rsi(df["close"], length=7)
        stoch = pta.stoch(df["high"], df["low"], df["close"])
        out["stoch_k"] = stoch["STOCHk_14_3_3"]
        out["stoch_d"] = stoch["STOCHd_14_3_3"]

        # --- Volatility ---
        bb = pta.bbands(df["close"], length=20, std=2)
        out["bb_upper"]  = bb["BBU_20_2.0"]
        out["bb_mid"]    = bb["BBM_20_2.0"]
        out["bb_lower"]  = bb["BBL_20_2.0"]
        out["bb_width"]  = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
        out["bb_pct"]    = (df["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

        out["atr_14"] = pta.atr(df["high"], df["low"], df["close"], length=14)
        out["atr_pct"] = out["atr_14"] / df["close"]  # Normalized ATR

        # --- Volume ---
        out["volume_sma_20"] = df["volume"].rolling(20).mean()
        out["volume_ratio"]  = df["volume"] / out["volume_sma_20"]
        out["obv"]           = pta.obv(df["close"], df["volume"])
        out["vwap"]          = pta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # --- Price-Derived ---
        out["log_return"]      = np.log(df["close"] / df["close"].shift(1))
        out["return_5"]        = df["close"].pct_change(5)
        out["return_24"]       = df["close"].pct_change(24)
        out["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
        out["close_pct_range"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)

        # --- Higher-Timeframe Context (merged in pipeline, not here) ---
        # See FeaturePipeline for multi-TF merge

        return out.dropna()  # Drop NaN rows from indicator warmup periods
```

### 3.2 Module: `src/features/patterns.py`

```python
import pandas as pd
import pandas_ta as pta

class ChartPatternFeatures:
    """Candlestick pattern detection using pandas_ta CDL module."""

    PATTERNS = [
        "cdl_doji", "cdl_hammer", "cdl_engulfing",
        "cdl_shootingstar", "cdl_morningstar", "cdl_eveningstar",
        "cdl_harami", "cdl_marubozu"
    ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for pattern in self.PATTERNS:
            try:
                result = getattr(pta, pattern)(
                    df["open"], df["high"], df["low"], df["close"]
                )
                if result is not None:
                    col_name = pattern.replace("cdl_", "pat_")
                    out[col_name] = result.fillna(0).astype(int)
            except Exception:
                pass
        return out
```

### 3.3 Module: `src/features/target.py`

**This is the most critical file for preventing lookahead bias.**

```python
import pandas as pd
import numpy as np
from enum import IntEnum

class SignalType(IntEnum):
    SHORT  = -1
    HOLD   =  0
    LONG   =  1

class TargetBuilder:
    """
    Creates forward-looking labels for supervised learning.
    Labels at time t are computed using future bars [t+1 .. t+horizon].
    NEVER use these labels as input features — only as targets.
    """

    def build_classification_target(
        self,
        df: pd.DataFrame,
        horizon: int = 12,      # 12 bars ahead (12h for 1h data)
        threshold: float = 0.015 # 1.5% move = signal
    ) -> pd.Series:
        """
        Returns:
            +1 (LONG)  if price rises >threshold in next `horizon` bars
            -1 (SHORT) if price falls >threshold
             0 (HOLD)  otherwise
        """
        future_return = df["close"].shift(-horizon) / df["close"] - 1

        conditions = [
            future_return > threshold,
            future_return < -threshold
        ]
        choices = [SignalType.LONG, SignalType.SHORT]
        target = pd.Series(
            np.select(conditions, choices, default=SignalType.HOLD),
            index=df.index,
            name=f"target_{horizon}h"
        )
        # Drop the last `horizon` rows — they have no valid future
        target.iloc[-horizon:] = np.nan
        return target

    def build_regression_target(
        self, df: pd.DataFrame, horizon: int = 12
    ) -> pd.Series:
        """Continuous log-return over horizon. For regression models."""
        return np.log(df["close"].shift(-horizon) / df["close"]).rename(f"log_ret_{horizon}h")
```

### 3.4 Feature Pipeline

```python
# src/features/pipeline.py
import pandas as pd
from crypto.src.features.technical import TechnicalFeatures
from crypto.src.features.patterns import ChartPatternFeatures
from crypto.src.features.target import TargetBuilder
from crypto.src.data.loader import DataLoader
from crypto.config.settings import settings

class FeaturePipeline:
    """
    Orchestrates full feature construction for one symbol/timeframe.
    Merges multi-timeframe context and saves processed dataset.
    """

    def __init__(self):
        self.loader   = DataLoader()
        self.tech     = TechnicalFeatures()
        self.patterns = ChartPatternFeatures()
        self.targets  = TargetBuilder()

    def run(
        self,
        symbol: str = settings.SYMBOL,
        primary_tf: str = settings.PRIMARY_TF,
        higher_tf: str = settings.HIGHER_TF,
    ) -> pd.DataFrame:
        # Load both timeframes
        df_primary = self.loader.load(symbol, primary_tf)
        df_daily   = self.loader.load(symbol, higher_tf)

        # Compute features on primary TF
        df = self.tech.compute(df_primary)
        df = self.patterns.compute(df)

        # Merge daily macro context (forward-fill to 1h bars)
        daily_feats = self.tech.compute(df_daily)[
            ["ema_50", "ema_200", "rsi_14", "atr_pct", "bb_pct"]
        ].add_prefix("d1_")
        df = df.join(daily_feats.resample(primary_tf).ffill(), how="left")

        # Build target
        df["target"] = self.targets.build_classification_target(df)

        # Drop warmup and future NaNs
        df.dropna(inplace=True)

        # Save
        safe = symbol.replace("/", "_")
        out_path = settings.DATA_DIR / "processed" / f"{safe}_{primary_tf}_features.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")

        return df
```

### 3.5 Time-Series Train/Val/Test Split

```python
# src/data/splitter.py
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    # test_ratio implied: 0.15
) -> DataSplit:
    """
    Strict chronological split. No shuffling, no leakage.
    Gap between splits is NOT introduced here but should be
    considered if using walk-forward validation.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    split = DataSplit(
        train = df.iloc[:train_end].copy(),
        val   = df.iloc[train_end:val_end].copy(),
        test  = df.iloc[val_end:].copy(),
    )
    print(f"Train: {split.train.index[0]} → {split.train.index[-1]} ({len(split.train)} rows)")
    print(f"Val:   {split.val.index[0]}   → {split.val.index[-1]} ({len(split.val)} rows)")
    print(f"Test:  {split.test.index[0]}  → {split.test.index[-1]} ({len(split.test)} rows)")
    return split
```

> **Walk-Forward Validation (WFV):** For production-quality evaluation, use expanding or sliding window retraining instead of a single split. See Phase 7 for WFV implementation.

---

## 4. Phase 3 — Strategy Design (Rule-Based)

**Objective:** Define deterministic trading logic that generates buy/sell signals without any ML.

### 4.1 Base Strategy Interface

```python
# src/strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Series of {-1 (short), 0 (hold), 1 (long)}
        indexed identically to df.
        No future data may be used. Signal at t uses df.iloc[:t+1] only.
        """

    def _validate_signals(self, signals: pd.Series, df: pd.DataFrame):
        assert signals.index.equals(df.index), "Signal index mismatch"
        assert signals.isin([-1, 0, 1]).all(), "Invalid signal values"
```

### 4.2 Example: EMA Crossover + RSI Filter

```python
# src/strategies/ema_rsi.py
import pandas as pd
from crypto.src.strategies.base import BaseStrategy

class EMACrossoverRSI(BaseStrategy):
    """
    Long  when: EMA9 crosses above EMA21 AND RSI14 > 50
    Short when: EMA9 crosses below EMA21 AND RSI14 < 50
    Exit  when: Opposite cross OR RSI extreme reversal
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema_cross = (df["ema_9"] > df["ema_21"]).astype(int)
        prev_cross = ema_cross.shift(1)

        cross_up   = (ema_cross == 1) & (prev_cross == 0)
        cross_down = (ema_cross == 0) & (prev_cross == 1)

        signals = pd.Series(0, index=df.index, name="signal")
        signals[cross_up   & (df["rsi_14"] > 50)] =  1
        signals[cross_down & (df["rsi_14"] < 50)] = -1

        self._validate_signals(signals, df)
        return signals
```

### 4.3 Signal → Position Mapping

```python
# src/strategies/position_manager.py
import pandas as pd

def signals_to_positions(signals: pd.Series, mode: str = "long_only") -> pd.Series:
    """
    Converts raw entry signals to positions held over time.
    mode: "long_only" | "long_short"
    Position = 1 (long), -1 (short), 0 (flat)
    """
    positions = signals.copy()

    if mode == "long_only":
        # Ignore short signals, hold long until opposite signal
        pos = 0
        result = []
        for s in signals:
            if s == 1:
                pos = 1
            elif s == -1:
                pos = 0   # Exit long, no short
            result.append(pos)
        positions = pd.Series(result, index=signals.index)

    elif mode == "long_short":
        # Full long/short, hold until opposite
        positions = signals.replace(0, method="ffill").fillna(0)

    return positions
```

---

## 5. Phase 4 — Machine Learning Models (XGBoost + LSTM)

### 5.1 Feature Matrix Construction

```python
# src/models/feature_selector.py
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import shap

FEATURE_COLS = [
    "ema_cross_9_21", "macd_hist", "rsi_14", "rsi_7",
    "stoch_k", "stoch_d", "bb_width", "bb_pct",
    "atr_pct", "volume_ratio", "log_return",
    "return_5", "return_24", "high_low_spread", "close_pct_range",
    "d1_ema_50", "d1_rsi_14", "d1_atr_pct",
    # Pattern features
    "pat_doji", "pat_hammer", "pat_engulfing", "pat_shootingstar",
]

def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df["target"].copy()
    # Ensure no NaN (should be clean after pipeline)
    assert not X.isnull().any().any()
    assert not y.isnull().any()
    return X, y
```

### 5.2 XGBoost Classifier

```python
# src/models/xgboost_model.py
import xgboost as xgb
import optuna
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
from crypto.config.settings import settings

class XGBoostTradingModel:
    """
    Multi-class classifier: LONG / HOLD / SHORT
    Uses GPU acceleration (tree_method='hist', device='cuda')
    """

    def __init__(self):
        self.model = None
        self.le = LabelEncoder()

    def train(self, X_train, y_train, X_val, y_val, n_trials: int = 50):
        y_train_enc = self.le.fit_transform(y_train)
        y_val_enc   = self.le.transform(y_val)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                # GPU settings
                "tree_method": "hist",
                "device":      "cuda",
                "objective":   "multi:softmax",
                "num_class":   3,
                "eval_metric": "mlogloss",
                "seed":        settings.RANDOM_SEED,
            }
            clf = xgb.XGBClassifier(**params, early_stopping_rounds=30, verbosity=0)
            clf.fit(
                X_train, y_train_enc,
                eval_set=[(X_val, y_val_enc)],
                verbose=False
            )
            preds = clf.predict(X_val)
            # Penalize overtrading: weight F1 of LONG/SHORT classes
            report = classification_report(y_val_enc, preds, output_dict=True, zero_division=0)
            return report["weighted avg"]["f1-score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            "tree_method": "hist", "device": "cuda",
            "objective": "multi:softmax", "num_class": 3,
            "eval_metric": "mlogloss", "seed": settings.RANDOM_SEED
        })

        self.model = xgb.XGBClassifier(**best_params, early_stopping_rounds=30)
        self.model.fit(
            X_train, y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            verbose=100
        )

        with mlflow.start_run(run_name="xgboost_opt"):
            mlflow.log_params(best_params)
            mlflow.log_metric("val_f1", study.best_value)
            mlflow.xgboost.log_model(self.model, "xgboost_model")

        return self.model

    def predict(self, X) -> np.ndarray:
        encoded = self.model.predict(X)
        return self.le.inverse_transform(encoded)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
```

### 5.3 LSTM Model (PyTorch)

```python
# src/models/lstm_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from crypto.config.settings import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    """Sliding window dataset for LSTM. Window at t uses [t-seq_len+1 .. t]."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],           # (seq_len, n_features)
            self.y[idx + self.seq_len]                   # scalar label
        )

class LSTMTradingModel(nn.Module):
    """Stacked LSTM → FC classifier with dropout regularization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Last timestep
        return self.fc(out)

def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seq_len: int = 60,
    epochs: int = 50,
    batch_size: int = 256,   # Fits well in RTX 3060 8GB for seq_len=60
    lr: float = 1e-3,
) -> LSTMTradingModel:

    # Normalize (fit on train only!)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val - mean) / std

    train_ds = TimeSeriesDataset(X_train_n, y_train, seq_len)
    val_ds   = TimeSeriesDataset(X_val_n,   y_val,   seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = LSTMTradingModel(input_size=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), settings.MODEL_DIR / "lstm_best.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load(settings.MODEL_DIR / "lstm_best.pt"))
    return model
```

### 5.4 Model Ensemble

```python
# src/models/ensemble.py
import numpy as np

class ModelEnsemble:
    """Weighted voting ensemble across XGBoost + LSTM probabilities."""

    def __init__(self, models: list, weights: list = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict_proba(self, X_static, X_seq) -> np.ndarray:
        """
        X_static: (n_samples, n_features) for XGBoost
        X_seq:    (n_samples, seq_len, n_features) for LSTM
        Returns:  (n_samples, 3) probability over [SHORT, HOLD, LONG]
        """
        probs = []
        for model, w in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):   # XGBoost
                probs.append(model.predict_proba(X_static) * w)
            else:                                  # LSTM
                import torch
                with torch.no_grad():
                    logits = model(torch.tensor(X_seq, dtype=torch.float32).to(DEVICE))
                    p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p * w)
        return np.sum(probs, axis=0)

    def predict(self, X_static, X_seq, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X_static, X_seq)
        signals = np.argmax(proba, axis=1) - 1  # {0,1,2} → {-1,0,1}
        # Confidence filter: only signal if max prob > threshold
        max_conf = proba.max(axis=1)
        signals[max_conf < threshold] = 0
        return signals
```

---

## 6. Phase 5 — Reinforcement Learning (PPO)

### 6.1 Trading Environment Design

```python
# src/models/rl/trading_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from crypto.config.settings import settings

class BTCTradingEnv(gym.Env):
    """
    Single-asset BTC/USD trading environment for Stable-Baselines3 PPO.

    State:  Concatenation of technical features for last `window` bars
            + portfolio state (position, unrealized PnL, cash ratio)
    Action: Discrete(3) → {0: SHORT/EXIT, 1: HOLD, 2: BUY/LONG}
    Reward: Log return of position over one step, penalized for excessive trading
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        window: int = 30,
        initial_capital: float = settings.INITIAL_CAPITAL,
        commission: float = settings.COMMISSION,
        slippage: float = settings.SLIPPAGE,
        max_position: float = settings.MAX_POSITION_SIZE,
        long_only: bool = True,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window = window
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position = max_position
        self.long_only = long_only
        self.n_features = len(feature_cols)

        # Action space: 0=exit/short, 1=hold, 2=buy/long
        self.action_space = spaces.Discrete(3)

        # Observation: (window, n_features) + 3 portfolio scalars
        obs_dim = window * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.cash = self.initial_capital
        self.position = 0.0    # BTC units held
        self.entry_price = 0.0
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        price = self.df.loc[self.current_step, "close"]
        exec_price = price * (1 + self.slippage)  # Slippage on buys

        prev_portfolio_value = self._portfolio_value(price)

        # Execute action
        if action == 2 and self.position == 0:   # BUY
            units = (self.cash * self.max_position) / exec_price
            cost = units * exec_price * (1 + self.commission)
            if cost <= self.cash:
                self.position = units
                self.cash -= cost
                self.entry_price = exec_price
                self.trades.append({"type": "buy", "price": exec_price, "step": self.current_step})

        elif action == 0 and self.position > 0:  # SELL/EXIT
            sell_price = price * (1 - self.slippage)
            proceeds = self.position * sell_price * (1 - self.commission)
            self.cash += proceeds
            self.trades.append({
                "type": "sell", "price": sell_price, "step": self.current_step,
                "pnl": proceeds - self.position * self.entry_price
            })
            self.position = 0.0

        # Advance
        self.current_step += 1
        new_price = self.df.loc[self.current_step, "close"]
        new_portfolio_value = self._portfolio_value(new_price)

        # Reward: Log return, penalize overtrading
        log_ret = np.log(new_portfolio_value / (prev_portfolio_value + 1e-9))
        trade_penalty = -0.001 if action != 1 else 0.0   # Penalize unnecessary action
        reward = log_ret + trade_penalty

        self.portfolio_values.append(new_portfolio_value)

        done = self.current_step >= len(self.df) - 1
        obs = self._get_obs()
        info = {"portfolio_value": new_portfolio_value, "position": self.position}

        return obs, reward, done, False, info

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.position * price

    def _get_obs(self) -> np.ndarray:
        # Feature window (normalized per-window to avoid distribution shift)
        window_data = self.df.loc[
            self.current_step - self.window : self.current_step - 1, self.feature_cols
        ].values.astype(np.float32)
        mean = window_data.mean(axis=0)
        std  = window_data.std(axis=0) + 1e-8
        norm_window = ((window_data - mean) / std).flatten()

        # Portfolio state
        price = self.df.loc[self.current_step, "close"]
        portfolio_val = self._portfolio_value(price)
        unrealized_pnl = (price - self.entry_price) / (self.entry_price + 1e-9) if self.position > 0 else 0.0
        cash_ratio = self.cash / (portfolio_val + 1e-9)
        portfolio_state = np.array([float(self.position > 0), unrealized_pnl, cash_ratio], dtype=np.float32)

        return np.concatenate([norm_window, portfolio_state])
```

### 6.2 PPO Training

```python
# src/models/rl/ppo_trainer.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecNormalize
import mlflow
from crypto.config.settings import settings
from crypto.src.models.rl.trading_env import BTCTradingEnv

def train_ppo(df_train, df_val, feature_cols, total_timesteps: int = 500_000):
    def make_env():
        return BTCTradingEnv(df_train, feature_cols)

    # Vectorized env (4 parallel) — CPU parallelism, not GPU
    vec_env = make_vec_env(make_env, n_envs=4)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(
        make_vec_env(lambda: BTCTradingEnv(df_val, feature_cols), n_envs=1),
        norm_obs=True, norm_reward=False, training=False
    )

    # PPO hyperparameters tuned for financial env
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=settings.RANDOM_SEED,
        device="cuda",         # Use RTX 3060 for policy network
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(settings.MODEL_DIR / "ppo_best"),
        log_path=str(settings.MODEL_DIR / "ppo_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    with mlflow.start_run(run_name="ppo_training"):
        model.learn(total_timesteps=total_timesteps, callback=eval_cb)
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("n_envs", 4)

    vec_env.save(str(settings.MODEL_DIR / "vec_normalize.pkl"))
    return model
```

### 6.3 RL State / Action / Reward Summary

| Component | Definition | Notes |
|---|---|---|
| **State** | Last 30 bars × N features + (position, unrealized PnL, cash ratio) | Normalized per-window to handle non-stationarity |
| **Action** | Discrete(3): Exit=0, Hold=1, Buy=2 | Long-only mode; extend for short |
| **Reward** | Log portfolio return per step − trade penalty | Log return ensures symmetry; trade penalty reduces churning |
| **Episode** | Full backtest period (reset at start) | One episode = one pass through the dataset |
| **Termination** | End of data, or portfolio value < 10% of initial | Hard stop prevents catastrophic loss during training |

---

## 7. Phase 6 — Backtesting Engine

### 7.1 Design Principles

**Avoid these common errors:**
1. Using close price to trigger and execute in the same bar → Use next bar's open for execution
2. Ignoring transaction costs and slippage
3. Fitting strategy parameters on test set
4. Survivorship bias (not applicable for single-asset, but relevant for multi-asset)

### 7.2 Vectorized Backtester (VectorBT)

```python
# src/backtesting/vectorbt_backtest.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from crypto.config.settings import settings

class VectorBTBacktester:
    """
    High-performance vectorized backtesting using VectorBT.
    Simulates realistic execution: signals on bar t, execute on bar t+1 open.
    """

    def __init__(self):
        self.commission = settings.COMMISSION
        self.slippage   = settings.SLIPPAGE

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,    # {-1, 0, 1} indexed like df
        long_only: bool = True,
    ) -> vbt.Portfolio:
        # Execute on NEXT BAR's open to prevent lookahead
        exec_open = df["open"].shift(-1)

        entries = signals == 1
        exits   = signals == -1

        if long_only:
            portfolio = vbt.Portfolio.from_signals(
                close   = exec_open,
                entries = entries.shift(1).fillna(False),  # Signal on t, execute on t+1
                exits   = exits.shift(1).fillna(False),
                init_cash  = settings.INITIAL_CAPITAL,
                fees       = self.commission,
                slippage   = self.slippage,
                freq       = "1h",
            )
        else:
            short_entries = signals == -1
            short_exits   = signals == 1
            portfolio = vbt.Portfolio.from_signals(
                close         = exec_open,
                entries       = entries.shift(1).fillna(False),
                exits         = exits.shift(1).fillna(False),
                short_entries = short_entries.shift(1).fillna(False),
                short_exits   = short_exits.shift(1).fillna(False),
                init_cash     = settings.INITIAL_CAPITAL,
                fees          = self.commission,
                slippage      = self.slippage,
                freq          = "1h",
            )

        return portfolio

    def get_metrics(self, portfolio: vbt.Portfolio) -> dict:
        stats = portfolio.stats()
        return {
            "total_return_pct":    stats["Total Return [%]"],
            "annualized_return":   stats["Annualized Return [%]"],
            "sharpe_ratio":        stats["Sharpe Ratio"],
            "sortino_ratio":       stats["Sortino Ratio"],
            "calmar_ratio":        stats["Calmar Ratio"],
            "max_drawdown_pct":    stats["Max Drawdown [%]"],
            "win_rate_pct":        stats["Win Rate [%]"],
            "profit_factor":       stats["Profit Factor"],
            "total_trades":        stats["Total Trades"],
            "avg_trade_duration":  stats["Avg Winning Trade Duration"],
            "expectancy":          stats["Expectancy"],
        }
```

### 7.3 Walk-Forward Validation

```python
# src/backtesting/walk_forward.py
import pandas as pd
import numpy as np
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class WFVResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict

def walk_forward_validation(
    df: pd.DataFrame,
    strategy_fn: Callable,       # fn(train_df) → trained strategy/model
    signal_fn: Callable,         # fn(model, test_df) → signals Series
    backtester,
    train_window: int = 8760,    # 1 year of hourly bars
    test_window: int  = 2190,    # 3 months of hourly bars
    step: int         = 2190,    # Roll forward by test_window
) -> List[WFVResult]:
    """
    Expanding or rolling WFV.
    Each fold: train on [0..train_end], test on [train_end..test_end].
    """
    results = []
    start = 0

    while start + train_window + test_window <= len(df):
        train_df = df.iloc[start : start + train_window]
        test_df  = df.iloc[start + train_window : start + train_window + test_window]

        model   = strategy_fn(train_df)
        signals = signal_fn(model, test_df)
        port    = backtester.run(test_df, signals)
        metrics = backtester.get_metrics(port)

        results.append(WFVResult(
            fold        = len(results) + 1,
            train_start = train_df.index[0],
            train_end   = train_df.index[-1],
            test_start  = test_df.index[0],
            test_end    = test_df.index[-1],
            metrics     = metrics,
        ))

        start += step  # Roll forward

    return results
```

### 7.4 Evaluation Metrics Reference

| Metric | Formula / Source | Target |
|---|---|---|
| **Sharpe Ratio** | (R_p − R_f) / σ_p, annualized | > 1.5 |
| **Sortino Ratio** | (R_p − R_f) / downside σ | > 2.0 |
| **Calmar Ratio** | Annualized Return / Max Drawdown | > 1.0 |
| **Max Drawdown** | max peak-to-trough decline | < 25% |
| **Win Rate** | Winning trades / Total trades | > 50% |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.5 |
| **Expectancy** | Win Rate × Avg Win − Loss Rate × Avg Loss | > 0 |
| **Total Trades** | Count over period | Avoid < 30 (statistical insignificance) |

---

## 8. Phase 7 — Evaluation & Optimization

### 8.1 Overfitting Detection

```python
# src/backtesting/overfitting_check.py
import pandas as pd
import numpy as np

def check_overfitting(train_metrics: dict, test_metrics: dict, threshold: float = 0.5):
    """
    Flags overfitting if test Sharpe is less than threshold × train Sharpe.
    Also compares win rate and profit factor degradation.
    """
    checks = {}

    for metric in ["sharpe_ratio", "win_rate_pct", "profit_factor"]:
        train_val = train_metrics.get(metric, 0)
        test_val  = test_metrics.get(metric, 0)
        ratio     = test_val / (train_val + 1e-9)
        is_ok     = ratio >= threshold
        checks[metric] = {"train": train_val, "test": test_val, "ratio": ratio, "ok": is_ok}
        if not is_ok:
            print(f"WARNING: {metric} degraded by {(1-ratio)*100:.1f}% (train={train_val:.2f}, test={test_val:.2f})")

    return checks
```

### 8.2 Monte Carlo Simulation

```python
def monte_carlo_returns(daily_returns: pd.Series, n_simulations: int = 1000, n_days: int = 365):
    """
    Bootstrap resampling of trade returns to estimate distribution of outcomes.
    Gives confidence intervals on equity curve.
    """
    np.random.seed(settings.RANDOM_SEED)
    results = []
    for _ in range(n_simulations):
        sampled = np.random.choice(daily_returns, size=n_days, replace=True)
        cumulative = (1 + sampled).cumprod()
        results.append(cumulative[-1])  # Final equity multiplier

    results = np.array(results)
    print(f"Median final equity: {np.median(results):.2f}x")
    print(f"5th percentile:      {np.percentile(results, 5):.2f}x")
    print(f"95th percentile:     {np.percentile(results, 95):.2f}x")
    print(f"Probability of loss: {(results < 1.0).mean()*100:.1f}%")
    return results
```

---

## 9. Phase 8 — Experiment Tracking & Logging

### 9.1 MLflow Setup

```python
# Run in terminal before experiments:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# crypto/config/mlflow_config.py
import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENTS = {
    "xgboost":  "crypto_xgboost",
    "lstm":     "crypto_lstm",
    "ppo":      "crypto_ppo",
    "ensemble": "crypto_ensemble",
}

def get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id
```

### 9.2 Structured Logging

```python
# src/utils/logger.py
from loguru import logger
import sys
from pathlib import Path
from crypto.config.settings import settings

def setup_logger(name: str = "trading_bot"):
    log_path = settings.BASE_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)

    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(
        str(log_path),
        rotation="50 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
    )
    return logger
```

### 9.3 Reproducibility Contract

Every experiment entry in MLflow must log:

```python
with mlflow.start_run():
    mlflow.log_params({
        "random_seed":  settings.RANDOM_SEED,
        "symbol":       settings.SYMBOL,
        "timeframe":    settings.PRIMARY_TF,
        "train_start":  str(split.train.index[0]),
        "train_end":    str(split.train.index[-1]),
        "test_start":   str(split.test.index[0]),
        "test_end":     str(split.test.index[-1]),
        "features":     sorted(FEATURE_COLS),
        "model_type":   "xgboost",
        "git_commit":   get_git_commit(),  # subprocess.check_output(["git", "rev-parse", "HEAD"])
    })
    mlflow.log_artifact("crypto/config/settings.py")
```

---

## 10. Phase 9 — Scaling to Multi-Asset

### 10.1 Shared Architecture Pattern

The key insight: crypto, forex, and commodities share identical pipeline stages. Only data sources and symbol-specific parameters differ.

```
Trading-Bot/
├── shared/                     ← NEW: cross-asset shared code
│   ├── base_fetcher.py         ← Abstract OHLCV fetcher interface
│   ├── base_features.py        ← Technical indicators (symbol-agnostic)
│   ├── base_strategy.py        ← Strategy interface
│   ├── base_backtest.py        ← Backtest runner
│   └── base_env.py             ← RL Gym environment base class
├── crypto/
│   ├── config/
│   │   └── settings.py         ← SYMBOL="BTC/USDT", EXCHANGE="binance"
│   ├── src/
│   │   ├── data/fetcher.py     ← CCXT-based fetcher (inherits base_fetcher)
│   │   └── ...
├── forex/
│   ├── config/
│   │   └── settings.py         ← SYMBOL="AUDUSD", EXCHANGE="oanda"
│   ├── src/
│   │   ├── data/fetcher.py     ← OANDA API fetcher (inherits base_fetcher)
│   │   └── ...
└── commodities/
    ├── config/
    │   └── settings.py         ← SYMBOL="XAUUSD", EXCHANGE="fxcm"
    └── src/
        └── data/fetcher.py     ← MT5/yfinance fetcher (inherits base_fetcher)
```

### 10.2 Abstract Base Fetcher

```python
# shared/base_fetcher.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseFetcher(ABC):
    """All asset fetchers must implement this contract."""

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, since: str) -> pd.DataFrame:
        """Return UTC-indexed OHLCV DataFrame with columns: open, high, low, close, volume"""

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common post-processing: sort, deduplicate, validate dtypes."""
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df
```

### 10.3 Asset-Specific Considerations

| Asset | Data Source | Market Hours | Commission Model | Key Difference |
|---|---|---|---|---|
| **BTC/USD** | CCXT (Binance) | 24/7 | 0.1% taker | High volatility, no gaps |
| **AUD/USD** | OANDA API | 24/5 (Mon–Fri) | Spread-based (~1 pip) | Weekend gaps, pip sizing |
| **XAU/USD** | yFinance / MT5 | 23/5 | Spread (~0.3 USD) | Safe-haven correlation, session-driven |

### 10.4 Multi-Asset Portfolio Manager (Future)

```python
# shared/portfolio_manager.py
class MultiAssetPortfolio:
    """
    Aggregates signals from all asset modules.
    Allocates capital using risk-parity or fixed fractional sizing.
    """

    def __init__(self, assets: list, total_capital: float):
        self.assets = assets           # ["crypto", "forex", "commodities"]
        self.total_capital = total_capital
        self.allocations = {a: 1/len(assets) for a in assets}  # Equal weight start

    def risk_parity_weights(self, volatilities: dict) -> dict:
        """Inverse-volatility weighting."""
        inv_vol = {k: 1/v for k, v in volatilities.items()}
        total = sum(inv_vol.values())
        return {k: v/total for k, v in inv_vol.items()}
```

---

## 11. Code Architecture Reference

### 11.1 Module Dependency Graph

```
main.py
  └── FeaturePipeline
        ├── OHLCVFetcher  (data source)
        ├── DataLoader    (persistence)
        ├── TechnicalFeatures
        ├── ChartPatternFeatures
        └── TargetBuilder
              │
              ▼
        DataSplitter  →  [train / val / test]
              │
     ┌────────┼────────────┐
     ▼        ▼            ▼
  XGBoost    LSTM         PPO
  Model      Model        Trainer
     │        │            │
     └────────┴────────────┘
              │
        ModelEnsemble
              │
        VectorBTBacktester
              │
        MetricsEvaluator
              │
           MLflow
```

### 11.2 Module Communication Rules

1. **Modules never import from sibling modules directly.** Use the pipeline orchestrator (`main.py` or `FeaturePipeline`) as the coordinator.
2. **All data passed between modules as DataFrames or numpy arrays.** No shared mutable state.
3. **Config is the single source of truth.** No hardcoded constants outside `settings.py`.
4. **Models are serialized to disk, not passed in memory across runs.** Enables reproducibility.

### 11.3 `main.py` Entry Point

```python
# crypto/main.py
import argparse
from crypto.config.settings import settings
from crypto.src.data.fetcher import OHLCVFetcher
from crypto.src.features.pipeline import FeaturePipeline
from crypto.src.data.splitter import temporal_split
from crypto.src.models.xgboost_model import XGBoostTradingModel
from crypto.src.backtesting.vectorbt_backtest import VectorBTBacktester
from crypto.src.utils.logger import setup_logger

logger = setup_logger()

def run_pipeline(mode: str = "full"):
    logger.info(f"Starting CryptoTradingBot | Mode: {mode}")

    if mode in ("fetch", "full"):
        fetcher = OHLCVFetcher()
        fetcher.fetch(settings.SYMBOL, settings.PRIMARY_TF, settings.START_DATE)
        fetcher.fetch(settings.SYMBOL, settings.HIGHER_TF, settings.START_DATE)

    if mode in ("features", "full"):
        pipeline = FeaturePipeline()
        df = pipeline.run()
        logger.info(f"Feature matrix: {df.shape}")

    if mode in ("train", "full"):
        from crypto.src.features.feature_selector import get_X_y, FEATURE_COLS
        import pandas as pd
        df = pd.read_parquet(settings.DATA_DIR / "processed" / "BTC_USDT_1h_features.parquet")
        split = temporal_split(df)

        X_train, y_train = get_X_y(split.train)
        X_val,   y_val   = get_X_y(split.val)
        X_test,  y_test  = get_X_y(split.test)

        model = XGBoostTradingModel()
        model.train(X_train, y_train, X_val, y_val)

        signals = pd.Series(model.predict(X_test), index=split.test.index, name="signal")
        bt = VectorBTBacktester()
        portfolio = bt.run(split.test, signals)
        metrics = bt.get_metrics(portfolio)
        logger.info(f"Backtest results: {metrics}")
        portfolio.plot().show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fetch", "features", "train", "full"], default="full")
    args = parser.parse_args()
    run_pipeline(args.mode)
```

---

## 12. Best Practices Checklist

### Anti-Overfitting

- [ ] Temporal train/val/test split — no random shuffling
- [ ] Walk-forward validation with at least 4 folds
- [ ] Optuna early stopping on validation loss (not test)
- [ ] SHAP feature analysis: remove features with near-zero importance
- [ ] Minimum 30 trades per backtest window for statistical validity
- [ ] Check in-sample vs out-of-sample Sharpe degradation < 50%
- [ ] No parameter fitting on test set — ever
- [ ] Penalize model complexity (L1/L2 regularization in XGBoost; weight decay in LSTM)

### Lookahead Bias Prevention

- [ ] All indicators computed with `df.iloc[:t+1]` semantics — no future data in rolling windows
- [ ] Target variable always shifted forward: `df["close"].shift(-horizon)`
- [ ] Signals generated on bar close, execution on next bar open
- [ ] Multi-TF features: daily values forward-filled to hourly (not back-filled)
- [ ] Train/val split boundary never crossed during feature normalization (fit scaler on train only)

### Reproducibility

- [ ] `RANDOM_SEED` set in numpy, torch, sklearn, and XGBoost
- [ ] Git commit hash logged with every experiment
- [ ] `requirements.txt` pinned with exact versions
- [ ] Config file (settings.py) committed to MLflow as artifact
- [ ] Data hash logged (SHA256 of parquet file)

### System Performance (i5-12400F + RTX 3060)

- [ ] XGBoost: `tree_method="hist"`, `device="cuda"` — 10× speedup over CPU
- [ ] LSTM: batch_size=256–512 for 8GB VRAM; use `pin_memory=True` in DataLoader
- [ ] PPO: 4 parallel envs on CPU; policy network on CUDA
- [ ] Use parquet for all data I/O (10× faster than CSV for feature matrices)
- [ ] VectorBT is 100% vectorized — never loop over bars in backtesting

### Testing

```
tests/
├── test_data_pipeline.py      # Validate OHLCV integrity, no NaNs, sorted index
├── test_features.py           # Check no lookahead: assert features[t] == f(df[:t+1])
├── test_target.py             # Verify labels are future-shifted
├── test_backtest.py           # Commission/slippage correctly applied
└── test_signals.py            # Signal values ∈ {-1, 0, 1}
```

```python
# tests/test_features.py
def test_no_lookahead(df_full, df_subset):
    """Features on subset must match features on full dataset at same timestamps."""
    tech = TechnicalFeatures()
    feats_full   = tech.compute(df_full)
    feats_subset = tech.compute(df_subset)  # df_subset = df_full.iloc[:500]

    overlap_idx = feats_full.index.intersection(feats_subset.index)
    pd.testing.assert_frame_equal(
        feats_full.loc[overlap_idx],
        feats_subset.loc[overlap_idx],
        check_exact=False, rtol=1e-5
    )
```

---

## Execution Roadmap

| Phase | Key Deliverable | Est. Time |
|---|---|---|
| 1. Environment Setup | Verified GPU, installed deps, config layer | 0.5 days |
| 2. Data Engineering | Clean BTC/USDT parquet files (1h + 1d) | 1 day |
| 3. Feature Engineering | Feature matrix with 30+ indicators, target labels | 2 days |
| 4. Rule-Based Strategy | EMA/RSI signal generator + backtest | 1 day |
| 5. XGBoost Model | Trained & Optuna-optimized classifier | 2 days |
| 6. LSTM Model | Sequence model, GPU-trained | 2 days |
| 7. RL / PPO | Trading environment + trained PPO agent | 3 days |
| 8. Ensemble + WFV | Combined model + walk-forward validation | 2 days |
| 9. Evaluation | Full metrics, overfitting checks, Monte Carlo | 1 day |
| 10. Refactor for Multi-Asset | Shared base classes, forex/commodities stubs | 2 days |
| **Total** | **Production-ready prototype** | **~17 days** |

---

*Built for: Trading-Bot / crypto module | Target: BTC/USD → Multi-Asset*  
*Architecture: Python 3.11 + XGBoost + PyTorch LSTM + Stable-Baselines3 PPO + VectorBT*
