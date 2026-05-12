# Trading-Bot: Production-Grade Implementation Guide

**Target:** CryptoTradingBot (BTC/USD) → Multi-Asset System  
**Stack:** Python 3.11 | Windows | i5-12400F | RTX 3060 12GB | 16GB RAM  
**Last Updated:** May 2026

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
# Environment is managed with Conda (Anaconda)
# Environment name: tradingbot | Python: 3.11.15

# List environments
conda info --envs

# Activate the trading bot environment
conda activate tradingbot

# Verify Python version
python --version  # Python 3.11.15

# Install dependencies (run from crypto/ directory)
cd crypto
pip install -r requirements.txt
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
scipy==1.13.0

# ML
xgboost==2.0.3
scikit-learn==1.5.0
optuna==3.6.1          # Hyperparameter optimization
shap==0.45.0           # Feature importance

# Deep Learning / LSTM
# IMPORTANT: Install via CUDA index, NOT plain pip install:
# pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
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
from loguru import logger


def verify_gpu(verbose: bool = True) -> dict:
    """Verifies CUDA GPU availability and logs a summary.

    Raises RuntimeError if CUDA is not available.
    """
    result = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }

    if not result["cuda_available"]:
        logger.error("CUDA is NOT available. Check your PyTorch CUDA build and NVIDIA drivers.")
        raise RuntimeError(
            "CUDA not available. Fix with: "
            "pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 "
            "--index-url https://download.pytorch.org/whl/cu121"
        )

    props = torch.cuda.get_device_properties(0)
    result["device_name"] = props.name
    result["vram_total_gb"] = round(props.total_memory / 1e9, 2)

    freemem, _ = torch.cuda.mem_get_info(0)
    result["vram_free_gb"] = round(freemem / 1e9, 2)

    if verbose:
        separator = "-" * 45
        logger.info(separator)
        logger.info("GPU CHECK")
        logger.info(separator)
        logger.info(f"CUDA Available : {result['cuda_available']}")
        logger.info(f"Device         : {result['device_name']}")
        logger.info(f"VRAM Total     : {result['vram_total_gb']} GB")
        logger.info(f"VRAM Free      : {result['vram_free_gb']} GB")
        logger.info(f"CUDA Version   : {result['cuda_version']}")
        logger.info(f"PyTorch Version: {result['torch_version']}")
        logger.info(separator)

    return result


if __name__ == "__main__":
    verify_gpu()
```

### 1.4 Configuration Layer

```python
# crypto/config/settings.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Load from .env if present
load_dotenv()

# Paths
BASEDIR = Path(__file__).resolve().parent.parent  # .../crypto
DATADIR = BASEDIR / "data"
MODELDIR = BASEDIR / "models"
LOGDIR = BASEDIR / "logs"

# Create directories if they don't exist
for d in [DATADIR / "raw", DATADIR / "processed", DATADIR / "splits", MODELDIR, LOGDIR]:
    d.mkdir(parents=True, exist_ok=True)

# Exchange & Symbol
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
PRIMARYTF = os.getenv("PRIMARYTF", "1h")
HIGHERTF = os.getenv("HIGHERTF", "1d")
STARTDATE = os.getenv("STARTDATE", "2020-01-01")

# Trading Parameters
INITIALCAPITAL = float(os.getenv("INITIALCAPITAL", 10_000.0))
COMMISSION = float(os.getenv("COMMISSION", 0.001))      # 0.1% Binance taker
SLIPPAGE = float(os.getenv("SLIPPAGE", 0.0005))         # 0.05% slippage estimate
MAXPOSITIONSIZE = float(os.getenv("MAXPOSITIONSIZE", 0.95))
RISKPERTRADE = float(os.getenv("RISKPERTRADE", 0.02))

# Reproducibility
RANDOMSEED = int(os.getenv("RANDOMSEED", 42))
```

---

## 2. Phase 1 — Data Engineering

Objective: Build a clean, timestamped, multi‑timeframe OHLCV dataset stored in an efficient format.

The data layer is responsible for:
    * Downloading OHLCV from the exchange via CCXT
    * Persisting it as Parquet under data/raw/
    * Loading and validating it before feature engineering
    * Providing a consistent directory structure that other phases depend on

### 2.1 Module: `src/data/fetcher.py`

```python
import ccxt
import pandas as pd
from loguru import logger

from crypto.config import settings


class OHLCVFetcher:
    """Fetches OHLCV data from exchange via CCXT. Rate-limit safe."""

    def __init__(self, exchange_id: str = settings.EXCHANGE):
        self.exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    def fetch(
        self,
        symbol: str = settings.SYMBOL,
        timeframe: str = settings.PRIMARYTF,
        since: str = settings.STARTDATE,
        save: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV and optionally save to data/raw as Parquet."""
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
            since_ms = ohlcv[-1][0] + 1  # continue from last timestamp

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

    def _save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        safe_symbol = symbol.replace("/", "_")
        path = settings.DATADIR / "raw" / f"{safe_symbol}_{timeframe}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {len(df)} rows → {path}")
```

**Why Parquet?**  Columnar storage, compressed (~5x vs CSV), preserves dtypes, fast pandas/PyArrow reads. Use `snappy` compression for best read speed. Reserve CSV for human-readable exports only.

### 2.2 Module: `src/data/loader.py`

```python
import pandas as pd

from crypto.config import settings


class DataLoader:
    """Loads, validates, and optionally resamples persisted data."""

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        safe_symbol = symbol.replace("/", "_")
        path = settings.DATADIR / "raw" / f"{safe_symbol}_{timeframe}.parquet"
        df = pd.read_parquet(path, engine="pyarrow")
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        required = ["open", "high", "low", "close", "volume"]
        assert all(c in df.columns for c in required), "Missing OHLCV columns"
        assert df.index.is_monotonic_increasing, "Index not sorted"
        assert not df.isnull().any().any(), f"NaN values found: {df.isnull().sum()}"
        assert (df["high"] >= df["low"]).all(), "High < Low detected"

    def resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Downsample to a higher timeframe. E.g., '4h', '1d'."""
        return (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
```
DataLoader is the single entry point for reading persisted OHLCV before feature engineering and splitting. It enforces basic data integrity (sorted index, no NaNs, valid OHLCV relationships).

### 2.3 Data Directory Structure

Actual structure, based on config/settings.py and later phases:
```
crypto/
└── data/
    ├── raw/
    │   ├── BTCUSDT_1h.parquet
    │   └── BTCUSDT_1d.parquet
    ├── processed/
    │   └── BTCUSDT_1h_features.parquet
    └── splits/
        ├── train.parquet
        ├── val.parquet
        └── test.parquet
```
    * data/raw/ – exchange OHLCV pulls (per symbol × timeframe)
    * data/processed/ – feature matrices produced by the feature pipeline
    * data/splits/ – optional persisted train/val/test splits for reproducibility

All these directories are created automatically when config/settings.py is imported.

### 2.4 Expected Outputs

| Output                                | Description                                           |
| ------------------------------------- | ----------------------------------------------------- |
| raw/BTCUSDT_1h.parquet                | Hourly OHLCV for primary timeframe (2020–present)     |
| raw/BTCUSDT_1d.parquet                | Daily OHLCV for macro context                         |
| processed/BTCUSDT_1h_features.parquet | Feature matrix with indicators, patterns, target      |
| Validation logs                       | Zero NaNs, sorted index, OHLCV integrity (via loader) |
---

Exact row counts depend on STARTDATE and current date, but for a 2020 start you typically see tens of thousands of hourly bars and a few thousand daily bars.

## 3. Phase 2 — Feature Engineering

Objective: Compute technical indicators as ML‑ready features, avoiding lookahead bias.

Golden Rule of Lookahead Prevention: Every feature at time t must be computable using only data up to and including bar t. Never shift a feature backward into the past. All merges between timeframes in the pipeline are done via forward‑fill from already‑closed bars, not from future data.

### 3.1 Module: `src/features/technical.py`

Your current implementation uses the ta library and is consumed directly by the feature pipeline. The key idea is the same: compute trend, momentum, volatility, volume, and price‑derived features in a strictly causal manner.

```python
# crypto/src/features/technical.py
import pandas as pd
import numpy as np
import ta


class TechnicalFeatures:
    """
    Computes indicators using the `ta` library.
    All values at index t use only [0..t] data — no lookahead.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- Trend ---
        out["ema9"]   = ta.trend.ema_indicator(df["close"], window=9)
        out["ema21"]  = ta.trend.ema_indicator(df["close"], window=21)
        out["ema50"]  = ta.trend.ema_indicator(df["close"], window=50)
        out["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
        out["ema_cross_9_21"] = (out["ema9"] > out["ema21"]).astype(int)

        # --- MACD ---
        macd = ta.trend.MACD(
            df["close"], window_slow=26, window_fast=12, window_sign=9
        )
        out["macd"]        = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macdhist"]    = macd.macd_diff()

        # --- Momentum ---
        out["rsi14"] = ta.momentum.rsi(df["close"], window=14)
        out["rsi7"]  = ta.momentum.rsi(df["close"], window=7)

        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=14, smooth_window=3
        )
        out["stochk"] = stoch.stoch()
        out["stochd"] = stoch.stoch_signal()

        # --- Volatility ---
        bb = ta.volatility.BollingerBands(
            df["close"], window=20, window_dev=2
        )
        out["bbupper"] = bb.bollinger_hband()
        out["bbmid"]   = bb.bollinger_mavg()
        out["bblower"] = bb.bollinger_lband()
        out["bbwidth"] = bb.bollinger_wband()
        out["bbpct"]   = bb.bollinger_pband()

        out["atr14"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )
        out["atrpct"] = out["atr14"] / df["close"]

        # --- Volume ---
        out["volumesma20"] = df["volume"].rolling(20).mean()
        out["volumeratio"] = df["volume"] / out["volumesma20"]
        out["obv"]         = ta.volume.on_balance_volume(df["close"], df["volume"])

        vwap = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], df["close"], df["volume"]
        )
        out["vwap"] = vwap.volume_weighted_average_price()

        # --- Price-Derived ---
        out["logreturn"]      = np.log(df["close"] / df["close"].shift(1))
        out["return5"]        = df["close"].pct_change(5)
        out["return24"]       = df["close"].pct_change(24)
        out["highlowspread"]  = (df["high"] - df["low"]) / df["close"]
        out["closepctrange"]  = (
            (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
        )

        return out.dropna()
```
Names like ema9, ema21, ema50, rsi14, macdhist, volumeratio, and vwap are intentionally aligned with what your strategies and backtester expect later.

### 3.2 Module: `src/features/patterns.py`

The new implementation no longer uses pandas_ta CDL functions; instead, it derives a small set of robust candlestick patterns from raw OHLC using simple rules. These become pat_* binary features for ML.

```python
# crypto/src/features/patterns.py
import pandas as pd
import numpy as np


class ChartPatternFeatures:
    """
    Candlestick pattern detection using pure pandas.
    No external library dependency.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        body      = df["close"] - df["open"]
        abs_body  = body.abs()
        rng       = df["high"] - df["low"] + 1e-9
        upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
        lower_wick = df[["close", "open"]].min(axis=1) - df["low"]

        # Doji: body < 10% of range
        out["pat_doji"] = (abs_body < 0.1 * rng).astype(int)

        # Hammer: small body, long lower wick (>2x body), tiny upper wick
        out["pat_hammer"] = (
            (lower_wick > 2 * abs_body)
            & (upper_wick < 0.3 * abs_body)
            & (abs_body > 0)
        ).astype(int)

        # Shooting Star: small body, long upper wick, tiny lower wick
        out["pat_shootingstar"] = (
            (upper_wick > 2 * abs_body)
            & (lower_wick < 0.3 * abs_body)
            & (abs_body > 0)
        ).astype(int)

        # Bullish Engulfing
        prev_body  = body.shift(1)
        prev_open  = df["open"].shift(1)
        prev_close = df["close"].shift(1)
        out["pat_engulfing_bull"] = (
            (prev_body < 0)   # previous bar bearish
            & (body > 0)      # current bar bullish
            & (df["open"] < prev_close)
            & (df["close"] > prev_open)
        ).astype(int)

        # Bearish Engulfing
        out["pat_engulfing_bear"] = (
            (prev_body > 0)   # previous bar bullish
            & (body < 0)      # current bar bearish
            & (df["open"] > prev_close)
            & (df["close"] < prev_open)
        ).astype(int)

        # Marubozu: body > 90% of range (strong momentum candle)
        out["pat_marubozu"] = (abs_body > 0.9 * rng).astype(int)

        return out.fillna(0)
```
These pattern columns are primarily used as additional explanatory features for ML, not by the rule‑based strategies.

### 3.3 Module: `src/features/target.py`

This file defines the classification labels (-1/0/1) and an optional regression target. It is central to avoiding lookahead: targets are computed from future returns and must never be used as input features.

```python
# crypto/src/features/target.py
import pandas as pd
import numpy as np
from enum import IntEnum


class SignalType(IntEnum):
    SHORT = -1
    HOLD  =  0
    LONG  =  1


class TargetBuilder:
    """
    Creates forward-looking labels for supervised learning.
    Labels at time t are computed using future bars [t+1 .. t+horizon].
    NEVER use these labels as input features — only as targets.
    """

    def build_classification(
        self,
        df: pd.DataFrame,
        horizon: int = 12,       # 12 bars ahead (12h for 1h data)
        threshold: float = 0.015 # 1.5% move = signal
    ) -> pd.Series:
        """
        Returns:
            +1 (LONG)  if price rises > threshold in next `horizon` bars
            -1 (SHORT) if price falls > threshold
             0 (HOLD)  otherwise
        """
        future_return = df["close"].shift(-horizon) / df["close"] - 1.0

        conditions = [
            future_return > threshold,
            future_return < -threshold,
        ]
        choices = [SignalType.LONG, SignalType.SHORT]

        target = pd.Series(
            np.select(conditions, choices, default=SignalType.HOLD),
            index=df.index,
            name="target",
        )
        # Last `horizon` rows have no valid future → set NaN
        target.iloc[-horizon:] = np.nan
        return target

    def build_regression(
        self, df: pd.DataFrame, horizon: int = 12
    ) -> pd.Series:
        """Continuous log-return over horizon. For regression models."""
        return np.log(df["close"].shift(-horizon) / df["close"]).rename(
            f"logret_{horizon}h"
        )
```
Downstream code (models and backtests) use the target column only as the label; it is explicitly excluded from the FEATURECOLS list later.

### 3.4 Feature Pipeline

The pipeline orchestrates:
    1. Loading raw 1h and 1d data
    2. Computing 1h technical + pattern features
    3. Computing 1d technical “macro” features
    4. Merging daily features into hourly via forward‑fill (no lookahead)
    5. Building the classification target
    6. Dropping NaNs (warmup + trailing horizon)
    7. Recording FEATURECOLS and saving to data/processed

```python
# crypto/src/features/pipeline.py
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Allow running pipeline as script if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))

import config.settings as cfg
from src.data.loader import DataLoader
from src.features.technical import TechnicalFeatures
from src.features.patterns import ChartPatternFeatures
from src.features.target import TargetBuilder

FEATURECOLS: list[str] = []  # populated after first run


class FeaturePipeline:
    """
    Orchestrates full feature construction for one symbol/timeframe.
    Merges multi-timeframe context and saves processed dataset.
    """

    def __init__(self) -> None:
        self.loader   = DataLoader()
        self.tech     = TechnicalFeatures()
        self.patterns = ChartPatternFeatures()
        self.targets  = TargetBuilder()

    def run(
        self,
        symbol: str = cfg.SYMBOL,
        primary_tf: str = cfg.PRIMARYTF,
        higher_tf: str = cfg.HIGHERTF,
        horizon: int = 12,
        threshold: float = 0.015,
        save: bool = True,
    ) -> pd.DataFrame:
        logger.info("FeaturePipeline starting ...")

        # Step 1–2: Load raw data
        df_1h = self.loader.load(symbol, primary_tf)
        df_1d = self.loader.load(symbol, higher_tf)
        logger.info(f"Loaded 1h: {len(df_1h)} rows, 1d: {len(df_1d)} rows")

        # Step 3: Technical indicators on 1h
        logger.info("Computing 1h technical indicators ...")
        df_1h = self.tech.compute(df_1h)

        # Step 4: Candlestick patterns on 1h
        logger.info("Computing candlestick patterns ...")
        df_1h = self.patterns.compute(df_1h)

        # Step 5: Technical indicators on 1d (macro)
        logger.info("Computing 1d macro indicators ...")
        df_1d_feats = self.tech.compute(df_1d)

        # Step 6: Merge daily macro features into 1h (forward-fill, no lookahead)
        logger.info("Merging daily macro features into 1h ...")
        df_1h = self.merge_macro(df_1h, df_1d_feats, primary_tf)

        # Step 7: Build classification target
        logger.info(
            f"Building classification target: horizon={horizon}h, "
            f"threshold={threshold*100:.1f}% ..."
        )
        df_1h["target"] = self.targets.build_classification(
            df_1h, horizon=horizon, threshold=threshold
        )

        # Step 8: Drop NaN rows (indicator warmup + trailing horizon)
        before = len(df_1h)
        df_1h.dropna(inplace=True)
        logger.info(f"After dropna: {len(df_1h)} rows (dropped {before - len(df_1h)})")

        # Record feature columns (exclude raw prices, volume, target)
        exclude = {"open", "high", "low", "close", "volume", "target"}
        global FEATURECOLS
        FEATURECOLS = [c for c in df_1h.columns if c not in exclude]
        logger.info(f"Feature columns: {len(FEATURECOLS)}")

        # Step 9: Save processed matrix
        if save:
            self.save(df_1h, symbol, primary_tf)
            logger.info("FeaturePipeline complete.")

        return df_1h

    def merge_macro(
        self,
        df_primary: pd.DataFrame,
        df_daily: pd.DataFrame,
        primary_tf: str,
    ) -> pd.DataFrame:
        """
        Forward-fill daily indicator values into the primary 1h DataFrame.

        A daily bar closes at its daily close time and is then propagated
        forward into subsequent intraday bars (never backward), so no
        future daily information leaks into past intraday bars.
        """
        daily_indicator_cols = [
            "ema50", "ema200", "rsi14", "atrpct", "bbpct", "bbwidth",
            "adx", "macdhist", "realisedvol24",
        ]
        available = [c for c in daily_indicator_cols if c in df_daily.columns]
        daily_sub = df_daily[available].add_prefix("d1_")

        daily_resampled = daily_sub.resample(primary_tf).ffill()
        merged = df_primary.join(daily_resampled, how="left")

        missing = merged[daily_sub.columns].isnull().sum().sum()
        if missing > 0:
            logger.warning(
                f"Macro merge: {missing} NaN values remain "
                f"(check date alignment between timeframes)."
            )
        else:
            logger.info(
                f"Macro merge: {len(daily_sub.columns)} daily features merged "
                f"into {len(merged)} hourly rows with no NaNs."
            )

        return merged

    def save(self, df: pd.DataFrame, symbol: str, primary_tf: str) -> None:
        safe = symbol.replace("/", "_")
        outpath = cfg.DATADIR / "processed" / f"{safe}_{primary_tf}_features.parquet"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outpath, engine="pyarrow", compression="snappy")
        logger.info(f"Feature matrix saved: {outpath.name}, shape={df.shape}")
```
This pipeline is exactly what main.py calls in --mode features and implicitly in --mode strategy / --mode train when the processed file does not exist.

### 3.5 Time-Series Train/Val/Test Split

The temporal split function is defined under src/data/splitter.py and is used in both rule‑based strategy backtests and ML training/evaluation. It respects chronological order to avoid leakage.

```python
# crypto/src/data/splitter.py
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
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )

    print(
        f"Train: {split.train.index[0]} → {split.train.index[-1]} "
        f"({len(split.train)} rows)"
    )
    print(
        f"Val:   {split.val.index[0]}   → {split.val.index[-1]} "
        f"({len(split.val)} rows)"
    )
    print(
        f"Test:  {split.test.index[0]}  → {split.test.index[-1]} "
        f"({len(split.test)} rows)"
    )

    return split
```

Walk‑Forward Validation (WFV): For production‑quality evaluation, use expanding or sliding‑window retraining instead of a single split. Later phases describe WFV utilities that build on this temporal split.

---

## 4. Phase 3 — Strategy Design (Rule-Based)

Objective: Define deterministic trading logic that generates buy/sell signals without any ML.

Rule‑based strategies operate on the feature matrix from Phase 2 and must obey the same lookahead constraints: a signal at time t can only depend on data available up to and including bar t.

### 4.1 Base Strategy Interface

```python
# crypto/src/strategies/base.py
"""
Abstract base class for all trading strategies.

Every strategy must implement generate_signals() and respect the
lookahead contract: signal at time t may only use data up to and
including bar t. No future data, ever.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    # Subclasses set these for identification
    NAME: str = "base"
    VERSION: str = "1.0"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate raw entry/exit signals from a feature DataFrame.

        Args:
            df: Feature DataFrame with OHLCV + indicator columns,
                UTC-indexed, sorted ascending.

        Returns:
            pd.Series of int, same index as df:
                +1 → LONG signal (buy / enter long)
                 0 → HOLD (no action / maintain position)
                -1 → SHORT signal (sell / enter short / exit long)

        Lookahead contract:
            Signal at index t must only reference df.iloc[:t+1].
            Vectorised operations on the full df are fine as long as
            they don't depend on future values (e.g. no shift(-n)).
        """

    # ── Shared validation helpers ───────────────────────────────────────
    def _validate_signals(self, signals: pd.Series, df: pd.DataFrame):
        assert signals.index.equals(df.index), (
            "Signal index does not match DataFrame index."
        )
        assert signals.isin([-1, 0, 1]).all(), (
            f"Invalid signal values found: {signals.unique()}"
        )
        assert not signals.isnull().any(), "NaN values in signals."

    def _require_columns(self, df: pd.DataFrame, cols: list[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.NAME}: required columns not found in DataFrame: {missing}\n"
                f"Available: {list(df.columns)}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.NAME}, version={self.VERSION})"
```
All concrete strategies inherit from this base, enforce required feature columns, and return an integer signal series aligned to the feature matrix index.

### 4.2  Strategy 1: EMA Crossover + RSI Filter

```python
# crypto/src/strategies/ema_rsi.py
"""
Strategy 1: EMA Crossover + RSI Filter

Entry logic (LONG):
- EMA9 crosses above EMA21 (momentum shift)
- RSI14 > rsi_long_threshold (trend confirmation, not overbought)
- Price above EMA50 (macro trend filter)

Exit logic (signal = -1):
- EMA9 crosses below EMA21 OR
- RSI14 > rsi_exit_ob (overbought exit) OR
- Price drops below EMA50 (trend breakdown)

No short entries in default mode (long_only=True).
"""

import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy


class EMACrossoverRSI(BaseStrategy):

    NAME = "ema_crossover_rsi"
    VERSION = "1.0"

    def __init__(
        self,
        rsi_long_threshold: float = 50.0,  # RSI must be above this to enter long
        rsi_exit_ob: float = 75.0,        # Exit long if RSI becomes overbought
        rsi_short_threshold: float = 50.0, # RSI must be below this to enter short
        rsi_exit_os: float = 25.0,        # Exit short if RSI becomes oversold
        use_ema50_filter: bool = True,    # Require price > EMA50 for longs
        long_only: bool = True,           # Trade longs only
    ):
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_exit_ob = rsi_exit_ob
        self.rsi_short_threshold = rsi_short_threshold
        self.rsi_exit_os = rsi_exit_os
        self.use_ema50_filter = use_ema50_filter
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        required = ["ema_9", "ema_21", "rsi_14", "close"]
        if self.use_ema50_filter:
            required.append("ema_50")
        self._require_columns(df, required)

        # ── EMA crossover detection ─────────────────────────────────────
        fast_above_slow = df["ema_9"] > df["ema_21"]
        cross_up = fast_above_slow & ~fast_above_slow.shift(1).fillna(False)
        cross_down = ~fast_above_slow & fast_above_slow.shift(1).fillna(True)

        # ── Filters ─────────────────────────────────────────────────────
        rsi_long_ok = df["rsi_14"] > self.rsi_long_threshold
        rsi_exit_ob = df["rsi_14"] > self.rsi_exit_ob
        rsi_short_ok = df["rsi_14"] < self.rsi_short_threshold
        rsi_exit_os = df["rsi_14"] < self.rsi_exit_os

        price_above_ema50 = (
            df["close"] > df["ema_50"]
            if self.use_ema50_filter
            else pd.Series(True, index=df.index)
        )

        # ── Signal construction ─────────────────────────────────────────
        signals = pd.Series(0, index=df.index, dtype=int)

        # Long entry: cross up + RSI confirms + macro trend ok
        long_entry = cross_up & rsi_long_ok & price_above_ema50
        signals[long_entry] = 1

        # Long exit: cross down OR overbought OR trend breakdown
        long_exit = cross_down | rsi_exit_ob | ~price_above_ema50
        signals[long_exit & (signals != 1)] = -1  # don't override same-bar entry

        if not self.long_only:
            # Short entry: cross down + RSI confirms bearish
            short_entry = cross_down & rsi_short_ok & ~price_above_ema50
            signals[short_entry] = -1
            # (Short exit via rsi_exit_os etc. could be added symmetrically)

        self._validate_signals(signals, df)

        n_long = (signals == 1).sum()
        n_exit = (signals == -1).sum()
        logger.info(
            f"{self.NAME}: {n_long} long entries | {n_exit} exit signals "
            f"| {len(df):,} total bars"
        )

        return signals
```
In main.py --mode strategy, this class is instantiated with specific hyperparameters (e.g., RSI thresholds, EMA50 filter, long_only=True) and applied to the test split features.

### 4.3  Strategy 2: Multi‑Indicator Confluence

This strategy trades only when multiple independent signals agree (trend, momentum, volume, VWAP), aiming for fewer but higher‑quality trades.

```python
# crypto/src/strategies/multi_indicator.py
"""
Strategy 2: Multi-Indicator Confluence

Philosophy: only trade when multiple independent signals agree.
Reduces false signals at the cost of fewer total trades.
"""

import pandas as pd
from loguru import logger

from src.strategies.base import BaseStrategy


class MultiIndicatorConfluence(BaseStrategy):

    NAME = "multi_indicator_confluence"
    VERSION = "1.0"

    def __init__(
        self,
        rsi_low: float = 45.0,
        rsi_high: float = 70.0,
        rsi_ob_exit: float = 78.0,
        rsi_os_exit: float = 22.0,
        volume_surge: float = 1.2,     # volume must be > N × 20-bar avg
        macd_rising_bars: int = 2,     # MACD hist must have risen for N bars
        long_only: bool = True,
    ):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.rsi_ob_exit = rsi_ob_exit
        self.rsi_os_exit = rsi_os_exit
        self.volume_surge = volume_surge
        self.macd_rising_bars = macd_rising_bars
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        required = [
            "ema_9", "ema_21", "ema_50",
            "rsi_14", "macd_hist", "volume_ratio",
            "vwap", "close", "atr_14",
        ]
        self._require_columns(df, required)

        # ── Trend conditions ────────────────────────────────────────────
        ema_bullish = (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])
        ema_bearish = (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])

        # ── RSI conditions ──────────────────────────────────────────────
        rsi_bull_range = (df["rsi_14"] > self.rsi_low) & (df["rsi_14"] < self.rsi_high)
        rsi_bear_range = (
            df["rsi_14"] > (100 - self.rsi_high)
        ) & (df["rsi_14"] < (100 - self.rsi_low))
        rsi_overbought = df["rsi_14"] > self.rsi_ob_exit
        rsi_oversold = df["rsi_14"] < self.rsi_os_exit

        # ── MACD conditions ─────────────────────────────────────────────
        macd_positive = df["macd_hist"] > 0
        macd_negative = df["macd_hist"] < 0

        # MACD histogram rising/falling for N consecutive bars
        macd_rising = df["macd_hist"] > df["macd_hist"].shift(self.macd_rising_bars)
        macd_falling = df["macd_hist"] < df["macd_hist"].shift(self.macd_rising_bars)

        # ── Volume conditions ───────────────────────────────────────────
        vol_surge = df["volume_ratio"] > self.volume_surge

        # ── Price vs VWAP ───────────────────────────────────────────────
        above_vwap = df["close"] > df["vwap"]
        below_vwap = df["close"] < df["vwap"]

        # ── Entry signals — ALL conditions must agree ───────────────────
        long_entry = (
            ema_bullish
            & rsi_bull_range
            & macd_positive
            & macd_rising
            & vol_surge
            & above_vwap
        )

        short_entry = (
            ema_bearish
            & rsi_bear_range
            & macd_negative
            & macd_falling
            & vol_surge
            & below_vwap
        )

        # ── Exit signals ────────────────────────────────────────────────
        long_exit = (~ema_bullish) | rsi_overbought  # trend breakdown or extreme OB
        short_exit = (~ema_bearish) | rsi_oversold   # trend breakdown or extreme OS

        # ── Build signal series ─────────────────────────────────────────
        signals = pd.Series(0, index=df.index, dtype=int)

        signals[long_entry] = 1
        signals[long_exit & (signals != 1)] = -1

        if not self.long_only:
            signals[short_entry] = -1
            signals[short_exit & (signals != -1)] = 1

        self._validate_signals(signals, df)

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        logger.info(
            f"{self.NAME}: {n_long} long | {n_short} short signals "
            f"| confluence filter active (all conditions required)"
        )

        return signals
```
In main.py, this strategy is used as a “confluence baseline” and compared against EMA+RSI and later against ML models on the same test split.

### 4.4 Signal → Position Mapping and Sizing

Signals are instantaneous events; positions represent the held state over time. This separation makes it easy to reason about entry/exit logic vs portfolio exposure.

```python
# crypto/src/strategies/position_manager.py
"""
Signal → Position conversion.

Signals are instantaneous (fire once per bar).
Positions represent what you are currently holding (persist between bars).

Signal series:   +1 / 0 / -1  (sparse — mostly 0)
Position series: +1 / 0 / -1  (dense — reflects current holding)
"""

import pandas as pd
import numpy as np
from loguru import logger


def signals_to_positions(
    signals: pd.Series,
    mode: str = "long_only",
) -> pd.Series:
    """
    Convert raw entry/exit signals into a held-position series.

    Args:
        signals: Series of {-1, 0, +1} indexed by timestamp.
        mode:
            "long_only"  — only long positions, -1 signal = exit to flat
            "long_short" — both long and short positions allowed

    Returns:
        pd.Series of {-1, 0, +1} with same index:
            +1 = currently long
             0 = flat (no position)
            -1 = currently short

    Example (long_only):
        signals:   0 0 1 0 0 -1 0 1 0
        positions: 0 0 1 1 1  0 0 1 1
    """
    assert mode in ("long_only", "long_short"), f"Unknown mode: {mode}"
    assert signals.isin([-1, 0, 1]).all(), "Signals must be in {-1, 0, 1}"

    positions = pd.Series(0, index=signals.index, dtype=int)
    current = 0

    for i, sig in enumerate(signals):
        if mode == "long_only":
            if sig == 1:
                current = 1   # enter long
            elif sig == -1:
                current = 0   # exit to flat (no shorting)
        elif mode == "long_short":
            if sig == 1:
                current = 1   # enter/switch to long
            elif sig == -1:
                current = -1  # enter/switch to short

        positions.iloc[i] = current

    _log_position_stats(positions, mode)
    return positions


def apply_position_sizing(
    positions: pd.Series,
    df: pd.DataFrame,
    risk_pct: float = 0.02,      # risk 2% of capital per trade
    atr_multiplier: float = 2.0, # stop-loss = entry ± atr_multiplier × ATR
) -> pd.Series:
    """
    Scale position size based on ATR-derived risk.
    Returns a Series of position sizes in [0, 1] representing
    fraction of capital to deploy.

    Formula:
        stop_distance = atr_multiplier × ATR14
        position_size = risk_pct / (stop_distance / close)

    Capped at 1.0 (100% of capital) to prevent over-leveraging.
    """
    if "atr_14" not in df.columns:
        logger.warning(
            "atr_14 not found — using fixed 100% position size. "
            "Run TechnicalFeatures.compute() first."
        )
        return positions.abs().astype(float)

    atr = df["atr_14"]
    close = df["close"]
    stop_dist = atr_multiplier * atr
    stop_pct = stop_dist / close

    # Avoid division by zero on zero-ATR rows
    raw_size = risk_pct / stop_pct.replace(0, np.nan)
    raw_size = raw_size.clip(upper=1.0).fillna(1.0)

    # Apply only where we have a position
    sized = positions.abs().astype(float) * raw_size
    sized = sized.clip(upper=1.0)

    logger.info(
        f"Position sizing: mean={sized[sized>0].mean():.2%} "
        f"min={sized[sized>0].min():.2%} max={sized[sized>0].max():.2%}"
    )

    return sized


def _log_position_stats(positions: pd.Series, mode: str):
    total = len(positions)
    n_long = (positions == 1).sum()
    n_flat = (positions == 0).sum()
    n_short = (positions == -1).sum()
    pct_invested = n_long / total * 100

    logger.info(
        f"Positions [{mode}]: "
        f"LONG {n_long:,} ({n_long/total*100:.1f}%) | "
        f"FLAT {n_flat:,} ({n_flat/total*100:.1f}%) | "
        f"SHORT {n_short:,} ({n_short/total*100:.1f}%) | "
        f"Time invested: {pct_invested:.1f}%"
    )
```

main.py typically uses signals_to_positions(..., mode="long_only") before passing positions into the backtester. The apply_position_sizing helper can be used later if you want risk‑based fractional exposure instead of unit positions.

---

## 5. Phase 4 — Machine Learning Models (XGBoost + LSTM)

### 5.1 Feature Matrix Construction

```python
# crypto/src/models/feature_selector.py
"""
Feature selection and stationarity analysis for trading models.

Core problem:
Absolute price-level features (EMA values, OBV, VWAP, BB bands)
have completely different numerical ranges across market regimes.
Tree-based models then memorise "time" via price scale.

Solution:
Use only dimensionless, scale‑invariant features for ML models.
"""

import numpy as np
import pandas as pd
from loguru import logger

# Non‑stationary features to avoid feeding into ML
NON_STATIONARY = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal",
    "bb_upper", "bb_mid", "bb_lower",
    "kc_upper", "kc_lower",
    "atr_14",
    "vwap",
    "obv",
    "volume_sma_20",
    "d1_ema_50", "d1_ema_200",
]

# Stationary, scale‑invariant features used for ML
STATIONARY = [
    # EMA cross flags
    "ema_cross_9_21", "ema_cross_21_50", "ema_cross_50_200",
    # Price vs EMA (%)
    "price_vs_ema50", "price_vs_ema200",
    # MACD histogram + cross
    "macd_hist", "macd_cross",
    # ADX, RSI, Stoch, Williams, MFI
    "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "rsi_overbought", "rsi_oversold",
    "stoch_k", "stoch_d",
    "williams_r",
    "roc_5", "roc_10",
    # Bands and volatility
    "bb_width", "bb_pct",
    "atr_pct",
    "kc_pct",
    "volume_ratio",
    "vwap_gap",
    "mfi_14",
    # Returns and price anatomy
    "log_return", "return_3", "return_5", "return_24",
    "high_low_spread", "close_pct_range",
    "realised_vol_24", "realised_vol_168",
    # Daily macro (dimensionless versions only)
    "d1_rsi_14", "d1_atr_pct", "d1_bb_pct", "d1_bb_width",
    "d1_adx", "d1_macd_hist", "d1_realised_vol_24",
    # Pattern flags
    "pat_doji", "pat_hammer", "pat_inv_hammer", "pat_shooting_star",
    "pat_marubozu_bull", "pat_marubozu_bear", "pat_spinning_top",
    "pat_engulf_bull", "pat_engulf_bear",
    "pat_harami_bull", "pat_harami_bear",
    "pat_piercing", "pat_dark_cloud",
    "pat_any_bull", "pat_any_bear",
]


def get_stationary_features(df: pd.DataFrame) -> list[str]:
    """Return list of stationary features present in df."""
    available = set(df.columns)
    selected = [f for f in STATIONARY if f in available]
    removed = [f for f in NON_STATIONARY if f in available]

    logger.info(
        f"Feature selection: {len(selected)} stationary kept | "
        f"{len(removed)} non-stationary removed"
    )
    if removed:
        logger.info(f"Removed: {removed}")

    return selected


def analyse_stationarity(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Split dataset into thirds; measure mean drift for each feature.

    Instability = (max_period_mean - min_period_mean) / |overall_mean|
    Instability > 0.5 → likely non-stationary (memorisation risk).
    """
    n = len(df)
    third = n // 3
    parts = [df.iloc[:third], df.iloc[third:2*third], df.iloc[2*third:]]

    records = []
    for col in feature_cols:
        try:
            means = [p[col].mean() for p in parts]
            mean_range = max(means) - min(means)
            mean_abs = abs(np.mean(means)) + 1e-9
            instability = mean_range / mean_abs
            records.append({
                "feature": col,
                "instability": round(instability, 4),
                "early_mean": round(means[0], 4),
                "mid_mean": round(means[1], 4),
                "late_mean": round(means[2], 4),
                "verdict": "✗ UNSTABLE" if instability > 0.5 else "✓ STABLE",
            })
        except Exception:
            pass

    return pd.DataFrame(records).sort_values("instability", ascending=False)


def print_feature_analysis(df: pd.DataFrame, feat_cols: list[str]) -> None:
    """Pretty‑print stationarity report."""
    stats = analyse_stationarity(df, feat_cols)
    sep = "─" * 72

    print(f"\n{'═'*72}")
    print(f" FEATURE STATIONARITY ANALYSIS ({len(feat_cols)} features)")
    print(f" Instability > 0.5 = likely non-stationary (memorisation risk)")
    print(f"{'═'*72}")
    print(f" {'Feature':<30} {'Instability':>12} {'Early':>9} {'Mid':>9} {'Late':>9} Verdict")
    print(sep)

    n_unstable = 0
    for _, row in stats.iterrows():
        print(
            f" {row['feature']:<30} {row['instability']:>12.4f} "
            f"{row['early_mean']:>9.4f} {row['mid_mean']:>9.4f} "
            f"{row['late_mean']:>9.4f} {row['verdict']}"
        )
        if "UNSTABLE" in str(row["verdict"]):
            n_unstable += 1

    print(sep)
    print(f" Unstable: {n_unstable}/{len(stats)} features")
    print(f"{'═'*72}\n")
```
In the pipeline, after FeaturePipeline.run() builds the full feature matrix, get_stationary_features(df) is used to select a stable subset for both XGBoost and LSTM training.

### 5.2 XGBoost Classifier v2

The v2 XGBoost model adds:
    * More robust labelling via fixed LABELMAP −1,0,1↔0,1,2.
    * Class imbalance handling with inverse‑frequency sample weights.
    * Optuna search over a richer hyperparameter space.
    * Native multi:softprob objective and a confidence threshold for signals.
    * Save/load utilities and feature importance helpers.
Conceptual summary (your file is longer; this matches its behavior):

```python
# crypto/src/models/xgboost_v2.py
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config.settings as cfg

LABEL_MAP    = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}
LABEL_NAMES  = {0: "SHORT", 1: "HOLD", 2: "LONG"}


class XGBoostTradingModel:
    """3-class XGBoost classifier for trading signals with GPU + Optuna."""

    def __init__(self, threshold: float = 0.45):
        self.model = None
        self.best_params = None
        self.feature_cols: list[str] = []
        self.threshold = threshold

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        use_gpu: bool = True,
    ) -> "XGBoostTradingModel":
        self.feature_cols = list(X_train.columns)
        y_tr = y_train.map(LABEL_MAP).values.astype(int)
        y_vl = y_val.map(LABEL_MAP).values.astype(int)

        w_tr = self.compute_weights(y_tr)
        w_vl = self.compute_weights(y_vl)

        device = "cuda" if use_gpu else "cpu"
        logger.info(f"XGBoost device: {device}")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("lr", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "device": device,
                "seed": cfg.RANDOM_SEED,
                "verbosity": 0,
            }
            clf = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            clf.fit(
                X_train, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_vl)],
                sample_weight_eval_set=[w_vl],
                verbose=False,
            )
            preds = clf.predict(X_val)
            return f1_score(y_vl, preds, average="weighted", zero_division=0)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
            study_name="xgb_btcusdt",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        # Rename lr → learning_rate if needed
        if "lr" in self.best_params:
            self.best_params["learning_rate"] = self.best_params.pop("lr")

        logger.info(f"Best val weighted-F1: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        final_params = {
            **self.best_params,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": device,
            "seed": cfg.RANDOM_SEED,
            "verbosity": 0,
        }
        self.model = xgb.XGBClassifier(**final_params, early_stopping_rounds=30)
        self.model.fit(
            X_train, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_vl)],
            sample_weight_eval_set=[w_vl],
            verbose=100,
        )
        self._print_eval(X_val, y_vl, label="Validation")
        self.save()
        return self

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Return {-1,0,1} signals with confidence filter."""
        assert self.model is not None, "Call .train() or .load() first."
        thr = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        raw = np.argmax(proba, axis=1)
        conf = proba.max(axis=1)
        raw[conf < thr] = 1  # force HOLD for low confidence
        return np.vectorize(LABEL_MAP_INV.get)(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X)

    # plus: evaluate(), feature_importance(), shap_importance(), save(), load(), compute_weights()
```
This version is what main-4.py uses when running --mode train-ml or when called from walk_forward and the experiment tracker

### 5.3 LSTM Sequence Model
The LSTM implementation in lstm_model-3.py is already robust and aligned with the guide. Its main differences from the earlier pseudocode are:
    * Separate SequenceDataset that returns windows and their last‑bar label.
    * LSTMClassifier with BatchNorm on inputs and dropout before the head.
    * LSTMTradingModel wrapper handling scaling, training loop, early stopping, evaluation, and save/load.

You do not need to change the guide much here; you can keep the existing 5.3 description but clarify:
    * Input: stationary STATIONARY features chosen by get_stationary_features.
    * Windowing: window of seqlen (default 60) bars, predicting the label at the last bar in the window.
    * Label contract: labels are the same target from TargetBuilder, with the −1,0,1 mapping as in XGBoost.


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