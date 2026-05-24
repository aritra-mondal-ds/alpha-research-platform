# Multi-Asset ML Trading Bot

A production-grade machine learning trading system for **BTC/USDT**, **AUD/USD**, and **XAU/USD (Gold)**. Fetches live market data, engineers 73 technical and macro features per asset, trains XGBoost and deep learning models, and validates results with walk-forward validation across 8 years of data.

---

## Results

### BTC/USDT — XGBoost v2 (Test: Feb 2025 → May 2026)

| Metric | XGBoost v2 | Buy & Hold |
|---|---|---|
| Total Return | **+695.74%** | -15.30% |
| Sharpe Ratio | **8.642** | — |
| Max Drawdown | **-11.44%** | -62%+ |
| Win Rate | **94.4%** | — |
| Total Trades | 54 | — |

### Walk-Forward Validation (12 folds, 2020–2026)

| Strategy | Mean Sharpe | Profitable Folds | Verdict |
|---|---|---|---|
| **XGBoost** | **+10.877** | **100%** | ✅ PASS ALL |
| Confluence | -0.630 | 33.3% | ❌ FAIL ALL |
| EMA+RSI | -1.799 | 41.7% | ❌ FAIL ALL |

XGBoost was profitable in every single 6-month test window across 8 years — including the 2022 bear market, 2023 recovery, and 2025 volatility.

### Monte Carlo (1,000 simulations, 50 test trades)

| Percentile | Return |
|---|---|
| 5th (worst case) | +346% |
| Median | +531% |
| 95th (best case) | +921% |
| **Probability of loss** | **0.0%** |

### Cross-Asset Comparison

| Asset | Strategy | Return | Sharpe | Buy & Hold |
|---|---|---|---|---|
| BTC/USDT | XGBoost v2 | **+695.74%** | 8.64 | -15.30% |
| AUD/USD | EMA+RSI | +1.58% | 2.10 | +2.19% |
| XAU/USD | Confluence | +4.39% | 12.49 | -10.82% |

### Portfolio Allocation (Inverse-Volatility, $30,000)

| Asset | Ann. Vol | Weight | Capital |
|---|---|---|---|
| AUD/USD | 11.67% | 61.0% | $18,289 |
| BTC/USDT | 29.85% | 24.5% | $7,341 |
| Gold (GLD) | 50.15% | 14.5% | $4,370 |

---

## Architecture

```
Trading-Bot/
├── main.py                   ← multi-asset orchestrator
├── shared/                   ← base classes + portfolio manager
├── crypto/                   ← BTC/USDT (full ML pipeline)
│   └── src/
│       ├── data/             ← CCXT fetcher, loader, splitter
│       ├── features/         ← 73-feature pipeline
│       ├── strategies/       ← EMA+RSI, Confluence
│       ├── backtesting/      ← backtest, WFV, metrics
│       ├── models/           ← XGBoost v1/v2, LSTM, PPO RL
│       └── utils/            ← GPU check, MLflow, report generator
├── forex/                    ← AUD/USD (yfinance)
└── commodities/              ← XAU/USD via GLD ETF (yfinance)
```

All 73 features, backtesting engine, walk-forward validator, and ML models live in `crypto/src/` and are **shared** by the forex and commodities modules.

---

## Setup

### Requirements

- Python 3.11.x (3.11.15 tested)
- CUDA 12.1 compatible GPU (optional — XGBoost runs on CPU too)
- Windows / Linux

### Install

```bash
conda create -n tradingbot python=3.11.15
conda activate tradingbot

pip install pandas numpy pyarrow xgboost optuna ccxt yfinance ta loguru mlflow
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium
```

> **Note:** Use the `ta` library throughout. `pandas-ta` is incompatible with Python 3.11.

---

## Quick Start

```bash
# Fetch data for all assets
python main.py --mode fetch-all

# Run the full crypto pipeline (takes ~60 min including WFV)
cd crypto
python main.py --mode full --trials 20

# Cross-asset portfolio view
cd ..
python main.py --mode portfolio
python main.py --mode compare-all
```

---

## Crypto Module Modes

```bash
cd crypto

python main.py --mode fetch          # download BTC/USDT 1h + 1d from Binance
python main.py --mode features       # build 73-feature matrix
python main.py --mode strategy       # backtest EMA+RSI and Confluence
python main.py --mode train-v2       # train XGBoost v2 with Optuna (GPU)
python main.py --mode compare        # compare all saved models
python main.py --mode wfv            # walk-forward validation (12 folds)
python main.py --mode evaluate       # overfitting check + Monte Carlo
python main.py --mode report         # generate markdown report
```

Forex and commodities use the same mode structure:
```bash
cd forex       && python main.py --mode full --trials 30
cd commodities && python main.py --mode full --trials 30 --force
```

---

## Feature Engineering

**73 features** across 7 groups per asset:

| Group | Count | Examples |
|---|---|---|
| Trend | 13 | EMA crosses, MACD hist, ADX |
| Momentum | 12 | RSI-14, Stochastic, Williams %R |
| Volatility | 10 | Bollinger bands, ATR %, Keltner |
| Volume | 7 | Volume ratio, OBV, VWAP gap, MFI |
| Price-Derived | 8 | Log return, realised vol (24h/168h) |
| Macro 1d | 9 | Daily RSI, ATR %, BB position, ADX |
| Patterns | 15 | Doji, hammer, engulfing, harami |

**Zero-volume handling:** Forex pairs (AUDUSD=X) report 0 volume via yfinance. Volume-based indicators are automatically replaced with price proxies (ATR ratio, price-direction OBV, typical-price EMA, RSI of typical price).

**Target label:** 3-class `{-1: SHORT, 0: HOLD, 1: LONG}` based on forward return over 12h vs a per-asset threshold (±1.5% crypto, ±0.2% forex, ±0.5% gold).

---

## XGBoost v2 — Anti-Overfit Design

Standard XGBoost memorises the training set completely (astronomical in-sample returns). v2 addresses this with:

1. **Stationary features only** — removes 17 absolute price-level features (`ema_50`, `obv`, `vwap`, etc.) that change meaning across price regimes. Keeps 56 dimensionless features (RSI, ATR%, BB%, crossover flags).

2. **Tighter regularisation** — `max_depth ≤ 6`, `max_leaves ≤ 64`, `min_child_weight ≥ 10`, stronger L1/L2.

3. **Overfitting penalty in Optuna** — objective penalises models where `train_F1 - val_F1 > 0.15`.

The overfitting check uses Sharpe ratio and win rate (not total return) to assess generalisation:
- Train Sharpe 14.45 → Test Sharpe 8.56 (ratio 0.59 — passes ≥ 0.5 threshold)
- Train Win Rate 96.8% → Test Win Rate 96.0% (ratio 0.99)

---

## Data Sources

| Asset | Source | Symbol | History |
|---|---|---|---|
| BTC/USDT | Binance via CCXT | `BTC/USDT` | 2020 → present (1h) |
| AUD/USD | Yahoo Finance | `AUDUSD=X` | ~730 days (1h yfinance cap) |
| Gold | Yahoo Finance | `GLD` (ETF) | 2024 → present (1h) |

> Gold uses `GLD` (SPDR Gold ETF) rather than `GC=F` (Gold Futures). yfinance caps `GC=F` at ~60 days of hourly data; `GLD` has full history and 0.01% tracking error vs spot.

---

## Windows — Critical Notes

Importing PyTorch in a fresh process alongside TensorFlow causes a fatal `0xC0000005` ACCESS_VIOLATION (not catchable by try/except). The entire Python process dies silently.

**Rules enforced in this codebase:**
- All `torch`, `stable_baselines3`, and `LSTMTradingModel` imports are **inside functions**, never at module level
- `step_gpu_check()` is only called from training modes (`train`, `train-v2`, `train-lstm`, `train-rl`)
- `compare`, `wfv`, `report`, `strategy`, `features` modes never import PyTorch

**Diagnostic:**
```bash
cd crypto
python test_imports.py 2>&1    # stops printing at the crashing import
python main.py --mode X 2>&1   # captures stderr for hidden errors
```

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| 2yr yfinance 1h cap for forex/gold | XGBoost can't learn with limited data | Use EMA+RSI (rule-based) for forex |
| LSTM 0 SHORT predictions | LSTM underperforms on HOLD-heavy crypto data | XGBoost v2 is the primary model |
| GLD 50% annualised vol | Inflated by 2yr sample; real Gold vol ~15-18% | Will normalise as history grows |
| Commodities: 4 trades in test | Confluence result is statistically thin | Requires more data to validate |

---

## Replication

A complete step-by-step replication guide is available at `TRADING_BOT_REPLICATION_GUIDE.md`. It covers:
- Every architectural decision with the reasoning
- All known bugs and their exact fixes
- Phase-by-phase run commands
- File placement reference for all 30+ source files

---

## Tech Stack

| Layer | Library |
|---|---|
| Data fetching | `ccxt` (crypto), `yfinance` (forex/commodities) |
| Feature engineering | `ta`, `pandas`, `numpy` |
| ML — gradient boosting | `xgboost` 3.2.0 (CUDA) + `optuna` |
| ML — sequence model | `torch` 2.3.0 (LSTM) |
| ML — reinforcement learning | `stable-baselines3` (PPO) + `gymnasium` |
| Storage | `pyarrow` parquet |
| Experiment tracking | `mlflow` + JSON fallback |
| Logging | `loguru` |