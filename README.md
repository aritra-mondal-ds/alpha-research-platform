# Quantitative Alpha Research Platform (`alpha-research-platform`)

> An institutional-grade quantitative research, feature engineering, and machine learning platform for multi-asset statistical arbitrage and directional alpha generation. Designed for rigorous out-of-sample validation, zero-leakage cross-validation, and seamless model serialization for live execution.

**Tech Stack:** Python 3.11 · LightGBM · XGBoost · Optuna · PyTorch · Marcos López de Prado Methodologies · Apache Arrow / Parquet · MLflow · Loguru

---

## Executive Summary & System Capabilities

This repository serves as the **offline quantitative research and alpha modeling laboratory**, decoupled from the low-latency live execution gateway (`mt5-execution-gateway`). It is architected around institutional quant principles: scale-invariant stationary features, dynamic volatility-scaled labeling, strict boundary purging, and realistic continuous-return backtesting.

| Domain | Institutional Quantitative Engineering Standard |
|---|---|
| **Data Engineering** | Multi-source ingestion (Binance via CCXT, Yahoo Finance, MetaTrader 5 tick history), UTC-aligned timestamps, multi-timeframe forward-fill without lookahead, compressed Parquet caching |
| **Feature Engineering** | 73 reproducible features; automatic zero-volume synthetic proxies for FX; automated stationarity filtering removing price-level biases |
| **Labeling Methodology** | **Triple-Barrier Method** (Marcos López de Prado) scaling upper/lower barriers dynamically with local ATR/volatility ($\sigma_t$) |
| **Cross-Validation** | **Purged & Embargoed Walk-Forward Cross-Validation** (eliminating label overlap leakage across train/validation/test folds) |
| **Model Architectures** | **LightGBM** (CPU-optimized, histogram binning, zero platform dependency) & **XGBoost** (GPU-accelerated, anti-overfitting regularizers) |
| **Backtesting Engine** | Vectorized, continuous-return stream including cash holdings, dynamic slippage modeling, and proportional transaction costs |
| **Model Serialization** | Standardized model metadata export (`model.txt` / `model.json` + `meta.json` schema) for real-time inference in MT5 |

---

## Quantitative Research Architecture

```
alpha-research-platform/
├── main.py                        ← Top-level multi-asset research orchestrator
├── shared/                        ← Base fetchers, portfolio allocator, math utilities
├── crypto/                        ← Core quantitative research pipeline (BTC/USDT)
│   ├── config/settings.py         ← Centralized hyperparameters, paths, and risk constraints
│   └── src/
│       ├── data/                  ← Ingestion (CCXT, MT5), loader, temporal splitter
│       ├── features/
│       │   ├── pipeline.py        ← End-to-end feature orchestration & daily macro merge
│       │   ├── technical.py       ← 73 technical indicators with zero-volume fallbacks
│       │   ├── patterns.py        ← Vectorized candlestick pattern recognition
│       │   └── target.py          ← Dynamic Triple-Barrier Method & continuous return targets
│       ├── models/
│       │   ├── feature_selector.py← Stationarity analysis & dimensionless feature filtering
│       │   ├── lightgbm_model.py  ← Production CPU-first LightGBM classifier with Optuna
│       │   ├── xgboost_model.py   ← XGBoost baseline with confidence thresholding
│       │   ├── xgboost_v2.py      ← Anti-overfitting regularized XGBoost
│       │   └── lstm_model.py      ← PyTorch sequence model (isolated lazy evaluation)
│       └── backtesting/
│           ├── simple_backtest.py ← Vectorized backtest engine (continuous return stream)
│           ├── walk_forward.py    ← Purged & Embargoed Walk-Forward Cross-Validation
│           └── metrics.py         ← Institutional risk-adjusted metrics (Sharpe, Sortino, Omega)
├── forex/                         ← Forex research module (AUD/USD)
└── commodities/                   ← Commodities research module (XAU/USD)
```

---

## Quant Engineering Deep-Dive: Methodologies & Pitfalls Solved

### 1. The Continuous-Return Sharpe Ratio Standard
* **Pitfall:** Naive backtesters frequently evaluate `returns.replace(0, np.nan).dropna()` before computing Sharpe ratios. When a strategy sits in cash, dropping zero-return periods artificially compresses return variance and scales active hours as if they occurred continuously across all 8,760 hours of the year. This produces astronomical, synthetic Sharpe ratios ($> 8.0$) that instantly collapse in production.
* **Quant Standard:** All metrics in this platform evaluate the **full continuous investment timeline**, treating cash periods as true zero-return bars:
  $$\text{Sharpe} = \frac{\mathbb{E}[R_t] - R_f}{\sigma(R_t)} \cdot \sqrt{N_{\text{annual}}}$$
  where $R_t$ includes all active and flat holding periods.

### 2. Marcos López de Prado Triple-Barrier Labeling
* **Pitfall:** Fixed percentage targets (e.g., $+1.5\%$ / $-1.5\%$) fail across market regimes. During low-volatility consolidation, barriers are unreachable; during high-volatility expansions, barriers are triggered by ordinary noise.
* **Quant Standard:** The `TargetBuilder` implements dynamic volatility-scaled barriers:
  $$U_t = P_t \cdot (1 + k_{\text{pt}} \cdot \sigma_t), \quad L_t = P_t \cdot (1 - k_{\text{sl}} \cdot \sigma_t)$$
  The label is assigned based on whether the price hits the upper barrier (+1), lower barrier (-1), or vertical time horizon (0) first, matching real-world trade exits.

### 3. Purged & Embargoed Walk-Forward Cross-Validation
* **Pitfall:** Standard rolling walk-forward splits cause **label leakage**. If a label looks $H$ bars into the future, samples within the final $H$ bars of the training set are labeled using prices that occurred inside the test set.
* **Quant Standard:** `WalkForwardValidator` enforces strict **purging**:
  $$\text{Train}_{\text{purged}} = \text{Train}_{0 \,:\, T_{\text{end}} - H}$$
  The boundary between internal validation sets for early stopping is likewise purged, guaranteeing zero lookahead contamination.

### 4. Stationarity Enforcement
* **Pitfall:** Feeding absolute price-level indicators (`ema_50`, `vwap`, `bb_upper`) into tree models causes memorization of price levels rather than regime dynamics.
* **Quant Standard:** `get_stationary_features()` strips all absolute price indicators, restricting models to dimensionless, scale-invariant metrics (ratios, normalized oscillator ranges $[0, 100]$, rate of change, and ATR percentages).

---

## Model Architectures & Optimization

### A. LightGBM (CPU-First Baseline)
* **Optimization:** Histogram-based gradient boosting running on all available CPU threads (`n_jobs=-1`). Avoids Windows CUDA driver/DLL conflicts while achieving sub-millisecond per-bar inference.
* **Regularization:** Bounded leaves (`num_leaves ≤ 63`), L1 penalty (`reg_alpha`), L2 penalty (`reg_lambda`), and subsampling to control tree complexity.
* **Export:** Saves binary model graph and explicit `meta.json` feature manifest for direct consumption by the execution gateway.

### B. XGBoost v2 (GPU-Accelerated)
* **Optimization:** GPU histogram construction via `tree_method='hist'`.
* **Objective:** Multi-class log-loss over inverse-frequency weighted samples to combat market class imbalance (preponderance of HOLD bars).

---

## Documented Research Post-Mortem: 5-Minute Scalping Failure

In the spirit of institutional quantitative research, negative results and regime failures are thoroughly documented:

* **Experiment:** A 5-minute scalping strategy was deployed on live demo data without training an asset-specific 5m ML model, relying instead on an EMA(9/21) + RSI rule-based fallback.
* **Outcome:** Suffered a **-48.7% drawdown** over 8 trading sessions.
* **Root Cause Analysis:** Fast-timeframe moving average crossovers at 5-minute granularity capture market microstructure noise and bid-ask bounce rather than directional trend. With transaction costs and spread widening, break-even required a $>60\%$ win rate, whereas the strategy realized $\sim 25\%$.
* **Corrective Action:** Timeframe transferability was formally rejected. Sub-hourly strategies now require independent volatility recalibration, separate feature matrices, and full purged walk-forward validation before consideration.

---

## Quickstart & Replication

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/aritra-mondal-ds/TradingBot.git qresearch-engine
cd qresearch-engine

# Create virtual environment
conda create -n quant-research python=3.11 -y
conda activate quant-research

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipeline

```bash
cd crypto

# 1. Download historical bars from Binance
python main.py --mode fetch

# 2. Compute 73-feature matrix with Triple-Barrier labels
python main.py --mode features

# 3. Train CPU LightGBM model with Optuna hyperparameter optimization
python main.py --mode train-lgb --trials 50

# 4. Train GPU XGBoost model
python main.py --mode train-v2 --trials 50

# 5. Run Purged Walk-Forward Cross-Validation
python main.py --mode wfv --trials 20

# 6. Evaluate out-of-sample test performance & Monte Carlo simulation
python main.py --mode evaluate

# 7. Multi-model comparative tear-sheet
python main.py --mode compare
```

---

## Live Execution Bridge

Models trained in this research repository export serializable artifacts (`model.txt` / `model.json` + `meta.json`) into `models/`. These are ingested directly by the sister repository:

👉 **[`mt5-execution-gateway`](../TradingBotMT5)**: Production asynchronous MetaTrader 5 execution engine featuring live socket reconnection, broker position reconciliation, and drawdown circuit breakers.
