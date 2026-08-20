# Multi-Asset ML Trading Bot

> End-to-end machine learning system spanning data ingestion, feature engineering, model training, validation, and live execution across three asset classes. Built as a portfolio project to demonstrate production-grade ML engineering practices.

**Tech stack:** Python 3.11 · XGBoost · Optuna · PyTorch (LSTM) · Stable-Baselines3 (PPO) · CCXT · MetaTrader5 · Parquet · MLflow · Loguru

---

## What This Project Covers

| Stage | What was built |
|---|---|
| **Data engineering** | Multi-source ingestion (Binance via CCXT, Yahoo Finance, MetaTrader5), incremental parquet caching, validation |
| **Feature engineering** | 73 reproducible features across 7 groups; automatic zero-volume handling for FX assets |
| **Model development** | XGBoost v2 with anti-overfit design, Optuna hyperparameter search, LSTM, PPO RL |
| **Validation** | Walk-forward validation across 12 folds, Monte Carlo simulation, train/val overfitting checks |
| **Live deployment** | MetaTrader5 broker integration: order execution, ATR-based sizing, deal reconciliation, error handling |
| **Iteration** | Documented failure (5m scalping), root-cause analysis, revert, and recovery — the full feedback loop |

---

## Backtested Results

### BTC/USDT — XGBoost v2

| Metric | XGBoost v2 | Buy & Hold |
|---|---|---|
| Total Return | **+695.74%** | -15.30% |
| Sharpe Ratio | **8.642** | — |
| Max Drawdown | **-11.44%** | -62%+ |
| Win Rate | **94.4%** | — |
| Trades (test set) | 54 | — |

Test period: Feb 2025 → May 2026 (out-of-sample, never seen during training).

### Walk-Forward Validation — 12 Folds (2020–2026)

The model was retrained and re-evaluated on rolling 6-month windows across 8 years, including the 2022 bear market and 2023–2025 recovery:

| Strategy | Mean Sharpe | Profitable Folds |
|---|---|---|
| **XGBoost v2** | **+10.877** | **12 / 12 (100%)** |
| Confluence (rule-based) | -0.630 | 4 / 12 |
| EMA+RSI (rule-based) | -1.799 | 5 / 12 |

### Monte Carlo Simulation — 1,000 Runs

| Percentile | Return |
|---|---|
| 5th (worst case) | +346% |
| Median | +531% |
| 95th (best case) | +921% |
| **Probability of loss** | **0.0%** |

### Cross-Asset Summary

| Asset | Strategy | Return | Sharpe |
|---|---|---|---|
| BTC/USDT | XGBoost v2 | **+695.74%** | 8.64 |
| AUD/USD | EMA+RSI | +1.58% | 2.10 |
| XAU/USD | Confluence | +4.39% | 12.49 |

---

## Architecture

```
Trading-Bot/
├── main.py                        ← multi-asset orchestrator
├── shared/                        ← base classes, portfolio allocator, MT5 fetcher
├── crypto/                        ← BTC/USDT — full ML pipeline
│   └── src/
│       ├── data/                  ← CCXT fetcher, parquet loader, temporal splitter
│       ├── features/              ← 73-feature pipeline (technical + patterns + macro)
│       ├── strategies/            ← EMA+RSI, Confluence rule-based baselines
│       ├── backtesting/           ← vectorised backtester, WFV, metrics, Monte Carlo
│       ├── models/                ← XGBoost v1/v2, LSTM, PPO reinforcement learning
│       └── utils/                 ← GPU checks, MLflow logging, report generator
├── forex/                         ← AUD/USD — shares crypto feature pipeline
├── commodities/                   ← XAU/USD — shares crypto feature pipeline
└── mt5_integration/               ← live broker integration
    ├── mt5_config.py              ← per-asset config (symbols, risk params, model paths)
    ├── data_feed.py               ← live OHLCV bars + broker symbol info from MT5
    ├── signal_engine.py           ← runs feature pipeline + model inference on live data
    ├── risk_manager.py            ← ATR position sizing + broker minimum stop enforcement
    ├── executor.py                ← order placement, fill-mode retry, deal reconciliation
    └── live_trader.py             ← hourly event loop with UTC logging + startup audit
```

The feature pipeline, backtesting engine, and models all live in `crypto/src/` and are reused without modification by the forex and commodities modules.

---

## Feature Engineering — 73 Features

Features are engineered in seven groups and are designed to be **stationary** (dimensionless ratios, not absolute price levels) so they generalise across different price regimes without leakage.

| Group | Count | Examples |
|---|---|---|
| Trend | 13 | EMA crossover flags, MACD histogram, ADX |
| Momentum | 12 | RSI-14, Stochastic, Williams %R, CCI |
| Volatility | 10 | Bollinger Band %, ATR %, Keltner Channel % |
| Volume | 7 | Volume ratio, VWAP gap, OBV rate of change, MFI |
| Price-derived | 8 | Log return, realised volatility (24h / 168h), high-low spread |
| Macro (1d) | 9 | Daily RSI, daily ATR %, daily BB position, daily ADX |
| Patterns | 15 | Doji, hammer, engulfing, harami, shooting star |

**Zero-volume assets:** Yahoo Finance reports zero tick volume for FX pairs. Volume-based indicators are automatically replaced with price-derived proxies (ATR ratio, direction-based OBV, typical-price RSI) with no manual intervention required.

**Stationarity enforcement:** 17 absolute-price features (`ema_50`, `obv`, `vwap`, etc.) are dropped before training. Only the 56 dimensionless features survive, ensuring the model doesn't overfit to specific price levels from the training period.

---

## Model Design — XGBoost v2 (Anti-Overfit)

Naive XGBoost on financial data memorises the training set — it can produce Sharpe ratios above 30 in-sample that collapse to negative out-of-sample. The v2 design addresses this with three constraints:

**1. Stationary features only**
The `FeatureSelector` drops all 17 non-stationary features before any model sees the data, forcing the model to learn from regime-invariant signals rather than price-level patterns that won't repeat.

**2. Tighter regularisation bounds**
Optuna search space is constrained: `max_depth ≤ 6`, `max_leaves ≤ 64`, `min_child_weight ≥ 10`, and stronger L1/L2 ranges — bounds chosen specifically to prevent individual tree memorisation.

**3. Overfit penalty in the Optuna objective**
```python
# Penalise if train F1 is more than 15pp above val F1
gap_penalty = max(0, train_f1 - val_f1 - 0.15) * 2.0
objective   = val_f1 - gap_penalty
```
Trials that overfit are automatically deprioritised regardless of validation accuracy.

**Overfit check (train vs test)**

| Metric | Train | Test | Ratio |
|---|---|---|---|
| Sharpe | 14.45 | 8.56 | 0.59 ✅ |
| Win Rate | 96.8% | 96.0% | 0.99 ✅ |

A train/test Sharpe ratio ≥ 0.50 and win rate ratio ≥ 0.90 are required to declare the model deployment-ready.

---

## Validation Methodology

Three layers of validation before any live deployment:

**Temporal train/val/test split** — 70% train, 15% validation (Optuna target), 15% test (held out until final evaluation). No data leakage.

**Walk-forward validation (WFV)** — 12 rolling folds across 2020–2026. The model is retrained from scratch on each expanding window and evaluated on the next 6-month period. A model must achieve positive Sharpe in at least 10/12 folds and pass the overfit check on each fold before being considered for live deployment.

**Monte Carlo simulation** — 1,000 bootstrap runs shuffling trade order to estimate the full distribution of possible outcomes, not just the point estimate from the test set.

This layered approach was what caught the XAU/USD model's weakness: while its single test-set return looked acceptable, its WFV fold count and train/val F1 gap (0.32 vs the 0.15 threshold) flagged it as overfit — subsequently confirmed in live trading.

---

## Live Trading — MT5 Integration

The trained models feed into a MetaTrader5 broker connection for live execution. The live system reuses the same feature pipeline used during training, with no code duplication.

### Strategy routing per asset

| Asset | MT5 Symbol | Strategy | Confidence threshold |
|---|---|---|---|
| BTC | `#BTCUSD` | XGBoost v2 (1h) | 0.45 — forces HOLD below this |
| AUD/USD | `AUDUSD` | EMA(50)+RSI(75), long-only | Rule-based, no threshold |
| Gold | `XAUUSD` | XGBoost v2 (1h) | 0.0 — pure argmax, sparse signal |

### Hourly execution cycle

```
MT5 bars (1h + 1d) → feature pipeline → model inference
→ compare to current position → ATR-based sizing → MT5 order
```

### Risk controls

- ATR-based stop loss and take profit (parameterised per asset)
- Broker minimum stop distance enforced before every order (`stops_level × point × buffer`)
- Max 1% equity per trade, 3% total open risk, 5% session drawdown halt
- One position per asset at a time
- Fill mode auto-detection: tries `ORDER_FILLING_RETURN → FOK → IOC`, caches result per symbol
- Weekend market closure guard for FX/Gold assets
- Startup audit: lists all open positions with open time on every restart

### Deal history reconciliation

MT5 broker-triggered stop-loss/take-profit hits are invisible to `order_send()` — the broker closes the position directly. A `get_closed_deals()` call against `history_deals_get()` runs every cycle, surfacing these broker-side closes in the logs with their actual P&L so the true win rate is always visible.

---

## Live Performance

### 5m scalping experiment — documented failure

After initial hourly deployment, a 5-minute scalping strategy was tested to increase trade frequency. This was deployed **without first training a 5m ML model** — a rule-based EMA(9/21)+RSI fallback was used instead. The result was a **-48.7% drawdown over 8 days** ($81,128 → $41,615).

| Day | P&L | Win Rate |
|---|---|---|
| 1 (before broker fixes) | -$2,397 | — |
| 2 | -$973 | 54% |
| 3 | -$622 | 44% |
| 4 | -$1,446 | 22% |
| 5 | -$1,136 | 0% |
| 6 | -$1,218 | 22% |
| 7 | -$68 | 50% |

**Root cause:** EMA(9/21) crossovers at 5-minute resolution generate noise, not signal. The strategy required ~60% win rate to break even given its stop/take-profit ratio; it achieved ~25% on average. Filters added mid-experiment (ADX trend filter, session-hour guard, daily trend filter) reduced trade count but did not fix the win rate.

5m XGBoost models were trained afterward for completeness — none were deployed:

| Asset | Test Return | Test Sharpe | Issue |
|---|---|---|---|
| BTC 5m | -88.87% | -3.31 | Negative |
| AUD/USD 5m | -0.51% | -0.13 | No edge |
| XAU/USD 5m | -7.20% | -8.54 | Overfit (F1 gap 0.32) |

**Learning:** a new timeframe requires its own full validation cycle. The 1h model's validated edge does not transfer to 5m without retraining, recalibrating targets, and re-running walk-forward validation.

### Post-revert — 1h XGBoost v2 restored

After reverting to the validated 1h strategy, live performance returned to expected behaviour:

| Asset | Live win rate | Backtest win rate |
|---|---|---|
| BTC/USD | **75–95%** across observed sessions | 94.4% |
| AUD/USD | Positive after long-only clamp bug fixed | +1.58% / Sharpe 2.10 in backtest |
| XAU/USD | Near breakeven, thin trade count | Retrained model flagged overfit — sized down |

---

## Bugs Found and Fixed in Production

Documenting these because they represent the kind of broker-specific knowledge that only emerges from live testing — not from backtests or tutorials:

| Bug | Symptom | Fix |
|---|---|---|
| Wrong fill mode | All orders rejected, `retcode=10030` | Fill-mode retry loop: `RETURN → FOK → IOC`, cached per symbol |
| Stop too tight | Orders rejected, `retcode=10016` | Read `stops_level × point` from broker, widen stop with 2× buffer |
| Long-only strategy going short | `EMACrossoverRSI(long_only=True)` returns -1 (exit), bot interprets as open short | `direction = max(0, raw)` clamp |
| Weekend stale data | FX signals generated from Friday's frozen bars all weekend | `_is_fx_weekend_closed()` check; HOLD if FX market is closed |
| Broker SL/TP hits invisible in logs | Position closed by broker, no log entry, P&L undercounted | `history_deals_get()` reconciliation each cycle |
| `model_dir` override silently ignored | 5m model saved over validated 1h model path | Override via attribute after `__init__`, verify actual saved path after every run |
| Log timestamps in local time | 2-hour mismatch between log prefix and UTC cycle banner | `loguru.configure(patcher=_utc_patcher)` forces UTC everywhere |
| Server timeout not retried | `retcode=10012` on single transient failure | Up to 2 retries with 2s pause before giving up |

---

## Setup

### Requirements

- Python 3.11.x (tested on 3.11.15)
- CUDA 12.1 GPU (optional — XGBoost also runs on CPU)
- MetaTrader5 terminal (live trading only)

### Install

```bash
conda create -n tradingbot python=3.11.15
conda activate tradingbot

pip install pandas numpy pyarrow xgboost optuna ccxt yfinance ta loguru mlflow
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium
pip install MetaTrader5    # live trading only
```

> Use `ta` not `pandas-ta` — `pandas-ta` is incompatible with Python 3.11.

### Run the full pipeline

```bash
# Fetch + train + validate + report (crypto, ~60 min with GPU)
cd crypto && python main.py --mode full --trials 50

# Cross-asset comparison
cd .. && python main.py --mode compare-all

# Live trading (MT5 must be open and logged in)
cd mt5_integration
python live_trader.py --once --dry-run   # verify, no orders
python live_trader.py                    # live
```

### Individual steps

```bash
cd crypto
python main.py --mode fetch       # download BTC/USDT 1h + 1d from Binance
python main.py --mode features    # build 73-feature matrix, cache to parquet
python main.py --mode train-v2    # XGBoost v2 + Optuna (GPU-accelerated)
python main.py --mode wfv         # walk-forward validation, 12 folds
python main.py --mode evaluate    # overfit check + Monte Carlo
python main.py --mode report      # generate markdown report
```

---

## Data Sources

| Asset | Training source | Live source | Coverage |
|---|---|---|---|
| BTC/USDT | Binance via CCXT | MT5 broker | 2020 → present (1h + 5m) |
| AUD/USD | Yahoo Finance (`AUDUSD=X`) | MT5 broker | ~730 days (yfinance 1h cap) |
| XAU/USD | Yahoo Finance (`GLD` ETF) | MT5 broker (`XAUUSD`) | 2024 → present (1h) |

Gold uses the GLD ETF (not `GC=F` futures) because yfinance caps futures hourly data at ~60 days while GLD provides full history. The bot uses only stationary/ratio features so the ~10× price difference between GLD (~$220) and XAUUSD spot (~$2,200+) does not affect signal quality.

---

## Known Limitations

| Limitation | Impact | Status |
|---|---|---|
| 2yr yfinance 1h cap for FX/Gold | Insufficient data for XGBoost on AUD/USD | EMA+RSI used for AUD/USD; XGBoost would need more data |
| GLD ETF ≠ XAUUSD spot | Different price scale; absolute features would leak | Stationary features sidestep this entirely |
| 4 trades in original XAU/USD test | Result is statistically thin | Flagged throughout; confirmed by weaker live retrain |
| LSTM produces 0 SHORT predictions | HOLD-heavy data means minority class is undertrained | XGBoost v2 is the production model; LSTM is experimental |
| 5m timeframe has no validated model | Live 5m trading lost -48.7% without a validated model | 5m deployment blocked until a model passes WFV |
| MT5 5m history capped at ~16 months | Limits future 5m model training data | Broker limitation; alternative data vendors could fill the gap |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | `ccxt`, `yfinance`, `MetaTrader5` |
| Feature engineering | `ta`, `pandas`, `numpy` |
| ML — gradient boosting | `xgboost` 3.2.0 (CUDA) + `optuna` |
| ML — deep learning | `torch` 2.3.0 (LSTM) |
| ML — reinforcement learning | `stable-baselines3` (PPO) + `gymnasium` |
| Storage | `pyarrow` parquet |
| Experiment tracking | `mlflow` + JSON fallback |
| Logging | `loguru` (UTC-patched) |
| Live execution | MetaTrader5 Python API, HF Markets demo account |

---

## Replication

`TRADING_BOT_REPLICATION_GUIDE.md` contains a full step-by-step walkthrough including every architectural decision with reasoning, all known bugs and their fixes, complete run-command sequences, and a file placement reference for all 30+ source files.