# Multi-Asset ML Trading Bot

A production-grade machine learning trading system for **BTC/USDT**, **AUD/USD**, and **XAU/USD (Gold)**. Fetches live market data, engineers 73 technical and macro features per asset, trains XGBoost and deep learning models, validates results with walk-forward validation across 8 years of data, and trades live via MetaTrader5.

> **Status note:** Backtested results below remain the validated reference. A live-trading section follows with real performance, a costly strategy detour, and the operational lessons that came from it. Read both before treating the backtest numbers as a guarantee of live behaviour.

---

## Results (Backtested)

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

> XAU/USD note: this result came from a 4-trade test window and was always flagged thin. A later live retrain on more recent data produced a materially weaker, overfit model — see [Live Trading Results](#live-trading-results--lessons-learned).

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
├── shared/                   ← base classes, portfolio manager, MT5 historical fetcher
├── crypto/                   ← BTC/USDT (full ML pipeline)
│   └── src/
│       ├── data/             ← CCXT fetcher, loader, splitter
│       ├── features/         ← 73-feature pipeline
│       ├── strategies/       ← EMA+RSI, Confluence
│       ├── backtesting/      ← backtest, WFV, metrics
│       ├── models/           ← XGBoost v1/v2, LSTM, PPO RL
│       └── utils/            ← GPU check, MLflow, report generator
├── forex/                    ← AUD/USD (yfinance for backtest, MT5 for live)
├── commodities/               ← XAU/USD via GLD ETF (backtest), XAUUSD spot (live)
└── mt5_integration/           ← live trading via MetaTrader5
    ├── mt5_config.py          ← symbols, risk params, model paths
    ├── data_feed.py           ← live OHLCV + symbol info from MT5
    ├── signal_engine.py       ← feature pipeline + model/rule inference
    ├── risk_manager.py        ← ATR position sizing, broker stop limits
    ├── executor.py            ← order placement, fill-mode handling, deal reconciliation
    └── live_trader.py         ← hourly trading loop
```

All 73 features, backtesting engine, walk-forward validator, and ML models live in `crypto/src/` and are **shared** by the forex and commodities modules.

---

## Setup

### Requirements

- Python 3.11.x (3.11.15 tested)
- CUDA 12.1 compatible GPU (optional — XGBoost runs on CPU too)
- Windows / Linux
- MetaTrader5 terminal (for live trading only)

### Install

```bash
conda create -n tradingbot python=3.11.15
conda activate tradingbot

pip install pandas numpy pyarrow xgboost optuna ccxt yfinance ta loguru mlflow
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium
pip install MetaTrader5    # live trading only
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

# Live trading (MT5 terminal must be open and logged in)
cd mt5_integration
python live_trader.py --once --dry-run   # verify signals, no real orders
python live_trader.py --dry-run          # full loop, no real orders
python live_trader.py                    # live
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

This check matters in practice — see the 5m model results below, where the gold model failed it (gap 0.32) and was never deployed.

---

## Live Trading (MT5 Integration)

The `mt5_integration/` module connects the trained models to a MetaTrader5 account for live execution, reusing the same feature pipeline used in training.

### Strategy routing

| Asset | Strategy | Basis |
|---|---|---|
| BTC/USD | XGBoost v2 (1h) | WFV 100% profitable, confidence threshold 0.45 |
| AUD/USD | EMA(50)+RSI(75), long-only | ML showed no edge on 2yr FX data in backtest |
| XAU/USD | XGBoost v2 (1h) | Pure argmax (confidence threshold 0.0) — sparse signal |

### Hourly cycle

```
Fetch 1h + 1d bars → feature pipeline → model/rule inference
→ ATR-based stop/TP → lot sizing → compare to open position → order
```

Risk is intentionally conservative: 0.3–1.0% of equity per trade, 3% max total open risk, 5% session drawdown halt, one position per asset at a time.

### Running

```bash
cd mt5_integration
python live_trader.py --once --dry-run   # one cycle, no orders, verify setup
python live_trader.py --dry-run          # full hourly loop, no orders
python live_trader.py                    # live
```

Configuration (broker symbol names, risk per trade, model paths) lives in `mt5_config.py`. Broker symbol names vary — confirm exact names in MT5's Market Watch panel before trading (e.g. this project's broker used `#BTCUSD`, not `BTCUSD`).

---

## Live Trading Results & Lessons Learned

The backtest results above are real and reproducible, but they describe the model in isolation. Putting the system on a live demo account surfaced problems no backtest catches: broker order mechanics, a strategy pivot that didn't work, and gaps in trade visibility.

### Timeline

| Phase | Outcome |
|---|---|
| 1h XGBoost v2, initial live deployment | Working as backtested |
| 5m scalping pivot (8 days) | **-48.7% account drawdown** |
| Reverted to 1h XGBoost v2 | Recovered to backtested behaviour |

### The 5m scalping detour — what happened

In pursuit of higher trade frequency (100+ trades/day), the strategy was switched to 5-minute bars with an EMA(9/21)+RSI(40) rule-based signal, since no 5-minute ML model existed yet. Over 8 trading days the account went from **$81,128 → $41,615**, a **-48.7%** drawdown:

| Day | Result | Win rate |
|---|---|---|
| Day 1 (pre-fixes) | -$2,397 | — |
| Day 2 | -$973 | 54% |
| Day 3 | -$622 | 44% |
| Day 4 | -$1,446 | 22% |
| Day 5 | -$1,136 | 0% |
| Day 6 | -$1,218 | 22% |
| Day 7 | -$68 | 50% |

**Root cause:** EMA(9/21) crossovers on 5-minute bars are dominated by noise, not signal. With a 1.5×ATR stop and 2×ATR take-profit, the strategy needed roughly 60%+ win rate to be profitable; it averaged ~25%. Filters added mid-detour (ADX trend filter, session-hour filter, daily-trend filter) reduced trade count but never fixed the underlying win rate.

5-minute XGBoost models were trained afterward, on the data already gathered, purely to evaluate the idea — none were deployed:

| Asset | Test Return | Test Sharpe | Verdict |
|---|---|---|---|
| BTC 5m | -88.87% | -3.311 | Unusable |
| AUD/USD 5m | -0.51% | -0.126 | Marginal, not deployable |
| XAU/USD 5m | -7.20% | -8.536 | Overfit (train/val F1 gap 0.32) |

**Conclusion:** trade frequency is not a substitute for edge. The 1-hour XGBoost model's edge does not transfer to 5-minute bars; a separate model would need its own full validation cycle (data, threshold calibration, walk-forward) before ever running live — exactly the process the 1h model went through and the 5m detour skipped.

### Current live performance (1h XGBoost v2, post-revert)

| Asset | Multi-day live result | Compares to backtest |
|---|---|---|
| BTC/USD | 75–95% win rate, consistently profitable | Matches backtested 94.4% WR closely |
| AUD/USD | Initially -14% WR (bug, see below), corrected | In line with original +1.58% / Sharpe 2.10 expectation after fix |
| XAU/USD | Near breakeven, thin trade count | Retrained model is weaker than original backtest (+1.62% return, Sharpe 1.105, flagged overfit on a 491-bar test window) — sized down accordingly |

### Operational lessons (worth knowing before running this live)

- **Broker fill modes are not universal.** `ORDER_FILLING_IOC` is common in tutorials but several brokers (this project's included) reject it. The executor now tries `RETURN → FOK → IOC` and caches whichever works per symbol.
- **Brokers enforce a minimum stop distance** (`stops_level`) independent of your ATR calculation. Stops tighter than this are rejected outright (`retcode=10016`); the risk manager now widens stops to the broker's minimum with a safety buffer.
- **Long-only rule-based strategies need an explicit clamp.** `EMACrossoverRSI(long_only=True)` returning `-1` means "exit any long," not "open a short" — left unclamped, this opened real short positions against the strategy's design and cost real money before being caught.
- **FX and Gold markets close on weekends.** MT5 keeps returning the last traded bar when polled on a closed market; without an explicit closed-market check, a bot will compute "fresh" signals from stale Friday data all weekend.
- **Broker-triggered stop-loss/take-profit fills are invisible to your own order-placement code.** If a position closes because price hit its SL, your bot never calls its own "close" function — that exit, and its P&L, simply won't appear in your logs unless you separately reconcile against the broker's deal history.
- **A `model_dir` override set after model instantiation isn't always honoured** — some training classes reset the save path internally during `train()`. Verify the actual saved file path after every training run; this gap silently overwrote a proven production model with an unvalidated one.
- **Log timestamps should be explicit about timezone.** Local-system-time log prefixes next to UTC-based trading logic produces confusing, hard-to-correlate logs during incident review.

---

## Data Sources

| Asset | Source | Symbol | History |
|---|---|---|---|
| BTC/USDT | Binance via CCXT | `BTC/USDT` | 2020 → present (1h), 2020 → present (5m) |
| AUD/USD | Yahoo Finance (backtest) / MT5 (live + 5m research) | `AUDUSD=X` / `AUDUSD` | ~730 days (yfinance 1h cap); ~16 months (MT5 5m, broker-limited) |
| Gold | Yahoo Finance (backtest) / MT5 (live + 5m research) | `GLD` / `XAUUSD` | 2024 → present (1h ETF); ~16 months (MT5 5m, broker-limited) |

> Gold backtest uses `GLD` (SPDR Gold ETF) rather than `GC=F` (Gold Futures) — yfinance caps `GC=F` at ~60 days of hourly data, while `GLD` has full history and 0.01% tracking error vs spot. Live trading uses the broker's `XAUUSD` spot CFD directly; note GLD trades around $220 while XAUUSD spot trades around $2,000+, so absolute price-level features must never be used (see XGBoost v2 stationary-feature design above) — only the ATR%-style ratio features transfer correctly between the two.

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| 2yr yfinance 1h cap for forex/gold | XGBoost can't learn with limited data | Use EMA+RSI (rule-based) for forex |
| LSTM 0 SHORT predictions | LSTM underperforms on HOLD-heavy crypto data | XGBoost v2 is the primary model |
| GLD 50% annualised vol | Inflated by 2yr sample; real Gold vol ~15-18% | Will normalise as history grows |
| Commodities: 4 trades in original test | Confluence result is statistically thin | Requires more data to validate |
| 5-minute timeframe has no validated model | The 1h model's edge does not transfer to 5m; live 5m trading lost -48.7% over 8 days | Stay on 1h until a 5m model passes WFV and overfitting checks, same as the 1h model did |
| Broker-side SL/TP closes aren't auto-logged | Win rate / P&L can appear understated without reconciliation | `executor.get_closed_deals()` queries MT5 deal history each cycle |
| FX/Gold markets close on weekends | Stale MT5 data can produce phantom signals if untreated | Weekend guard in `signal_engine.py` forces HOLD outside trading hours |
| MT5 5m history capped at ~16 months by broker | Limits training data for any future 5m model attempt | Accept the cap, or source 5m data from a different vendor |

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
| Data fetching | `ccxt` (crypto), `yfinance` (forex/commodities backtest), `MetaTrader5` (live + broker history) |
| Feature engineering | `ta`, `pandas`, `numpy` |
| ML — gradient boosting | `xgboost` 3.2.0 (CUDA) + `optuna` |
| ML — sequence model | `torch` 2.3.0 (LSTM) |
| ML — reinforcement learning | `stable-baselines3` (PPO) + `gymnasium` |
| Storage | `pyarrow` parquet |
| Experiment tracking | `mlflow` + JSON fallback |
| Logging | `loguru` (UTC-patched for live trading) |