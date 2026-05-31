# MT5 Integration — Live Trading Module

Connects the trained ML models and strategies from this project to MetaTrader5 for live execution. Reuses the existing feature pipeline and trained XGBoost v2 model with zero retraining.

---

## Architecture

```
mt5_integration/
├── config.py          ← symbols, risk params, model paths
├── data_feed.py       ← fetch live OHLCV bars from MT5
├── signal_engine.py   ← load models + run feature pipeline → signal
├── risk_manager.py    ← ATR-based lot sizing, drawdown guard
├── executor.py        ← place/close MT5 orders
└── live_trader.py     ← main hourly loop
```

**Data flow per cycle:**
```
MT5 bars → feature pipeline → XGBoost v2 / EMA+RSI → signal
→ ATR stop size → lot size → MT5 order_send()
```

---

## Strategy per Asset

| Asset | Strategy | Basis |
|---|---|---|
| BTCUSD | XGBoost v2 | WFV 100% profitable folds, Sharpe 8.64 on test |
| AUDUSD | EMA+RSI (rule-based) | ML had no edge on 2yr forex data |
| XAUUSD | XGBoost v2 | Positive test result, pure argmax threshold |

---

## Prerequisites

### 1. Install MetaTrader5 Python package

```bash
pip install MetaTrader5
```

### 2. MetaTrader5 terminal must be running and logged in

The Python API connects to your locally installed MT5 terminal. It cannot connect to a remote terminal. Make sure:
- MT5 desktop app is open
- You are logged into a live or demo account
- **Tools → Options → Expert Advisors → Allow automated trading** is checked
- **Allow DLL imports** is checked

### 3. Enable Python API in MT5

In MT5: **Tools → Options → Expert Advisors**  
- ✅ Allow automated trading  
- ✅ Allow DLL imports  
- ✅ Allow imports of external experts  

### 4. Trained models must exist

```
crypto/models/xgboost_v2/model.json       ← BTCUSD
commodities/models/xgboost_v2/model.json  ← XAUUSD
```

Run training first if not done:
```bash
cd crypto && python main.py --mode train-v2 --trials 50
cd ..\commodities && python main.py --mode train-v2 --trials 30
```

---

## Configuration

Edit `config.py` before running:

### Symbol names (broker-specific)

Different brokers use different naming conventions. Check your MT5 Market Watch panel for exact names.

```python
ASSETS = {
    "BTCUSD": {
        "mt5_symbol": "BTCUSD",    # some brokers: "BTCUSDT", "Bitcoin", "#BTC"
        ...
    },
    "AUDUSD": {
        "mt5_symbol": "AUDUSD",    # usually standard
        ...
    },
    "XAUUSD": {
        "mt5_symbol": "XAUUSD",    # some brokers: "GOLD", "XAUUSDm"
        ...
    },
}
```

### Risk parameters

```python
# Per-asset risk (fraction of equity per trade)
"risk_pct": 0.01      # 1% for BTC
"risk_pct": 0.005     # 0.5% for AUD/USD
"risk_pct": 0.008     # 0.8% for Gold

# Stop loss and take profit (multiples of ATR)
"stop_atr_mult": 2.0  # stop = 2× ATR from entry
"tp_atr_mult":   4.0  # take-profit = 4× ATR from entry

# Portfolio-level limits (in config.py globals)
MAX_TOTAL_RISK_PCT = 0.03   # never exceed 3% total open risk
MAX_DRAWDOWN_PCT   = 0.10   # halt trading if session drawdown hits 10%
MIN_EQUITY         = 500.0  # minimum equity to trade (USD)
```

---

## Running

### Dry-run first (recommended)

Generates signals and logs what orders *would* be placed — no real money:

```bash
cd mt5_integration
python live_trader.py --dry-run
```

### Single test cycle

Runs one cycle immediately and exits — useful for verifying data feed and signal logic:

```bash
python live_trader.py --once --dry-run
```

### Live trading

```bash
python live_trader.py
```

The bot:
1. Runs a cycle immediately on startup
2. Waits for the next hourly bar close (with a 5-second buffer)
3. Repeats until stopped with `Ctrl+C` or drawdown limit hit

---

## What Happens Each Hour

```
1. Fetch last 300× 1h bars + 300× 1d bars per asset from MT5
2. Run TechnicalFeatures (73 indicators) + ChartPatternFeatures
3. Merge daily macro features via reindex(ffill)
4. XGBoost v2: predict_proba → argmax → {-1, 0, 1}
   EMA+RSI:    generate_signals → {-1, 0, 1}
5. Compare signal to current position:
   - Same direction → hold (no action)
   - Changed direction → close existing, open new
   - HOLD signal → close if open, stay flat
6. ATR-based lot size: risk_amount / (stop_distance / tick × tick_value)
7. MT5 market order with SL and TP attached
```

---

## Position Management Rules

| Situation | Action |
|---|---|
| HOLD signal, no position | Stay flat |
| LONG signal, no position | Open BUY |
| LONG signal, already LONG | Hold (no action) |
| HOLD signal, position open | Close position |
| LONG signal, currently SHORT | Close SHORT, open LONG |
| Drawdown ≥ 10% from session start | Halt all trading |

---

## Logs

All logs saved to `mt5_integration/logs/live_trader_YYYY-MM-DD.log`

A new log file is created each day. Logs include:
- Every signal generated (direction, confidence, price, ATR)
- Every order placed (symbol, direction, lots, entry, SL, TP, risk $)
- Every position closed (PnL)
- Cycle summaries (equity, open positions)

---

## Integration with Your Visual Studio App

If your existing VS app already has an MT5 connection, you can call the signal engine directly without running `live_trader.py` as a standalone script:

```python
# In your existing app
import sys
sys.path.insert(0, r"C:\path\to\Trading-Bot\mt5_integration")
sys.path.insert(0, r"C:\path\to\Trading-Bot\crypto")

from signal_engine import SignalEngine
from data_feed import MT5DataFeed

feed   = MT5DataFeed()          # uses your existing MT5 connection
engine = SignalEngine()          # loads XGBoost v2 models once

# Call this every hour on bar close
df_1h, df_1d = feed.fetch_pair("BTCUSD")
signal = engine.generate("BTCUSD", df_1h, df_1d)

print(signal.direction)     # -1, 0, or 1
print(signal.confidence)    # XGBoost max class probability
print(signal.atr_pct)       # for stop sizing
```

---

## Known Limitations

| Issue | Notes |
|---|---|
| Symbol name mismatch | Every broker uses different names. Check Market Watch. |
| AUDUSD: yfinance data was used for training | MT5 data may have slightly different OHLC due to broker feed differences. Signal quality should be similar. |
| GLD (ETF) ≠ XAUUSD (spot) | Training used GLD ETF (~$220), MT5 trades XAUUSD spot (~$2200). ATR values will differ by ~10×. The model's directional signal still applies; only stop sizing needs adjustment. The `atr_pct` feature handles this automatically since it's a ratio. |
| 1-hour execution delay | Signals are computed on the *completed* bar. There is always 1 bar of delay vs optimal entry. |
| Broker spread and swap | Add broker's typical spread to commission assumption. Overnight positions incur swap charges not modelled in backtests. |