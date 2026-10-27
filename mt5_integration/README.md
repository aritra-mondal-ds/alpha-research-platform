# [DEPRECATED / ARCHIVED] Legacy MT5 Execution Prototype

> ⚠️ **ARCHITECTURAL NOTICE**: This directory contains the legacy, single-process synchronous prototype for MetaTrader 5 execution.
>
> In accordance with institutional quant infrastructure standards, execution and order-routing have been decoupled and promoted to the standalone repository:
>
> 🚀 **[`mt5-execution-gateway`](https://github.com/aritra-mondal-ds/mt5-execution-gateway)**

---

## Component Migration Matrix

| Legacy Component (`mt5_integration/`) | Production Component (`mt5-execution-gateway`) | Architectural Enhancements |
|:---|:---|:---|
| [`live_trader.py`](live_trader.py) | `main.py` (`TradingSystem`) | Migrated from blocking `time.sleep()` loop to asynchronous, non-blocking `asyncio` event loop with clean SIGINT/SIGTERM lifecycle shutdown. |
| [`data_feed.py`](data_feed.py) | `feeds/mt5_feed.py` & `feeds/reconnect.py` | Added thread-pool offloading for MT5 IPC calls, exponential backoff reconnection, and connection health pinging. |
| [`signal_engine.py`](signal_engine.py) | `features/pipeline.py` & `strategies/ml_strategy.py` | Migrated from slow Pandas iterative loops to sub-millisecond Polars feature pipelines and vectorized decision-tree inference. |
| [`risk_manager.py`](risk_manager.py) | `risk/engine.py` | Added persistent peak-baseline drawdown breaker (`DrawdownBreaker`), portfolio-level exposure caps, and dynamic ATR position sizing. |
| [`executor.py`](executor.py) | `execution/mt5_broker.py` & `execution/reconciler.py` | Added spread-blowout protection guards, fill-mode fallback (IOC $\rightarrow$ RETURN $\rightarrow$ FOK), and bidirectional state reconciliation (ghost/phantom detection). |
| [`mt5_config.py`](mt5_config.py) | `config/settings.py` & `config/live_params.json` | Replaced hardcoded constants with Pydantic validated settings and hot-reloadable parameter configurations without bot restarts. |

---

## Status
* **Status:** Archived & Deprecated for active trading.
* **Preservation:** Retained in this repository strictly for historical auditability and development reference.
* **Production Deployment:** Use [`mt5-execution-gateway`](https://github.com/aritra-mondal-ds/mt5-execution-gateway) for paper and live execution.

