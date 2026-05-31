# mt5_integration/live_trader.py
"""
High-frequency live trader — 5-minute loop, 100+ trades/day target.

Changes vs hourly version:
    - Loop runs every 5 minutes (was every 1 hour)
    - Multiple positions per asset allowed (max_open_trades from config)
    - Tracks per-asset trade count for daily reporting
    - Displays running P&L and trade stats each cycle

Usage:
    python live_trader.py                 # live trading
    python live_trader.py --dry-run       # signals + sizing, no real orders
    python live_trader.py --once          # one cycle then exit (testing)
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mt5_config    import ASSETS, LOG_DIR, MAX_DRAWDOWN_PCT
from data_feed     import MT5DataFeed
from signal_engine import SignalEngine
from risk_manager  import RiskManager
from executor      import MT5Executor

LOOP_INTERVAL_SECONDS = 300     # 5 minutes = 300 seconds


def _setup_logger():
    logger.remove()
    logger.add(
        sys.stdout, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(LOG_DIR / "live_trader_{time:YYYY-MM-DD}.log"),
        rotation="00:00", retention="30 days", level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def run_cycle(
    feed:         MT5DataFeed,
    engine:       SignalEngine,
    risk_mgr:     RiskManager,
    executor:     MT5Executor,
    trade_counts: dict,
    cycle_num:    int,
) -> dict:
    """Execute one 5-minute signal → execution cycle for all assets."""
    equity  = feed.account_equity()
    actions = {}

    if not risk_mgr.check_drawdown(equity):
        logger.error("Drawdown limit reached — skipping cycle.")
        return {}

    # ── 1. Fetch 5m primary bars + 1d macro for all assets ───────────────────
    live_data = {}
    for symbol, cfg in ASSETS.items():
        try:
            df_primary = feed.fetch(cfg["mt5_symbol"], cfg["primary_tf"], cfg["bars_needed"])
            df_daily   = feed.fetch(cfg["mt5_symbol"], "1d", 300)
            live_data[symbol] = (df_primary, df_daily)
        except Exception as e:
            logger.error(f"{symbol}: Data fetch failed — {e}")

    if not live_data:
        logger.error("No data fetched — skipping cycle.")
        return {}

    # ── 2. Generate signals ───────────────────────────────────────────────────
    signals       = engine.generate_all(live_data)
    open_risk_pct = executor.total_open_risk_pct(equity)

    # ── 3. Process each asset ─────────────────────────────────────────────────
    for symbol, signal in signals.items():
        cfg         = ASSETS[symbol]
        mt5_sym     = cfg["mt5_symbol"]
        max_trades  = cfg.get("max_open_trades", 1)
        open_pos    = executor.get_open_positions(mt5_sym)
        n_open      = len(open_pos)
        current_dir = executor.get_position_direction(mt5_sym)
        desired_dir = signal.direction
        action      = "hold"

        # Close positions that contradict the signal
        if current_dir != 0 and current_dir != desired_dir:
            logger.info(
                f"{symbol}: Signal reversed {_ds(current_dir)}→{_ds(desired_dir)} "
                f"— closing {n_open} position(s)."
            )
            ok     = executor.close_all_for_symbol(mt5_sym)
            action = "closed" if ok else "close_failed"
            if ok:
                trade_counts[symbol] += n_open
            open_pos    = []
            n_open      = 0
            current_dir = 0

        # Open new position if signal is directional and below max_open_trades
        if desired_dir != 0 and n_open < max_trades:
            sym_info = feed.symbol_info(mt5_sym)
            params   = risk_mgr.compute_trade_params(
                symbol=symbol, direction=desired_dir,
                entry_price=signal.current_price, atr_pct=signal.atr_pct,
                equity=equity, symbol_info=sym_info,
                open_risk_pct=open_risk_pct,
            )
            if params is not None:
                ok = executor.open_position(params)
                if ok:
                    action         = f"opened_{_ds(desired_dir)}"
                    open_risk_pct += params.risk_pct
                    trade_counts[symbol] += 1
                else:
                    action = "open_failed"

        actions[symbol] = action

    # ── 4. Cycle summary (every 12 cycles = 1 hour) ──────────────────────────
    if cycle_num % 12 == 0:
        total_trades = sum(trade_counts.values())
        all_pos      = executor.get_open_positions()
        open_pnl     = sum(p.profit for p in all_pos)
        logger.info(
            f"{'─'*55}\n"
            f"  HOURLY SUMMARY — Cycle {cycle_num} | "
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
            f"  Equity: ${equity:,.2f} | Open P&L: ${open_pnl:+.2f}\n"
            f"  Trades today: {total_trades} | "
            f"Open positions: {len(all_pos)}\n"
            f"  Per asset: " +
            " | ".join(f"{s}: {n}" for s, n in trade_counts.items()) +
            f"\n{'─'*55}"
        )

    return actions


def _ds(d: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(d, "?")


def _secs_to_next_5min() -> float:
    """Seconds until the next 5-minute boundary (00, 05, 10, ...) plus 2s buffer."""
    now     = datetime.now(timezone.utc)
    minutes = now.minute
    # Round up to next multiple of 5
    next_m  = ((minutes // 5) + 1) * 5
    if next_m >= 60:
        next_dt = now.replace(minute=0, second=2, microsecond=0) + timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_m, second=2, microsecond=0)
    return max(0.0, (next_dt - now).total_seconds())


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="MT5 High-Frequency Trader (5m)")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--once",    action="store_true", default=False)
    args = parser.parse_args()

    logger.info("═" * 55)
    logger.info("  MULTI-ASSET ML TRADING BOT — 5m SCALPING")
    logger.info(f"  Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"  Interval: every 5 minutes | Target: 100+ trades/day")
    logger.info(f"  Assets: {', '.join(ASSETS.keys())}")
    logger.info("═" * 55)

    feed = MT5DataFeed()
    acct = feed.account_info_full()
    logger.info(
        f"Account: {acct['currency']} | "
        f"Balance: ${acct['balance']:,.2f} | "
        f"Equity: ${acct['equity']:,.2f} | "
        f"Leverage: 1:{acct['leverage']}"
    )

    engine        = SignalEngine()
    risk_mgr      = RiskManager(session_start_equity=acct["equity"])
    executor      = MT5Executor(dry_run=args.dry_run)
    trade_counts  = defaultdict(int)
    cycle_num     = 0

    if args.once:
        run_cycle(feed, engine, risk_mgr, executor, trade_counts, 0)
        logger.info(f"Trades this cycle: {dict(trade_counts)}")
        feed.shutdown()
        return

    logger.info("Starting 5-minute loop. Press Ctrl+C to stop.")
    try:
        while True:
            cycle_num += 1
            run_cycle(feed, engine, risk_mgr, executor, trade_counts, cycle_num)

            if risk_mgr.is_halted:
                logger.error("Halted — drawdown limit reached.")
                break

            wait    = _secs_to_next_5min()
            next_dt = datetime.now(timezone.utc) + timedelta(seconds=wait)
            logger.info(
                f"Next cycle in {wait:.0f}s "
                f"({next_dt.strftime('%H:%M:%S UTC')}) | "
                f"Total trades today: {sum(trade_counts.values())}"
            )
            time.sleep(wait)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        total = sum(trade_counts.values())
        logger.info(
            f"Session summary: {total} total trades | "
            f"Per asset: {dict(trade_counts)}"
        )
        feed.shutdown()


if __name__ == "__main__":
    main()