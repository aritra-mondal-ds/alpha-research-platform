# mt5_integration/live_trader.py
"""
Live Trading Loop — 1h XGBoost v2 strategy.

Logger fix (this revision):
    loguru's {time} defaults to local system time. On this machine that
    was 2 hours ahead of UTC, causing log line prefixes to disagree with
    the UTC cycle banner and making manual log analysis confusing.
    A patcher now forces all log timestamps to true UTC, matching the
    UTC-based trading logic used everywhere else in the bot.

Deal history reconciliation (previous revision, retained):
    executor.get_closed_deals() surfaces broker-triggered SL/TP closes
    that close_position() never sees on its own.

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
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mt5_config    import ASSETS, LOG_DIR
from data_feed     import MT5DataFeed
from signal_engine import SignalEngine
from risk_manager  import RiskManager
from executor      import MT5Executor


def _utc_patcher(record):
    """Force loguru's {time} field to true UTC, overriding local system time."""
    record["time"] = datetime.now(timezone.utc)


def _setup_logger():
    logger.remove()
    logger.configure(patcher=_utc_patcher)
    logger.add(
        sys.stdout, level="INFO",
        format="<green>{time:HH:mm:ss} UTC</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(LOG_DIR / "live_trader_{time:YYYY-MM-DD}.log"),
        rotation="00:00", retention="30 days", level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} UTC | {level} | {message}",
    )


def run_cycle(feed, engine, risk_mgr, executor) -> dict:
    """One full signal → execution cycle for all assets (called hourly)."""
    equity  = feed.account_equity()
    actions = {}

    if not risk_mgr.check_drawdown(equity):
        logger.error("Drawdown limit reached — skipping cycle.")
        return {}

    logger.info(
        f"{'─'*55}\n"
        f"  CYCLE  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Equity: ${equity:,.2f}\n"
        f"{'─'*55}"
    )

    # ── 0. Reconcile broker-triggered closes (SL/TP hits we didn't initiate) ──
    executor.get_closed_deals(lookback_hours=2)

    # ── 1. Fetch bars ──────────────────────────────────────────────────────────
    live_data = {}
    for symbol, cfg in ASSETS.items():
        try:
            df_primary = feed.fetch(cfg["mt5_symbol"], cfg["primary_tf"], cfg["bars_needed"])
            df_daily   = feed.fetch(cfg["mt5_symbol"], "1d", 300)
            live_data[symbol] = (df_primary, df_daily)
        except Exception as e:
            logger.error(f"{symbol}: Data fetch failed — {e}")

    if not live_data:
        logger.error("No data fetched — aborting cycle.")
        return {}

    # ── 2. Generate signals ───────────────────────────────────────────────────
    signals       = engine.generate_all(live_data)
    open_risk_pct = executor.total_open_risk_pct(equity)

    # ── 3. Process each asset ─────────────────────────────────────────────────
    for symbol, signal in signals.items():
        cfg         = ASSETS[symbol]
        mt5_sym     = cfg["mt5_symbol"]
        current_dir = executor.get_position_direction(mt5_sym)
        desired_dir = signal.direction
        action      = "hold"

        if signal.strategy == "market_closed":
            actions[symbol] = "market_closed"
            continue

        # Close if signal reversed or went flat
        if current_dir != 0 and current_dir != desired_dir:
            logger.info(
                f"{symbol}: Signal changed {_ds(current_dir)}→{_ds(desired_dir)} "
                f"— closing position."
            )
            ok     = executor.close_all_for_symbol(mt5_sym)
            action = "closed" if ok else "close_failed"
            current_dir = 0

        # Open new position if signal is directional and currently flat
        if desired_dir != 0 and current_dir == 0:
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
                else:
                    action = "open_failed"

        actions[symbol] = action

    # ── 4. Cycle summary ──────────────────────────────────────────────────────
    all_pos  = executor.get_open_positions()
    open_pnl = sum(p.profit for p in all_pos)
    logger.info("  Actions this cycle:")
    for sym, act in actions.items():
        logger.info(f"    {sym:<12} → {act}")
    logger.info(
        f"  Open positions: {len(all_pos)} | Open P&L: ${open_pnl:+.2f} | "
        f"Equity: ${feed.account_equity():,.2f}"
    )

    return actions


def _ds(d: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(d, "?")


def _seconds_to_next_hour() -> float:
    now       = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
    return max(0.0, (next_hour - now).total_seconds())


def _print_open_positions_audit(executor):
    """
    Startup audit — list every currently open position for this bot's magic
    number, including open time. Lets the user catch anything unexpected
    (e.g. multiple concurrent positions on one asset) before trading resumes.
    """
    positions = executor.get_open_positions()
    if not positions:
        logger.info("Startup audit: no open positions.")
        return

    logger.info(f"Startup audit: {len(positions)} open position(s):")
    for p in positions:
        open_time = datetime.fromtimestamp(p.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ptype     = "BUY" if p.type == 0 else "SELL"
        logger.info(
            f"    {p.symbol:<10} {ptype:<5} vol={p.volume} | "
            f"opened={open_time} | SL={p.sl} TP={p.tp} | "
            f"P&L=${p.profit:+.2f} | ticket={p.ticket}"
        )

    # Flag any asset with more than one open position — shouldn't happen
    # given max_open_trades=1, but worth surfacing immediately if it does.
    by_symbol: dict[str, int] = {}
    for p in positions:
        by_symbol[p.symbol] = by_symbol.get(p.symbol, 0) + 1
    for sym, count in by_symbol.items():
        if count > 1:
            logger.warning(
                f"{sym}: {count} concurrent positions open — "
                f"expected at most 1. Review manually."
            )


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="MT5 Live Trader (1h XGBoost v2)")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--once",    action="store_true", default=False)
    args = parser.parse_args()

    logger.info("═" * 55)
    logger.info("  MULTI-ASSET ML TRADING BOT — 1h XGBoost v2")
    logger.info(f"  Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"  Interval: hourly")
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

    engine   = SignalEngine()
    risk_mgr = RiskManager(session_start_equity=acct["equity"])
    executor = MT5Executor(dry_run=args.dry_run)

    _print_open_positions_audit(executor)

    if args.once:
        run_cycle(feed, engine, risk_mgr, executor)
        feed.shutdown()
        return

    logger.info("Starting hourly loop. Press Ctrl+C to stop.")
    try:
        while True:
            run_cycle(feed, engine, risk_mgr, executor)

            if risk_mgr.is_halted:
                logger.error("Halted — drawdown limit reached.")
                break

            wait = _seconds_to_next_hour()
            nxt  = datetime.now(timezone.utc) + timedelta(seconds=wait)
            logger.info(f"Next cycle in {wait/60:.1f} min ({nxt.strftime('%H:%M UTC')})")
            time.sleep(wait)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        feed.shutdown()


if __name__ == "__main__":
    main()