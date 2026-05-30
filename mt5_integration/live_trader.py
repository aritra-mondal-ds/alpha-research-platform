# mt5_integration/live_trader.py
"""
Live Trading Loop — main entry point.

Usage:
    python live_trader.py                 # live trading
    python live_trader.py --dry-run       # signals only, no real orders
    python live_trader.py --once          # one cycle then exit (for testing)
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


def run_cycle(feed, engine, risk_mgr, executor) -> dict:
    """One full signal → execution cycle for all assets."""
    equity  = feed.account_equity()
    actions = {}

    if not risk_mgr.check_drawdown(equity):
        logger.error("Drawdown limit hit — skipping cycle.")
        return {}

    logger.info(
        f"{'─'*55}\n"
        f"  CYCLE  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Equity: ${equity:,.2f}\n"
        f"{'─'*55}"
    )

    # 1. Fetch bars
    live_data = {}
    for symbol, cfg in ASSETS.items():
        try:
            df_1h, df_1d = feed.fetch_pair(
                mt5_symbol=cfg["mt5_symbol"],
                n_bars_1h=cfg["bars_needed"],
                n_bars_1d=300,
            )
            live_data[symbol] = (df_1h, df_1d)
        except Exception as e:
            logger.error(f"{symbol}: Data fetch failed — {e}")

    if not live_data:
        logger.error("No data — aborting cycle.")
        return {}

    # 2. Generate signals
    signals       = engine.generate_all(live_data)
    open_risk_pct = executor.total_open_risk_pct(equity)

    # 3. Process each asset
    for symbol, signal in signals.items():
        mt5_sym     = ASSETS[symbol]["mt5_symbol"]
        current_dir = executor.get_position_direction(mt5_sym)
        desired_dir = signal.direction
        action      = "hold"

        # Close if signal reversed or went flat
        if current_dir != 0 and current_dir != desired_dir:
            logger.info(f"{symbol}: Closing — signal changed from "
                        f"{_ds(current_dir)} to {_ds(desired_dir)}")
            ok     = executor.close_all_for_symbol(mt5_sym)
            action = "closed" if ok else "close_failed"
            current_dir = 0

        # Open if signal is directional and currently flat
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

    # 4. Summary
    logger.info("  Actions this cycle:")
    for sym, act in actions.items():
        logger.info(f"    {sym:<12} → {act}")

    return actions


def _ds(d: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(d, "?")


def _secs_to_next_hour() -> float:
    now  = datetime.now(timezone.utc)
    next = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
    return max(0.0, (next - now).total_seconds())


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="MT5 Live Trader")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Log signals only — no real orders.")
    parser.add_argument("--once",    action="store_true", default=False,
                        help="Run one cycle and exit.")
    args = parser.parse_args()

    logger.info("═" * 55)
    logger.info("  MULTI-ASSET ML TRADING BOT — MT5 LIVE")
    logger.info(f"  Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"  Assets: {', '.join(ASSETS.keys())}")
    logger.info("═" * 55)

    feed    = MT5DataFeed()
    acct    = feed.account_info_full()
    logger.info(
        f"Account: {acct['currency']} | "
        f"Balance: ${acct['balance']:,.2f} | "
        f"Equity:  ${acct['equity']:,.2f} | "
        f"Leverage: 1:{acct['leverage']}"
    )

    engine   = SignalEngine()
    risk_mgr = RiskManager(session_start_equity=acct["equity"])
    executor = MT5Executor(dry_run=args.dry_run)

    if args.once:
        run_cycle(feed, engine, risk_mgr, executor)
        feed.shutdown()
        return

    logger.info("Starting hourly loop. Press Ctrl+C to stop.")
    try:
        while True:
            run_cycle(feed, engine, risk_mgr, executor)
            if risk_mgr.is_halted:
                logger.error("Halted due to drawdown. Exiting.")
                break
            wait = _secs_to_next_hour()
            nxt  = datetime.now(timezone.utc) + timedelta(seconds=wait)
            logger.info(f"Next cycle in {wait/60:.1f} min ({nxt.strftime('%H:%M UTC')})")
            time.sleep(wait)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        feed.shutdown()


if __name__ == "__main__":
    main()