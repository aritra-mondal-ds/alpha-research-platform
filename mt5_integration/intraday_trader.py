"""
Intraday live trader for MT5.

Design goals:
- continuous monitoring with a short polling interval,
- entries only on confirmed new bar close,
- open positions managed on new bars by default,
- stricter intraday risk controls than the hourly trader.

Usage:
    python intraday_trader.py
    python intraday_trader.py --dry-run
    python intraday_trader.py --once
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
from collections import defaultdict
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_feed import MT5DataFeed
from signal_engine import SignalEngine
from risk_manager import RiskManager
from executor import MT5Executor
from intraday_config import (
    INTRADAY_ASSETS,
    POLL_INTERVAL_SECONDS,
    HEARTBEAT_MINUTES,
    MAX_TOTAL_RISK_PCT,
    MAX_DRAWDOWN_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_NEW_TRADES_PER_CYCLE,
    REQUIRE_NEW_BAR_FOR_ENTRY,
    LOG_DIR,
)


class IntradayState:
    def __init__(self):
        self.last_bar_ts = {}
        self.last_trade_ts = {}
        self.daily_trade_count = defaultdict(int)
        self.day_key = datetime.now(timezone.utc).date()
        self.last_heartbeat_minute = None
        self.daily_start_equity = None

    def roll_day_if_needed(self, equity: float):
        today = datetime.now(timezone.utc).date()
        if today != self.day_key:
            self.day_key = today
            self.daily_trade_count.clear()
            self.last_trade_ts.clear()
            self.last_bar_ts.clear()
            self.daily_start_equity = equity
            logger.info(f"IntradayState: new UTC day {today} — counters reset.")


def _setup_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        str(LOG_DIR / "intraday_trader_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def _ds(d: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(d, "?")


def _now_utc():
    return datetime.now(timezone.utc)


def _heartbeat(state: IntradayState, equity: float, open_risk_pct: float):
    minute_bucket = _now_utc().minute // HEARTBEAT_MINUTES if HEARTBEAT_MINUTES > 0 else None
    stamp = (_now_utc().hour, minute_bucket)
    if state.last_heartbeat_minute != stamp:
        state.last_heartbeat_minute = stamp
        logger.info(
            f"Heartbeat | UTC {_now_utc().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Equity=${equity:,.2f} | OpenRisk={open_risk_pct*100:.2f}%"
        )


def _daily_loss_pct(state: IntradayState, equity: float) -> float:
    if not state.daily_start_equity or state.daily_start_equity <= 0:
        return 0.0
    return max(0.0, (state.daily_start_equity - equity) / state.daily_start_equity)


def _cooldown_ok(symbol: str, cfg: dict, state: IntradayState) -> bool:
    last_trade = state.last_trade_ts.get(symbol)
    if last_trade is None:
        return True
    elapsed_min = (_now_utc() - last_trade).total_seconds() / 60.0
    return elapsed_min >= cfg.get("cooldown_minutes", 0)


def _daily_trade_limit_ok(symbol: str, cfg: dict, state: IntradayState) -> bool:
    max_trades = cfg.get("max_trades_per_day")
    if max_trades is None:
        return True
    return state.daily_trade_count[symbol] < max_trades


def _spread_ok(feed: MT5DataFeed, mt5_symbol: str, cfg: dict) -> bool:
    max_spread = cfg.get("max_spread_points")
    if max_spread is None:
        return True
    info = feed.symbol_info(mt5_symbol)
    if info is None:
        return False
    spread = info.get("spread")
    if spread is None:
        return True
    return spread <= max_spread


def _signal_allowed(signal_dir: int, cfg: dict) -> bool:
    if signal_dir == 1 and not cfg.get("allow_long", True):
        return False
    if signal_dir == -1 and not cfg.get("allow_short", True):
        return False
    return True


def _build_intraday_risk_params(base_params, cfg: dict):
    return replace(base_params, risk_pct=min(base_params.risk_pct, cfg["risk_pct"]))


def _should_refresh_signal(symbol: str, cfg: dict, feed: MT5DataFeed, state: IntradayState, current_dir: int):
    timeframe = cfg["signal_timeframe"]
    prev_seen = state.last_bar_ts.get(symbol)
    is_new, latest_ts = feed.has_new_completed_bar(cfg["mt5_symbol"], timeframe, prev_seen)

    logger.info(
        f"{symbol}: tf={timeframe} last_seen={prev_seen} latest={latest_ts} "
        f"is_new={is_new} current_dir={current_dir}"
    )

    if latest_ts is not None:
        state.last_bar_ts[symbol] = latest_ts

    if is_new:
        return True, "new_bar"

    return False, "await_new_bar"


def run_cycle(feed, engine, risk_mgr, executor, state: IntradayState) -> dict:
    equity = feed.account_equity()
    state.roll_day_if_needed(equity)
    if state.daily_start_equity is None:
        state.daily_start_equity = equity

    actions = {}
    daily_loss = _daily_loss_pct(state, equity)
    open_risk_pct = executor.total_open_risk_pct(equity)
    _heartbeat(state, equity, open_risk_pct)

    daily_loss_limit_hit = (
        MAX_DAILY_LOSS_PCT is not None and
        daily_loss >= MAX_DAILY_LOSS_PCT
    )
    total_risk_cap_hit = (
        MAX_TOTAL_RISK_PCT is not None and
        open_risk_pct >= MAX_TOTAL_RISK_PCT
    )
    cycle_trade_cap_hit = (
        MAX_NEW_TRADES_PER_CYCLE is not None and
        MAX_NEW_TRADES_PER_CYCLE >= 0
    )

    if equity <= 0:
        logger.error("Invalid equity — skipping cycle.")
        return {}

    if daily_loss_limit_hit:
        logger.error(f"Daily loss limit hit ({daily_loss*100:.2f}%) — skipping entries.")

    if total_risk_cap_hit:
        logger.warning(f"Open risk already at {open_risk_pct*100:.2f}% — no new entries.")

    if not risk_mgr.check_drawdown(equity):
        logger.error("Drawdown limit hit — intraday trader halted for this cycle.")
        return {}

    live_data = {}
    refresh_reason = {}

    for symbol, cfg in INTRADAY_ASSETS.items():
        mt5_sym = cfg["mt5_symbol"]
        current_dir = executor.get_position_direction(mt5_sym)

        try:
            should_refresh, reason = _should_refresh_signal(symbol, cfg, feed, state, current_dir)
            refresh_reason[symbol] = reason

            if not should_refresh:
                actions[symbol] = reason
                continue

            df_tf, df_1d = feed.fetch_with_context(
                mt5_symbol=mt5_sym,
                signal_timeframe=cfg["signal_timeframe"],
                n_signal_bars=cfg["bars_needed"],
                context_timeframe="1d",
                n_context_bars=300,
            )
            live_data[symbol] = (df_tf, df_1d)
        except Exception as e:
            logger.error(f"{symbol}: Data fetch failed — {e}")
            actions[symbol] = "data_failed"

    if not live_data:
        logger.info(
            f"Intraday cycle | UTC {_now_utc().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Equity=${equity:,.2f} | DailyLoss={daily_loss*100:.2f}% | OpenRisk={open_risk_pct*100:.2f}%"
        )
        logger.info("  Actions this cycle:")
        for sym, act in actions.items():
            logger.info(f"    {sym:<12} → {act}")
        return actions

    signals = engine.generate_all(live_data)
    new_entries_this_cycle = 0

    for symbol, signal in signals.items():
        cfg = INTRADAY_ASSETS[symbol]
        mt5_sym = cfg["mt5_symbol"]
        current_dir = executor.get_position_direction(mt5_sym)
        desired_dir = signal.direction
        action = "hold"

        if not _signal_allowed(desired_dir, cfg):
            desired_dir = 0

        if current_dir != 0 and current_dir != desired_dir:
            logger.info(f"{symbol}: Closing — signal changed from {_ds(current_dir)} to {_ds(desired_dir)}")
            close_res = executor.close_all_for_symbol(mt5_sym)
            if close_res.ok:
                action = "closed"
                current_dir = 0
            else:
                action = close_res.status
                actions[symbol] = action
                continue

        if current_dir != 0:
            actions[symbol] = action
            continue

        if desired_dir == 0:
            actions[symbol] = action
            continue

        if REQUIRE_NEW_BAR_FOR_ENTRY and refresh_reason.get(symbol) != "new_bar":
            actions[symbol] = "await_new_bar"
            continue

        if daily_loss_limit_hit:
            actions[symbol] = "daily_loss_limit"
            continue

        if total_risk_cap_hit:
            actions[symbol] = "risk_cap_reached"
            continue

        if cycle_trade_cap_hit and new_entries_this_cycle >= MAX_NEW_TRADES_PER_CYCLE:
            actions[symbol] = "cycle_trade_cap"
            continue

        if not _cooldown_ok(symbol, cfg, state):
            actions[symbol] = "cooldown"
            continue

        if not _daily_trade_limit_ok(symbol, cfg, state):
            actions[symbol] = "daily_trade_cap"
            continue

        if not _spread_ok(feed, mt5_sym, cfg):
            actions[symbol] = "spread_too_wide"
            continue

        sym_info = feed.symbol_info(mt5_sym)
        params = risk_mgr.compute_trade_params(
            asset=symbol,
            mt5_symbol=mt5_sym,
            direction=desired_dir,
            entry_price=signal.current_price,
            atr_pct=signal.atr_pct,
            equity=equity,
            symbol_info=sym_info,
            open_risk_pct=open_risk_pct,
        )
        if params is None:
            actions[symbol] = "params_none"
            continue

        params = _build_intraday_risk_params(params, cfg)
        open_res = executor.open_position(params)
        if open_res.ok:
            action = f"opened_{_ds(desired_dir)}"
            open_risk_pct += params.risk_pct
            new_entries_this_cycle += 1
            state.last_trade_ts[symbol] = _now_utc()
            state.daily_trade_count[symbol] += 1
        else:
            action = open_res.status

        actions[symbol] = action

    for symbol in INTRADAY_ASSETS:
        actions.setdefault(symbol, refresh_reason.get(symbol, "hold"))

    logger.info(
        f"Intraday cycle | UTC {_now_utc().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Equity=${equity:,.2f} | DailyLoss={daily_loss*100:.2f}% | OpenRisk={open_risk_pct*100:.2f}%"
    )
    logger.info("  Actions this cycle:")
    for sym, act in actions.items():
        logger.info(f"    {sym:<12} → {act}")

    return actions


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(description="MT5 Intraday Live Trader")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Log signals only — no real orders.")
    parser.add_argument("--once", action="store_true", default=False, help="Run one cycle and exit.")
    args = parser.parse_args()

    logger.info("═" * 55)
    logger.info("  MULTI-ASSET ML TRADING BOT — MT5 INTRADAY")
    logger.info(f"  Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"  Assets: {', '.join(INTRADAY_ASSETS.keys())}")
    logger.info(f"  Poll interval: {POLL_INTERVAL_SECONDS}s")
    logger.info("═" * 55)

    feed = MT5DataFeed()
    acct = feed.account_info_full()
    logger.info(
        f"Account: {acct['currency']} | "
        f"Balance: ${acct['balance']:,.2f} | "
        f"Equity:  ${acct['equity']:,.2f} | "
        f"Leverage: 1:{acct['leverage']}"
    )

    engine = SignalEngine()
    risk_mgr = RiskManager(session_start_equity=acct["equity"])
    if MAX_TOTAL_RISK_PCT is not None:
        risk_mgr.max_total_risk_pct = MAX_TOTAL_RISK_PCT
    if MAX_DRAWDOWN_PCT is not None:
        risk_mgr.max_drawdown_pct = MAX_DRAWDOWN_PCT
    executor = MT5Executor(dry_run=args.dry_run)
    state = IntradayState()
    state.daily_start_equity = acct["equity"]

    if args.once:
        run_cycle(feed, engine, risk_mgr, executor, state)
        feed.shutdown()
        return

    logger.info("Starting continuous intraday loop. Press Ctrl+C to stop.")
    try:
        while True:
            run_cycle(feed, engine, risk_mgr, executor, state)
            if risk_mgr.is_halted:
                logger.error("Halted due to drawdown. Exiting.")
                break
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        feed.shutdown()


if __name__ == "__main__":
    main()
