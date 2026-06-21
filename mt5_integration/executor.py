# mt5_integration/executor.py
"""
MT5 order execution + deal history reconciliation.

New in this revision:
    1. retcode=10018 (MARKET_CLOSED) handled gracefully — logged as INFO,
       not retried through the full fill-mode/timeout stack (pointless
       when the market itself is shut). Confirmed occurring during the
       broker's daily ~23:00 UTC rollover window.

    2. get_closed_deals(): queries MT5's actual trade history via
       history_deals_get(). This captures broker-triggered SL/TP closes
       that close_position() never sees (the broker closes the position
       directly when a stop is hit — our bot only logs closes it
       initiates itself). Without this, win rate and P&L were undercounted;
       confirmed missing at least one XAUUSD stop-loss hit on Jun 19.
"""

import time
from datetime import datetime, timedelta, timezone
from loguru import logger
from mt5_config import MAGIC_NUMBER, SLIPPAGE_POINTS, ORDER_COMMENT
from risk_manager import TradeParams

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

_fill_mode_cache: dict[str, int] = {}

TIMEOUT_RETRIES   = 2
TIMEOUT_WAIT_SECS = 2.0

MARKET_CLOSED_RETCODE = 10018


def _send_with_retry(request: dict) -> tuple[bool, object]:
    """
    Send an order with fill-mode retry (RETURN→FOK→IOC) on 10030,
    and timeout retry on 10012. Stops immediately on MARKET_CLOSED (10018)
    — retrying fill modes or waiting won't help if the market itself is shut.
    """
    symbol = request["symbol"]
    modes  = [
        (mt5.ORDER_FILLING_RETURN, "RETURN"),
        (mt5.ORDER_FILLING_FOK,    "FOK"),
        (mt5.ORDER_FILLING_IOC,    "IOC"),
    ]
    if symbol in _fill_mode_cache:
        cached = _fill_mode_cache[symbol]
        modes  = sorted(modes, key=lambda x: x[0] != cached)

    last_result = None

    for fill_mode, fill_name in modes:
        req = {**request, "type_filling": fill_mode}

        for timeout_attempt in range(1 + TIMEOUT_RETRIES):
            result      = mt5.order_send(req)
            last_result = result

            if result is None:
                break

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                _fill_mode_cache[symbol] = fill_mode
                logger.debug(f"{symbol}: fill={fill_name} ✓")
                return True, result

            if result.retcode == MARKET_CLOSED_RETCODE:
                logger.info(
                    f"{symbol}: Market closed (10018) — skipping this cycle."
                )
                return False, result

            if result.retcode == 10012:
                if timeout_attempt < TIMEOUT_RETRIES:
                    logger.warning(
                        f"{symbol}: Server timeout (10012) — "
                        f"retry {timeout_attempt + 1}/{TIMEOUT_RETRIES} ..."
                    )
                    time.sleep(TIMEOUT_WAIT_SECS)
                    continue
                else:
                    logger.error(f"{symbol}: Timeout after {TIMEOUT_RETRIES} retries.")
                    return False, result

            if result.retcode == 10030:
                logger.debug(f"{symbol}: fill={fill_name} rejected, trying next ...")
                break

            return False, result

    return False, last_result


class MT5Executor:

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._last_deal_check = datetime.now(timezone.utc) - timedelta(hours=1)
        self._known_deal_tickets: set[int] = set()
        if dry_run:
            logger.warning("MT5Executor: DRY-RUN — no real orders will be placed.")

    def open_position(self, params: TradeParams) -> bool:
        order_type = mt5.ORDER_TYPE_BUY if params.direction == 1 else mt5.ORDER_TYPE_SELL
        tick       = mt5.symbol_info_tick(params.symbol)

        if tick is None:
            logger.error(f"{params.symbol}: symbol_info_tick() returned None.")
            return False

        price    = tick.ask if params.direction == 1 else tick.bid
        lot_size = round(params.lot_size, 2)

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    params.symbol,
            "volume":    lot_size,
            "type":      order_type,
            "price":     price,
            "sl":        params.stop_loss,
            "tp":        params.take_profit,
            "deviation": SLIPPAGE_POINTS,
            "magic":     MAGIC_NUMBER,
            "comment":   ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] {params.symbol} "
                f"{'BUY' if params.direction==1 else 'SELL'} "
                f"{lot_size} @ {price:.5f} | "
                f"SL={params.stop_loss:.5f} TP={params.take_profit:.5f}"
            )
            return True

        ok, result = _send_with_retry(request)
        if not ok:
            code = result.retcode if result else "None"
            if code != MARKET_CLOSED_RETCODE:
                logger.error(f"{params.symbol}: Order failed — retcode={code}.")
            return False

        logger.info(
            f"{params.symbol}: Order placed ✓ ticket={result.order} | "
            f"{'BUY' if params.direction==1 else 'SELL'} "
            f"{lot_size} @ {result.price:.5f}"
        )
        return True

    def close_position(self, position) -> bool:
        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price      = mt5.symbol_info_tick(position.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price      = mt5.symbol_info_tick(position.symbol).ask

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    position.symbol,
            "volume":    round(position.volume, 2),
            "type":      close_type,
            "position":  position.ticket,
            "price":     price,
            "deviation": SLIPPAGE_POINTS,
            "magic":     MAGIC_NUMBER,
            "comment":   f"{ORDER_COMMENT}_close",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Close {position.symbol} "
                f"ticket={position.ticket} @ {price:.5f}"
            )
            return True

        ok, result = _send_with_retry(request)
        if not ok:
            code = result.retcode if result else "None"
            if code != MARKET_CLOSED_RETCODE:
                logger.error(f"{position.symbol}: Close failed — retcode={code}.")
            return False

        logger.info(
            f"{position.symbol}: Closed ✓ ticket={position.ticket} "
            f"PnL=${position.profit:+.2f}"
        )
        return True

    def get_open_positions(self, symbol: str = None) -> list:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        return [p for p in positions if p.magic == MAGIC_NUMBER]

    def has_open_position(self, symbol: str) -> bool:
        return len(self.get_open_positions(symbol)) > 0

    def get_position_direction(self, symbol: str) -> int:
        positions = self.get_open_positions(symbol)
        if not positions:
            return 0
        return 1 if positions[0].type == mt5.POSITION_TYPE_BUY else -1

    def close_all_for_symbol(self, symbol: str) -> bool:
        positions = self.get_open_positions(symbol)
        return all(self.close_position(p) for p in positions) if positions else True

    def total_open_risk_pct(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for pos in self.get_open_positions():
            tick  = mt5.symbol_info_tick(pos.symbol)
            sinfo = mt5.symbol_info(pos.symbol)
            if tick is None or sinfo is None or pos.sl == 0 or sinfo.trade_tick_size == 0:
                continue
            px    = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            ticks = abs(px - pos.sl) / sinfo.trade_tick_size
            total += ticks * sinfo.trade_tick_value * pos.volume
        return total / equity

    # ── Deal history reconciliation (captures broker-triggered SL/TP) ────────
    def get_closed_deals(self, lookback_hours: int = 48) -> list[dict]:
        """
        Query MT5's actual trade history for closed deals belonging to this
        bot (filtered by MAGIC_NUMBER). Captures closes that happened via
        broker-side stop-loss/take-profit execution, which close_position()
        never sees since the broker — not our code — initiated the close.

        Call this once per cycle; it tracks which deal tickets have already
        been seen so each close is reported exactly once.

        Returns:
            List of new closed-deal dicts since the last check:
            [{symbol, ticket, volume, profit, close_type, time}, ...]
        """
        if self.dry_run:
            return []

        now   = datetime.now(timezone.utc)
        since = now - timedelta(hours=lookback_hours)

        deals = mt5.history_deals_get(since, now)
        if deals is None:
            return []

        new_closes = []
        for d in deals:
            if d.magic != MAGIC_NUMBER:
                continue
            # entry=1 means this deal CLOSED a position (entry=0 = opened it)
            if d.entry != 1:
                continue
            if d.ticket in self._known_deal_tickets:
                continue

            self._known_deal_tickets.add(d.ticket)
            new_closes.append({
                "symbol":     d.symbol,
                "ticket":     d.ticket,
                "position_id":d.position_id,
                "volume":     d.volume,
                "profit":     d.profit,
                "close_type": "BUY" if d.type == 0 else "SELL",
                "time":       datetime.fromtimestamp(d.time, tz=timezone.utc),
            })

        for c in new_closes:
            logger.info(
                f"{c['symbol']}: Broker-side close detected | "
                f"ticket={c['ticket']} | PnL=${c['profit']:+.2f} "
                f"(likely SL/TP hit, not bot-initiated)"
            )

        return new_closes