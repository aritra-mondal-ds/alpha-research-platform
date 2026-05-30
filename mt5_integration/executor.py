# mt5_integration/executor.py
"""
MT5 order execution and position management.
"""

from loguru import logger
from mt5_config import MAGIC_NUMBER, SLIPPAGE_POINTS, ORDER_COMMENT
from risk_manager import TradeParams

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class MT5Executor:

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        if dry_run:
            logger.warning("MT5Executor: DRY-RUN — no real orders will be placed.")

    # ── Open position ─────────────────────────────────────────────────────────
    def open_position(self, params: TradeParams) -> bool:
        order_type = mt5.ORDER_TYPE_BUY if params.direction == 1 else mt5.ORDER_TYPE_SELL
        tick       = mt5.symbol_info_tick(params.symbol)
        price      = tick.ask if params.direction == 1 else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       params.symbol,
            "volume":       params.lot_size,
            "type":         order_type,
            "price":        price,
            "sl":           params.stop_loss,
            "tp":           params.take_profit,
            "deviation":    SLIPPAGE_POINTS,
            "magic":        MAGIC_NUMBER,
            "comment":      ORDER_COMMENT,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] {params.symbol} "
                f"{'BUY' if params.direction==1 else 'SELL'} "
                f"{params.lot_size} @ {price:.5f} | "
                f"SL={params.stop_loss:.5f} TP={params.take_profit:.5f}"
            )
            return True

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error(f"{params.symbol}: Order failed — retcode={code}")
            return False
        logger.info(
            f"{params.symbol}: Order placed ✓ ticket={result.order} | "
            f"{'BUY' if params.direction==1 else 'SELL'} "
            f"{params.lot_size} @ {result.price:.5f}"
        )
        return True

    # ── Close position ────────────────────────────────────────────────────────
    def close_position(self, position) -> bool:
        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price      = mt5.symbol_info_tick(position.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price      = mt5.symbol_info_tick(position.symbol).ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       position.symbol,
            "volume":       position.volume,
            "type":         close_type,
            "position":     position.ticket,
            "price":        price,
            "deviation":    SLIPPAGE_POINTS,
            "magic":        MAGIC_NUMBER,
            "comment":      f"{ORDER_COMMENT}_close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Close {position.symbol} "
                f"ticket={position.ticket} @ {price:.5f}"
            )
            return True

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"{position.symbol}: Close failed.")
            return False
        logger.info(
            f"{position.symbol}: Closed ✓ ticket={position.ticket} "
            f"PnL=${position.profit:+.2f}"
        )
        return True

    # ── Query ─────────────────────────────────────────────────────────────────
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
        if not positions:
            return True
        return all(self.close_position(p) for p in positions)

    def total_open_risk_pct(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for pos in self.get_open_positions():
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None or pos.sl == 0:
                continue
            px     = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            sinfo  = mt5.symbol_info(pos.symbol)
            if sinfo is None or sinfo.trade_tick_size == 0:
                continue
            ticks  = abs(px - pos.sl) / sinfo.trade_tick_size
            total += ticks * sinfo.trade_tick_value * pos.volume
        return total / equity