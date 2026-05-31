# # mt5_integration/executor.py

"""
MT5 order execution and position management.
"""

import time
from dataclasses import dataclass
from loguru import logger
from mt5_config import MAGIC_NUMBER, SLIPPAGE_POINTS, ORDER_COMMENT
from risk_manager import TradeParams

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


@dataclass
class ExecutionResult:
    ok: bool
    status: str
    message: str = ""


class MT5Executor:

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        if dry_run:
            logger.warning("MT5Executor: DRY-RUN — no real orders will be placed.")

    def _get_tick(self, symbol: str, retries: int = 3, delay: float = 0.2):
        if not mt5.symbol_select(symbol, True):
            logger.error(f"{symbol}: symbol_select failed: {mt5.last_error()}")
            return None
        for _ in range(retries):
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                return tick
            time.sleep(delay)
        logger.error(f"{symbol}: symbol_info_tick returned None: {mt5.last_error()}")
        return None

    def _pick_filling_mode(self, info):
        supported = []
        mode = getattr(info, "filling_mode", None)

        if mode is None:
            return mt5.ORDER_FILLING_IOC, ["unknown"]

        if mode == 0:
            supported.append(mt5.ORDER_FILLING_RETURN)
        else:
            if mode & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
                supported.append(mt5.ORDER_FILLING_FOK)
            if mode & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
                supported.append(mt5.ORDER_FILLING_IOC)
            if mode & getattr(mt5, "SYMBOL_FILLING_BOC", 4):
                boc = getattr(mt5, "ORDER_FILLING_BOC", None)
                if boc is not None:
                    supported.append(boc)

        trade_exemode = getattr(info, "trade_exemode", None)
        if trade_exemode == getattr(mt5, "SYMBOL_TRADE_EXECUTION_EXCHANGE", object()):
            supported.append(mt5.ORDER_FILLING_RETURN)

        priority = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
        for candidate in priority:
            if candidate in supported:
                return candidate, supported

        return mt5.ORDER_FILLING_IOC, supported

    def _build_request(self, symbol, params, order_type, price, filling_mode):
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": params.lot_size,
            "type": order_type,
            "price": price,
            "sl": params.stop_loss,
            "tp": params.take_profit,
            "deviation": SLIPPAGE_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

    def open_position(self, params: TradeParams) -> ExecutionResult:
        symbol = params.mt5_symbol
        order_type = mt5.ORDER_TYPE_BUY if params.direction == 1 else mt5.ORDER_TYPE_SELL
        info = mt5.symbol_info(symbol)
        if info is None:
            msg = f"symbol_info failed: {mt5.last_error()}"
            logger.error(f"{params.asset}/{symbol}: {msg}")
            return ExecutionResult(False, "symbol_info_failed", msg)

        tick = self._get_tick(symbol)
        if tick is None:
            return ExecutionResult(False, "tick_unavailable", "symbol_info_tick returned None")
        price = tick.ask if params.direction == 1 else tick.bid

        filling_mode, supported = self._pick_filling_mode(info)
        request = self._build_request(symbol, params, order_type, price, filling_mode)

        logger.info(
            f"{params.asset}/{symbol}: fill_mode={filling_mode} supported={supported} "
            f"exec_mode={getattr(info, 'trade_exemode', 'n/a')}"
        )

        check = mt5.order_check(request)
        if check is not None and getattr(check, "retcode", 0) not in (0, mt5.TRADE_RETCODE_DONE):
            logger.warning(
                f"{params.asset}/{symbol}: order_check retcode={check.retcode} "
                f"comment={getattr(check, 'comment', '')}"
            )

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] {params.asset}/{symbol} "
                f"{'BUY' if params.direction==1 else 'SELL'} {params.lot_size} @ {price:.5f} | "
                f"SL={params.stop_loss:.5f} TP={params.take_profit:.5f} | fill={filling_mode}"
            )
            return ExecutionResult(True, "opened_dry_run")

        result = mt5.order_send(request)
        if result is None:
            msg = f"order_send returned None: {mt5.last_error()}"
            logger.error(f"{params.asset}/{symbol}: {msg}")
            return ExecutionResult(False, "order_send_none", msg)

        if result.retcode == mt5.TRADE_RETCODE_MARKET_CLOSED:
            logger.warning(f"{params.asset}/{symbol}: Market closed — skipping trade.")
            return ExecutionResult(False, "market_closed", getattr(result, 'comment', ''))

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"retcode={result.retcode} comment={getattr(result, 'comment', '')} last_error={mt5.last_error()}"
            logger.error(f"{params.asset}/{symbol}: Order failed — {msg}")
            return ExecutionResult(False, "order_failed", msg)

        logger.info(
            f"{params.asset}/{symbol}: Order placed ✓ ticket={result.order} | "
            f"{'BUY' if params.direction==1 else 'SELL'} {params.lot_size} @ {result.price:.5f}"
        )
        return ExecutionResult(True, "opened")

    def close_position(self, position) -> ExecutionResult:
        symbol = position.symbol
        info = mt5.symbol_info(symbol)
        if info is None:
            msg = f"symbol_info failed during close: {mt5.last_error()}"
            logger.error(f"{symbol}: {msg}")
            return ExecutionResult(False, "symbol_info_failed", msg)

        tick = self._get_tick(symbol)
        if tick is None:
            return ExecutionResult(False, "tick_unavailable", "symbol_info_tick returned None")

        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        filling_mode, supported = self._pick_filling_mode(info)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": SLIPPAGE_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": f"{ORDER_COMMENT}_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(f"{symbol}: close fill_mode={filling_mode} supported={supported}")

        if self.dry_run:
            logger.info(f"[DRY-RUN] Close {symbol} ticket={position.ticket} @ {price:.5f}")
            return ExecutionResult(True, "closed_dry_run")

        result = mt5.order_send(request)
        if result is None:
            msg = f"close order_send returned None: {mt5.last_error()}"
            logger.error(f"{symbol}: {msg}")
            return ExecutionResult(False, "order_send_none", msg)

        if result.retcode == mt5.TRADE_RETCODE_MARKET_CLOSED:
            logger.warning(f"{symbol}: Market closed — cannot close right now.")
            return ExecutionResult(False, "market_closed", getattr(result, 'comment', ''))

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"retcode={result.retcode} comment={getattr(result, 'comment', '')} last_error={mt5.last_error()}"
            logger.error(f"{symbol}: Close failed — {msg}")
            return ExecutionResult(False, "close_failed", msg)

        logger.info(f"{symbol}: Closed ✓ ticket={position.ticket} PnL=${position.profit:+.2f}")
        return ExecutionResult(True, "closed")

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

    def close_all_for_symbol(self, symbol: str) -> ExecutionResult:
        positions = self.get_open_positions(symbol)
        if not positions:
            return ExecutionResult(True, "no_position")
        results = [self.close_position(p) for p in positions]
        if all(r.ok for r in results):
            return ExecutionResult(True, "closed")
        first_fail = next((r for r in results if not r.ok), ExecutionResult(False, "close_failed"))
        return first_fail

    def total_open_risk_pct(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for pos in self.get_open_positions():
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None or pos.sl == 0:
                continue
            px = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            sinfo = mt5.symbol_info(pos.symbol)
            if sinfo is None or sinfo.trade_tick_size == 0:
                continue
            ticks = abs(px - pos.sl) / sinfo.trade_tick_size
            total += ticks * sinfo.trade_tick_value * pos.volume
        return total / equity
