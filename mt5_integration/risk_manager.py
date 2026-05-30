# mt5_integration/risk_manager.py
"""
ATR-based position sizing and portfolio risk management.

    stop_distance  = atr_price × stop_atr_mult
    risk_amount    = equity × risk_pct
    lot_size       = risk_amount / (stop_ticks × tick_value)
"""

import math
from dataclasses import dataclass
from loguru import logger

from mt5_config import ASSETS, MAX_TOTAL_RISK_PCT, MAX_DRAWDOWN_PCT, MIN_EQUITY


@dataclass
class TradeParams:
    symbol:      str
    direction:   int
    lot_size:    float
    entry_price: float
    stop_loss:   float
    take_profit: float
    risk_amount: float
    risk_pct:    float


class RiskManager:

    def __init__(self, session_start_equity: float):
        self._session_start_equity = session_start_equity
        self._halted = False
        logger.info(f"RiskManager ready — session equity: ${session_start_equity:,.2f}")

    def compute_trade_params(
        self,
        symbol:        str,
        direction:     int,
        entry_price:   float,
        atr_pct:       float,
        equity:        float,
        symbol_info:   dict,
        open_risk_pct: float = 0.0,
    ) -> "TradeParams | None":

        cfg = ASSETS.get(symbol, {})

        # ── Safety gates ──────────────────────────────────────────────────────
        if self._halted:
            logger.warning(f"{symbol}: Trading halted.")
            return None
        if equity < MIN_EQUITY:
            logger.warning(f"{symbol}: Equity ${equity:.2f} < minimum ${MIN_EQUITY}.")
            return None
        if open_risk_pct + cfg.get("risk_pct", 0.01) > MAX_TOTAL_RISK_PCT:
            logger.warning(f"{symbol}: Portfolio risk limit reached. Trade skipped.")
            return None
        drawdown = (self._session_start_equity - equity) / self._session_start_equity
        if drawdown >= MAX_DRAWDOWN_PCT:
            self._halted = True
            logger.error(f"HALT: session drawdown {drawdown*100:.1f}% ≥ limit.")
            return None
        if atr_pct <= 0:
            logger.warning(f"{symbol}: ATR is zero — cannot size position.")
            return None

        # ── Levels ────────────────────────────────────────────────────────────
        atr_price     = entry_price * atr_pct
        stop_dist     = atr_price * cfg.get("stop_atr_mult", 2.0)
        tp_dist       = atr_price * cfg.get("tp_atr_mult",   3.0)
        digits        = symbol_info["digits"]

        if direction == 1:
            sl = round(entry_price - stop_dist, digits)
            tp = round(entry_price + tp_dist,   digits)
        else:
            sl = round(entry_price + stop_dist, digits)
            tp = round(entry_price - tp_dist,   digits)

        # ── Lot size ──────────────────────────────────────────────────────────
        risk_amt      = equity * cfg.get("risk_pct", 0.01)
        tick_size     = symbol_info["trade_tick_size"]
        tick_value    = symbol_info["trade_tick_value"]
        ticks_to_stop = stop_dist / tick_size
        dollar_per_lot= ticks_to_stop * tick_value

        if dollar_per_lot <= 0:
            logger.warning(f"{symbol}: Invalid tick value.")
            return None

        vol_step = symbol_info["volume_step"]
        lots     = math.floor((risk_amt / dollar_per_lot) / vol_step) * vol_step
        lots     = max(symbol_info["volume_min"],
                   min(symbol_info["volume_max"], lots))

        if lots <= 0:
            logger.warning(
                f"{symbol}: Lot size rounds to 0 "
                f"(risk {risk_amt:.2f} / {dollar_per_lot:.2f} per lot). "
                f"Increase risk_pct or account size."
            )
            return None

        actual_risk_pct = (lots * dollar_per_lot) / equity
        logger.info(
            f"{symbol}: {'BUY' if direction==1 else 'SELL'} {lots} lots | "
            f"entry={entry_price:.5f} SL={sl:.5f} TP={tp:.5f} | "
            f"risk=${lots*dollar_per_lot:.2f} ({actual_risk_pct*100:.2f}%)"
        )
        return TradeParams(
            symbol=symbol, direction=direction, lot_size=lots,
            entry_price=entry_price, stop_loss=sl, take_profit=tp,
            risk_amount=lots * dollar_per_lot, risk_pct=actual_risk_pct,
        )

    def check_drawdown(self, equity: float) -> bool:
        dd = (self._session_start_equity - equity) / self._session_start_equity
        if dd >= MAX_DRAWDOWN_PCT:
            self._halted = True
            logger.error(f"DRAWDOWN HALT: {dd*100:.1f}%")
            return False
        return True

    def reset_session(self, equity: float):
        self._session_start_equity = equity
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted