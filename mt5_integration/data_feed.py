# mt5_integration/data_feed.py
"""
Live OHLCV data feed from MetaTrader5.
Returns UTC-indexed DataFrames compatible with crypto/src/features/pipeline.py.
"""

import pandas as pd
from pathlib import Path
from loguru import logger

from mt5_config import MT5_TIMEFRAME_MAP

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.error("MetaTrader5 not installed — pip install MetaTrader5")


class MT5DataFeed:

    def __init__(self):
        if not MT5_AVAILABLE:
            raise ImportError("pip install MetaTrader5")
        if not mt5.initialize():
            raise ConnectionError(
                f"MT5 initialize() failed: {mt5.last_error()}\n"
                f"Ensure MT5 terminal is running and logged in."
            )
        info = mt5.terminal_info()
        logger.info(
            f"MT5 connected — build {info.build} | "
            f"company: {mt5.account_info().company}"
        )

    def fetch(self, mt5_symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        """Fetch last n_bars of completed OHLCV bars (drops the forming bar)."""
        tf_const = self._get_tf_constant(timeframe)
        rates    = mt5.copy_rates_from_pos(mt5_symbol, tf_const, 0, n_bars + 1)

        if rates is None or len(rates) == 0:
            raise ValueError(
                f"No data for {mt5_symbol} [{timeframe}]. "
                f"Error: {mt5.last_error()}. "
                f"Check symbol name matches your broker's Market Watch."
            )

        df = pd.DataFrame(rates).iloc[:-1].copy()
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"}
                       )[["open","high","low","close","volume"]].astype(float)
        df.sort_index(inplace=True)
        return df

    def fetch_pair(
        self, mt5_symbol: str, n_bars_1h: int = 600, n_bars_1d: int = 300,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch both primary (5m) and 1d macro bars in one call."""
        tf = "5m"   # primary timeframe for scalping
        return self.fetch(mt5_symbol, tf, n_bars_1h), \
               self.fetch(mt5_symbol, "1d", n_bars_1d)

    def symbol_info(self, mt5_symbol: str) -> dict:
        """
        Return key symbol properties needed for position sizing.

        Includes stops_level and point — essential for enforcing the broker's
        minimum stop distance (TRADE_RETCODE_INVALID_STOPS prevention).

        stops_level: minimum distance in points from current price to SL/TP
        point:       value of one point in price (e.g. 0.00001 for AUDUSD)
        min_stop_distance = stops_level × point
        """
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            raise ValueError(f"Symbol '{mt5_symbol}' not found in MT5.")
        return {
            "point":              info.point,
            "digits":             info.digits,
            "trade_tick_size":    info.trade_tick_size,
            "trade_tick_value":   info.trade_tick_value,
            "volume_min":         info.volume_min,
            "volume_max":         info.volume_max,
            "volume_step":        info.volume_step,
            "spread":             info.spread,
            # Minimum stop distance — broker-enforced freeze level
            "stops_level":        getattr(info, "trade_stops_level", 0),
        }

    def account_equity(self) -> float:
        return mt5.account_info().equity

    def account_info_full(self) -> dict:
        a = mt5.account_info()
        return {
            "balance":      a.balance,  "equity":      a.equity,
            "margin":       a.margin,   "free_margin": a.margin_free,
            "currency":     a.currency, "leverage":    a.leverage,
        }

    def _get_tf_constant(self, timeframe: str) -> int:
        try:
            return {
                "1m": mt5.TIMEFRAME_M1,  "5m":  mt5.TIMEFRAME_M5,
                "15m":mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
                "1h": mt5.TIMEFRAME_H1,  "4h":  mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
            }[timeframe]
        except AttributeError:
            return MT5_TIMEFRAME_MAP[timeframe]

    def shutdown(self):
        mt5.shutdown()
        logger.info("MT5 connection closed.")