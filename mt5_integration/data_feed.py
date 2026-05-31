"""
Live OHLCV data feed from MetaTrader5.
Returns UTC-indexed DataFrames compatible with crypto/src/features/pipeline.py.

Enhancements:
- explicit timeframe-aware fetching for intraday use,
- lightweight latest-bar timestamp checks,
- generic multi-timeframe fetch helpers.
"""

import pandas as pd
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
                f"Ensure MetaTrader5 terminal is running and logged in."
            )
        info = mt5.terminal_info()
        logger.info(
            f"MT5 connected — build {info.build} | "
            f"company: {mt5.account_info().company}"
        )

    def fetch(self, mt5_symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        """
        Fetch last n_bars of completed OHLCV bars.
        Bar[0] from MT5 is the currently forming bar and is discarded.
        """
        tf_const = self._get_tf_constant(timeframe)
        rates = mt5.copy_rates_from_pos(mt5_symbol, tf_const, 0, n_bars + 1)

        if rates is None or len(rates) == 0:
            raise ValueError(
                f"No data for {mt5_symbol} [{timeframe}]. "
                f"Error: {mt5.last_error()}. "
                f"Check symbol name matches your broker's Market Watch."
            )

        if len(rates) < 2:
            raise ValueError(f"Insufficient bars for {mt5_symbol} [{timeframe}].")

        df = pd.DataFrame(rates).iloc[:-1].copy()
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        df.sort_index(inplace=True)

        logger.debug(
            f"Fetched {len(df)} {timeframe} bars | {mt5_symbol} | "
            f"{df.index[0]} → {df.index[-1]}"
        )
        return df

    def fetch_with_context(
        self,
        mt5_symbol: str,
        signal_timeframe: str,
        n_signal_bars: int = 300,
        context_timeframe: str = "1d",
        n_context_bars: int = 300,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch a signal timeframe and a higher-timeframe context series.
        Suitable for intraday systems like 1m/5m + 1d context.
        """
        return (
            self.fetch(mt5_symbol, signal_timeframe, n_signal_bars),
            self.fetch(mt5_symbol, context_timeframe, n_context_bars),
        )

    def fetch_pair(
        self,
        mt5_symbol: str,
        n_bars_1h: int = 300,
        n_bars_1d: int = 300,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backward-compatible helper for the hourly trader."""
        return self.fetch(mt5_symbol, "1h", n_bars_1h), self.fetch(mt5_symbol, "1d", n_bars_1d)

    def latest_completed_bar_time(self, mt5_symbol: str, timeframe: str):
        """
        Return the timestamp of the latest completed bar.
        Uses two bars and discards the currently forming one.
        """
        tf_const = self._get_tf_constant(timeframe)
        rates = mt5.copy_rates_from_pos(mt5_symbol, tf_const, 0, 2)
        if rates is None or len(rates) < 2:
            return None
        ts = pd.to_datetime(rates[-2]["time"], unit="s", utc=True)
        return ts

    def has_new_completed_bar(self, mt5_symbol: str, timeframe: str, last_seen_time) -> tuple[bool, object]:
        """
        Cheap polling helper for intraday loops.
        Returns (is_new_bar, latest_completed_bar_time).
        """
        latest = self.latest_completed_bar_time(mt5_symbol, timeframe)
        if latest is None:
            return False, last_seen_time
        if last_seen_time is None:
            return True, latest
        return latest > last_seen_time, latest

    def symbol_info(self, mt5_symbol: str) -> dict:
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            raise ValueError(f"Symbol '{mt5_symbol}' not found in MT5.")
        return {
            "point": info.point,
            "digits": info.digits,
            "trade_tick_size": info.trade_tick_size,
            "trade_tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "spread": info.spread,
            "trade_mode": getattr(info, "trade_mode", None),
            "visible": getattr(info, "visible", None),
            "select": getattr(info, "select", None),
        }

    def account_equity(self) -> float:
        return mt5.account_info().equity

    def account_info_full(self) -> dict:
        a = mt5.account_info()
        return {
            "balance": a.balance,
            "equity": a.equity,
            "margin": a.margin,
            "free_margin": a.margin_free,
            "currency": a.currency,
            "leverage": a.leverage,
        }

    def _get_tf_constant(self, timeframe: str) -> int:
        try:
            return {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "30m": mt5.TIMEFRAME_M30,
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
            }[timeframe]
        except AttributeError:
            return MT5_TIMEFRAME_MAP[timeframe]

    def shutdown(self):
        mt5.shutdown()
        logger.info("MT5 connection closed.")
