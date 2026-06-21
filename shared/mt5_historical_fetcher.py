# shared/mt5_historical_fetcher.py
"""
Fetches historical OHLCV bars from a connected MetaTrader5 terminal
and saves them as parquet files for model training.

Uses MT5's copy_rates_range() which can pull 2+ years of 5m data
for major forex pairs and indices — far beyond yfinance's 60-day cap.

Usage:
    from shared.mt5_historical_fetcher import MT5HistoricalFetcher

    fetcher = MT5HistoricalFetcher(data_dir=Path("data/raw"))
    df = fetcher.fetch("AUDUSD",  "5m", from_date="2023-01-01")
    df = fetcher.fetch("XAUUSD",  "5m", from_date="2023-01-01")
"""

import sys
import math
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# MT5 timeframe constants (fallback integers if mt5 not yet initialised)
_TF_MAP = {
    "1m":  1,       "5m":  5,    "15m": 15,
    "30m": 30,      "1h":  16385, "4h": 16388,
    "1d":  16408,
}


class MT5HistoricalFetcher:
    """
    Connects to a running MT5 terminal and bulk-downloads historical
    OHLCV data for training the 5m ML models.
    """

    # MT5 caps copy_rates_range at ~100,000 bars per call on some builds.
    # We chunk into 50k-bar windows to stay safe.
    CHUNK_BARS = 50_000

    def __init__(self, data_dir: Path, auto_connect: bool = True):
        if not MT5_AVAILABLE:
            raise ImportError(
                "MetaTrader5 not installed.\n"
                "Install with: pip install MetaTrader5"
            )
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._connected = False

        if auto_connect:
            self._connect()

    # ── Connection management ─────────────────────────────────────────────────
    def _connect(self):
        if not mt5.initialize():
            raise ConnectionError(
                f"MT5 initialize() failed: {mt5.last_error()}\n"
                f"Make sure the MT5 terminal is open and logged in."
            )
        info = mt5.terminal_info()
        logger.info(
            f"MT5HistoricalFetcher connected — "
            f"build {info.build} | {mt5.account_info().company}"
        )
        self._connected = True

    def shutdown(self):
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5HistoricalFetcher: connection closed.")

    # ── Main fetch method ─────────────────────────────────────────────────────
    def fetch(
        self,
        mt5_symbol: str,
        timeframe:  str,
        from_date:  str,
        to_date:    str | None = None,
        force:      bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars and cache as parquet.

        Args:
            mt5_symbol: exact MT5 symbol name (e.g. "AUDUSD", "XAUUSD", "#BTCUSD")
            timeframe:  "5m", "1m", "15m", "1h", "1d"
            from_date:  start date string "YYYY-MM-DD"
            to_date:    end date string (default: now)
            force:      re-download even if parquet exists

        Returns:
            UTC-indexed DataFrame with [open, high, low, close, volume]
        """
        safe_sym  = mt5_symbol.replace("/", "_").replace("#", "")
        parquet   = self.data_dir / f"{safe_sym}_{timeframe}.parquet"

        if parquet.exists() and not force:
            df = pd.read_parquet(parquet, engine="pyarrow")
            if len(df) > 0:
                logger.info(
                    f"Loaded cached {len(df):,} rows ← {parquet.name} | "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
                return df
            logger.warning(f"Cached parquet is empty — re-fetching.")

        # ── Date range ────────────────────────────────────────────────────────
        dt_from = datetime.strptime(from_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        dt_to   = (
            datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if to_date
            else datetime.now(timezone.utc)
        )

        logger.info(
            f"Fetching {mt5_symbol} [{timeframe}] "
            f"{dt_from.date()} → {dt_to.date()} from MT5 ..."
        )

        # ── Fetch in chunks ───────────────────────────────────────────────────
        tf_const    = self._tf_const(timeframe)
        all_chunks  = []
        chunk_start = dt_from

        while chunk_start < dt_to:
            chunk_end = min(
                chunk_start + self._chunk_timedelta(timeframe),
                dt_to,
            )
            rates = mt5.copy_rates_range(
                mt5_symbol, tf_const,
                chunk_start.replace(tzinfo=None),   # MT5 expects naive UTC
                chunk_end.replace(tzinfo=None),
            )
            if rates is not None and len(rates) > 0:
                all_chunks.append(rates)
                logger.debug(
                    f"  Chunk {chunk_start.date()} → {chunk_end.date()}: "
                    f"{len(rates):,} bars"
                )
            chunk_start = chunk_end

        if not all_chunks:
            err = mt5.last_error()
            raise ValueError(
                f"No data returned for {mt5_symbol} [{timeframe}]. "
                f"MT5 error: {err}\n"
                f"Check: symbol name, terminal connection, and date range."
            )

        # ── Assemble and clean ────────────────────────────────────────────────
        import numpy as np
        raw = np.concatenate(all_chunks)
        df  = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"}
                       )[["open","high","low","close","volume"]].astype(float)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        logger.info(
            f"Downloaded {len(df):,} rows | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )

        # ── Save ──────────────────────────────────────────────────────────────
        df.to_parquet(parquet, engine="pyarrow", compression="snappy")
        logger.info(f"Saved → {parquet.name}")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _tf_const(self, timeframe: str) -> int:
        try:
            return {
                "1m":  mt5.TIMEFRAME_M1,  "5m":  mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
                "1h":  mt5.TIMEFRAME_H1,  "4h":  mt5.TIMEFRAME_H4,
                "1d":  mt5.TIMEFRAME_D1,
            }[timeframe]
        except AttributeError:
            return _TF_MAP[timeframe]

    def _chunk_timedelta(self, timeframe: str) -> timedelta:
        """How much time does CHUNK_BARS cover for this timeframe?"""
        minutes_per_bar = {
            "1m":1, "5m":5, "15m":15, "30m":30,
            "1h":60, "4h":240, "1d":1440,
        }[timeframe]
        total_minutes = self.CHUNK_BARS * minutes_per_bar
        return timedelta(minutes=total_minutes)