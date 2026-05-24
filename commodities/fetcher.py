# commodities/fetcher.py
"""
Commodities OHLCV fetcher via yfinance.
Default symbol: GLD (SPDR Gold ETF) — reliable hourly history since 2004.

GLD trades NYSE hours only (Mon-Fri 09:30-16:00 ET).
yfinance returns only market-hours bars, so no weekend filtering needed.
"""

import sys
import pandas as pd
from pathlib import Path
from loguru import logger

_SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from base_fetcher import BaseFetcher

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.error("yfinance not installed — pip install yfinance")

TF_MAP = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
          "1h":"1h","4h":"4h","1d":"1d","1wk":"1wk"}


class CommoditiesFetcher(BaseFetcher):
    """
    Downloads GLD/commodity OHLCV from Yahoo Finance.
    Uses GLD (Gold ETF) by default for reliable hourly history.
    """

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)

    def fetch(
        self,
        symbol:    str,
        timeframe: str,
        since:     str,
        force:     bool = False,
    ) -> pd.DataFrame:
        assert YFINANCE_AVAILABLE, "pip install yfinance"

        if not force:
            cached = self._load_cached(symbol, timeframe)
            if cached is not None:
                logger.info(f"Data already exists → {symbol} [{timeframe}]")
                return cached

        logger.info(
            f"Fetching {symbol} [{timeframe}] from {since} via yfinance ..."
        )
        ticker      = yf.Ticker(symbol)
        yf_interval = TF_MAP.get(timeframe, "1h")

        # For hourly data, yfinance caps at ~730 days — use period="max"
        if timeframe == "1h":
            df = ticker.history(interval=yf_interval, period="max")
        else:
            df = ticker.history(
                interval=yf_interval, start=since, auto_adjust=True
            )

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for '{symbol}' [{timeframe}].\n"
                f"  Check the symbol is valid in Yahoo Finance.\n"
                f"  For Gold: try 'GLD' (ETF) or 'GC=F' (Futures)."
            )

        logger.info(
            f"Downloaded {len(df)} raw rows | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )

        # Standardise column names
        df = df.rename(columns={
            "Open":"open","High":"high","Low":"low",
            "Close":"close","Volume":"volume",
        })[["open","high","low","close","volume"]]

        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Remove any zero-price or zero-volume bars
        df = df[(df["close"] > 0) & (df["volume"] > 0)]

        if df.empty:
            raise ValueError(
                f"All bars were filtered out for {symbol} [{timeframe}]. "
                f"Check data quality."
            )

        df = self._standardize(df)
        self._save(df, symbol, timeframe)
        logger.info(
            f"Saved {len(df):,} rows | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df