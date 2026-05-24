# forex/fetcher.py
"""
AUD/USD OHLCV fetcher via yfinance.
Inherits BaseFetcher — filenames use symbol.replace("/","_") only,
so AUDUSD=X saves as AUDUSD=X_1h.parquet (matching DataLoader expectation).
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


class ForexFetcher(BaseFetcher):

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)

    def fetch(self, symbol: str, timeframe: str, since: str,
              force: bool = False) -> pd.DataFrame:
        assert YFINANCE_AVAILABLE, "pip install yfinance"

        if not force:
            cached = self._load_cached(symbol, timeframe)
            if cached is not None:
                logger.info(f"Data already exists → {symbol} [{timeframe}]")
                return cached

        logger.info(f"Fetching {symbol} [{timeframe}] via yfinance ...")
        ticker      = yf.Ticker(symbol)
        yf_interval = TF_MAP.get(timeframe, "1h")

        df = (ticker.history(interval=yf_interval, period="max")
              if timeframe == "1h"
              else ticker.history(interval=yf_interval, start=since, auto_adjust=True))

        if df.empty:
            raise ValueError(f"No data for {symbol}. Check Yahoo Finance symbol.")

        df = df.rename(columns={"Open":"open","High":"high","Low":"low",
                                "Close":"close","Volume":"volume"}
                       )[["open","high","low","close","volume"]]

        df.index = (df.index.tz_localize("UTC") if df.index.tz is None
                    else df.index.tz_convert("UTC"))

        # Remove weekend bars (forex closed Fri 22:00 UTC → Mon 00:00 UTC)
        df = df[df.index.dayofweek < 5]

        df = self._standardize(df)
        self._save(df, symbol, timeframe)
        logger.info(f"Saved {len(df):,} rows | "
                    f"{df.index[0].date()} → {df.index[-1].date()}")
        return df