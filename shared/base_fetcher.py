# shared/base_fetcher.py
"""
Abstract OHLCV fetcher.

Symbol → filename convention (matches crypto/src/data/loader.py exactly):
    Only "/" → "_". All other characters including "=" and "^" are KEPT.
    Examples:
        BTC/USDT  → BTC_USDT_1h.parquet
        AUDUSD=X  → AUDUSD=X_1h.parquet
        GC=F      → GC=F_1h.parquet
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from loguru import logger


class BaseFetcher(ABC):

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, since: str,
              force: bool = False) -> pd.DataFrame:
        """Download, validate, save, return OHLCV DataFrame."""

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df.astype({c: float for c in required})
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        assert df.index.is_monotonic_increasing, "Index not sorted"
        assert not df.isnull().any().any(),       "NaN values found"
        assert (df["high"] >= df["low"]).all(),   "High < Low"
        assert (df["close"] > 0).all(),           "Non-positive close"
        return df

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Only replace '/' with '_' — matches crypto/src/data/loader.py exactly."""
        safe = symbol.replace("/", "_")
        return self.data_dir / "raw" / f"{safe}_{timeframe}.parquet"

    def _save(self, df: pd.DataFrame, symbol: str, timeframe: str):
        path = self._parquet_path(symbol, timeframe)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {len(df):,} rows → {path.name}")

    def _load_cached(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._parquet_path(symbol, timeframe)
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            logger.info(f"Loaded cached {len(df):,} rows ← {path.name}")
            return df
        return None