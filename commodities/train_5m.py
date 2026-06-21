# commodities/train_5m.py
"""
Retrain XGBoost v2 on XAU/USD 5-minute bars from MT5.

Data source : MT5 terminal (copy_rates_range)
Horizon     : 6 bars × 5m = 30-minute forward return
Threshold   : ±0.15%
Model saved : commodities/models/xgboost_v2_5m/

Run (MT5 terminal must be open):
    cd commodities
    python train_5m.py
    python train_5m.py --trials 40 --force --from-date 2022-01-01
"""

import sys
import argparse
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CRYPTO    = REPO_ROOT / "crypto"
SHARED    = REPO_ROOT / "shared"

for p in [str(CRYPTO), str(SHARED)]:
    if p not in sys.path:
        sys.path.append(p)
sys.path.insert(0, str(ROOT))

from loguru import logger
import pandas as pd

import config.settings as cfg
from mt5_historical_fetcher       import MT5HistoricalFetcher
from src.features.technical       import TechnicalFeatures
from src.features.patterns        import ChartPatternFeatures
from src.features.target          import TargetBuilder
from src.models.xgboost_v2        import XGBoostTradingModelV2
from src.data.splitter            import temporal_split
from src.backtesting.simple_backtest  import SimpleBacktester
from src.strategies.position_manager import signals_to_positions

MT5_SYMBOL   = "XAUUSD"
TIMEFRAME    = "5m"
MACRO_TF     = "1d"
HORIZON      = 6
THRESHOLD    = 0.0015     # ±0.15%
FROM_DATE    = "2023-01-01"
MODEL_SUBDIR = "xgboost_v2_5m"


def _setup_logger():
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               colorize=True)


def build_features_5m(from_date: str, force: bool = False) -> pd.DataFrame:
    safe    = MT5_SYMBOL
    parquet = cfg.DATA_DIR / "processed" / f"{safe}_{TIMEFRAME}_features.parquet"

    if parquet.exists() and not force:
        df = pd.read_parquet(parquet, engine="pyarrow")
        if len(df) > 0:
            logger.info(f"Loaded cached 5m features: {len(df):,} rows ← {parquet.name}")
            return df

    fetcher = MT5HistoricalFetcher(cfg.DATA_DIR / "raw")
    df_5m   = fetcher.fetch(MT5_SYMBOL, TIMEFRAME, from_date, force=force)
    df_1d   = fetcher.fetch(MT5_SYMBOL, MACRO_TF, "2018-01-01", force=force)
    fetcher.shutdown()

    logger.info(f"Loaded: 5m={len(df_5m):,} | 1d={len(df_1d):,}")

    tech   = TechnicalFeatures()
    pat    = ChartPatternFeatures()
    target = TargetBuilder()

    df = tech.compute(df_5m.copy())
    df = pat.compute(df)

    df_daily = tech.compute(df_1d.copy())
    if len(df_daily) > 0:
        cols  = ["ema_50","ema_200","rsi_14","atr_pct","bb_pct","bb_width",
                 "adx","macd_hist","realised_vol_24"]
        avail = [c for c in cols if c in df_daily.columns]
        sub   = df_daily[avail].add_prefix("d1_")
        ri    = sub.reindex(df.index, method="ffill")
        for col in ri.columns:
            df[col] = ri[col].values

    df["target"] = target.build_classification(df, horizon=HORIZON, threshold=THRESHOLD)
    df.dropna(inplace=True)

    if len(df) < 5000:
        logger.warning(
            f"Only {len(df):,} rows — consider --from-date 2022-01-01 for more data."
        )

    counts = df["target"].value_counts().sort_index()
    total  = len(df)
    labels = {-1:"SHORT", 0:"HOLD", 1:"LONG"}
    logger.info(f"Feature matrix: {df.shape}")
    for val, cnt in counts.items():
        logger.info(f"  {labels.get(val,val):>6}: {cnt:>8,} ({cnt/total*100:>5.1f}%)")

    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, engine="pyarrow", compression="snappy")
    logger.info(f"Saved → {parquet.name}")
    return df


def train_5m(from_date: str, n_trials: int = 30, force: bool = False):
    df    = build_features_5m(from_date=from_date, force=force)
    split = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    fc    = [c for c in df.columns if c not in {"open","high","low","close","volume","target"}]

    logger.info(
        f"Split: train={len(split.train):,} | val={len(split.val):,} | test={len(split.test):,}\n"
        f"Train: {split.train.index[0].date()} → {split.train.index[-1].date()}\n"
        f"Test:  {split.test.index[0].date()} → {split.test.index[-1].date()}"
    )

    # ── Instantiate then override model_dir ───────────────────────────────────
    model           = XGBoostTradingModelV2()
    model.model_dir = ROOT / "models" / MODEL_SUBDIR
    model.model_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        split.train[fc], split.train["target"].astype(int),
        split.val[fc],   split.val["target"].astype(int),
        n_trials=n_trials,
    )

    X_te = split.test[model.feature_cols]
    y_te = split.test["target"].astype(int)
    model.evaluate(X_te, y_te, "Test")
    model.feature_importance(top_n=15)

    sigs = pd.Series(model.predict(X_te, threshold=0.0), index=split.test.index)
    bt   = SimpleBacktester(cfg.INITIAL_CAPITAL, cfg.COMMISSION, cfg.SLIPPAGE)
    res  = bt.run(split.test, signals_to_positions(sigs), "XGBoost v2 5m")

    m = res.metrics
    logger.info(
        f"\n5m Backtest (Test):\n"
        f"  Return={m.get('total_return_pct',0):+.2f}%  "
        f"Sharpe={m.get('sharpe',0):.3f}  "
        f"MaxDD={m.get('max_drawdown_pct',0):.2f}%  "
        f"Trades={m.get('total_trades',0)}  "
        f"WinRate={m.get('win_rate_pct',0):.1f}%  "
        f"B&H={m.get('buy_hold_pct',0):+.2f}%"
    )
    logger.info(f"✓ Model saved → commodities/models/{MODEL_SUBDIR}/")
    return model, res


if __name__ == "__main__":
    _setup_logger()
    parser = argparse.ArgumentParser(description="Train XGBoost v2 on XAUUSD 5m")
    parser.add_argument("--trials",    type=int,  default=30)
    parser.add_argument("--force",     action="store_true", default=False)
    parser.add_argument("--from-date", type=str,  default=FROM_DATE, dest="from_date")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"  XAU/USD — 5m XGBoost v2 (MT5 data) | ±{THRESHOLD*100:.3f}%")
    logger.info("=" * 60)
    train_5m(from_date=args.from_date, n_trials=args.trials, force=args.force)