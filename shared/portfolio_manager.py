# shared/portfolio_manager.py
"""
Multi-asset portfolio manager.

Aggregates signals from crypto, forex, and commodities modules,
then allocates capital across assets using configurable weighting schemes.

Weighting schemes:
    equal:          1/N allocation to each active asset
    inverse_vol:    risk-parity — weight inversely proportional to recent volatility
    fixed:          user-defined fixed weights (must sum to 1.0)

Usage:
    manager = MultiAssetPortfolio(
        assets=["crypto", "forex", "commodities"],
        total_capital=30_000.0,
        scheme="inverse_vol",
    )
    weights = manager.compute_weights(volatilities={"crypto": 0.04, "forex": 0.007, "commodities": 0.012})
    allocations = manager.allocate(weights)
"""

import numpy as np
import pandas as pd
from typing import Literal
from loguru import logger


class MultiAssetPortfolio:

    SCHEMES = ("equal", "inverse_vol", "fixed")

    def __init__(
        self,
        assets:        list,
        total_capital: float,
        scheme:        Literal["equal", "inverse_vol", "fixed"] = "equal",
        fixed_weights: dict = None,
        max_single:    float = 0.60,    # max allocation to any single asset
        min_single:    float = 0.05,    # min allocation to any active asset
    ):
        assert scheme in self.SCHEMES, f"scheme must be one of {self.SCHEMES}"
        if scheme == "fixed":
            assert fixed_weights is not None, "fixed_weights required for scheme='fixed'"
            assert abs(sum(fixed_weights.values()) - 1.0) < 1e-6, \
                "fixed_weights must sum to 1.0"

        self.assets        = assets
        self.total_capital = total_capital
        self.scheme        = scheme
        self.fixed_weights = fixed_weights
        self.max_single    = max_single
        self.min_single    = min_single

    def compute_weights(
        self,
        volatilities: dict = None,
        active_assets: list = None,
    ) -> dict:
        """
        Compute capital weights for each asset.

        Args:
            volatilities:  {asset: annualised_vol} — required for inverse_vol scheme.
            active_assets: Subset of self.assets to allocate to (others get 0).

        Returns:
            dict of {asset: weight} where weights sum to 1.0.
        """
        active = active_assets or self.assets

        if self.scheme == "equal":
            raw = {a: 1.0 / len(active) for a in active}

        elif self.scheme == "inverse_vol":
            assert volatilities, "volatilities required for inverse_vol scheme"
            inv = {a: 1.0 / (volatilities.get(a, 1.0) + 1e-9) for a in active}
            total = sum(inv.values())
            raw   = {a: v / total for a, v in inv.items()}

        elif self.scheme == "fixed":
            raw = {a: self.fixed_weights.get(a, 0.0) for a in active}
            total = sum(raw.values())
            raw   = {a: v / total for a, v in raw.items()}

        # Clip to [min_single, max_single] and renormalise
        clipped = {a: max(self.min_single, min(self.max_single, w))
                   for a, w in raw.items()}
        total   = sum(clipped.values())
        weights = {a: w / total for a, w in clipped.items()}

        # Assets not in active_assets get zero
        for a in self.assets:
            if a not in weights:
                weights[a] = 0.0

        self._log_weights(weights)
        return weights

    def allocate(self, weights: dict) -> dict:
        """Convert weights to dollar allocations."""
        alloc = {a: w * self.total_capital for a, w in weights.items()}
        logger.info("Capital allocation:")
        for asset, amount in alloc.items():
            pct = amount / self.total_capital * 100
            logger.info(f"  {asset:<15} ${amount:>10,.2f}  ({pct:.1f}%)")
        return alloc

    def rebalance(
        self,
        current_values: dict,
        target_weights: dict,
    ) -> dict:
        """
        Compute rebalancing trades given current portfolio values and target weights.

        Args:
            current_values: {asset: current_dollar_value}
            target_weights: {asset: target_weight}

        Returns:
            dict of {asset: trade_amount} — positive = buy, negative = sell.
        """
        total = sum(current_values.values())
        trades = {}
        for asset in self.assets:
            current = current_values.get(asset, 0.0)
            target  = target_weights.get(asset, 0.0) * total
            trades[asset] = target - current

        logger.info("Rebalancing trades:")
        for asset, trade in trades.items():
            direction = "BUY " if trade >= 0 else "SELL"
            logger.info(f"  {asset:<15} {direction} ${abs(trade):>10,.2f}")
        return trades

    @staticmethod
    def compute_volatilities(
        price_series: dict,
        window: int = 30,
        annualise: int = 8760,
    ) -> dict:
        """
        Compute annualised realised volatility per asset.

        Args:
            price_series: {asset: pd.Series of close prices}
            window:       Rolling window in bars
            annualise:    Bars per year (8760 for hourly)

        Returns:
            dict of {asset: annualised_volatility}
        """
        vols = {}
        for asset, prices in price_series.items():
            log_ret = np.log(prices / prices.shift(1)).dropna()
            vol = log_ret.rolling(window).std().iloc[-1] * np.sqrt(annualise)
            vols[asset] = round(float(vol), 6)
            logger.info(f"  {asset:<15} vol: {vol:.4f} ({vol*100:.2f}% annualised)")
        return vols

    def _log_weights(self, weights: dict):
        logger.info(f"Portfolio weights ({self.scheme}):")
        for asset, w in weights.items():
            bar = "█" * int(w * 30)
            logger.info(f"  {asset:<15} {w:>6.1%}  {bar}")
