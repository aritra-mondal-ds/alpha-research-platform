# crypto/src/backtesting/simple_backtest.py
"""
Vectorized backtester for rule-based strategies.

Design principles:
    1. Signals generated on bar close at time t
    2. Execution on NEXT bar's open (bar t+1) — no lookahead
    3. Commission applied on both entry and exit
    4. Slippage applied on execution price
    5. Long-only mode by default
    6. No external dependencies — pure pandas/numpy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg


@dataclass
class BacktestResult:
    strategy_name:      str
    equity_curve:       pd.Series          # portfolio value over time
    returns:            pd.Series          # per-bar returns
    positions:          pd.Series          # held positions over time
    trades:             pd.DataFrame       # trade log
    metrics:            dict = field(default_factory=dict)

    def summary(self) -> str:
        m   = self.metrics
        sep = "─" * 52
        lines = [
            f"\n{'═'*52}",
            f"  BACKTEST RESULTS — {self.strategy_name}",
            f"{'═'*52}",
            f"  {'Period':<30} {m.get('period','')}",
            sep,
            f"  {'Total Return':<30} {m.get('total_return_pct',0):>+.2f}%",
            f"  {'Annualised Return':<30} {m.get('ann_return_pct',0):>+.2f}%",
            f"  {'Buy & Hold Return':<30} {m.get('buy_hold_pct',0):>+.2f}%",
            sep,
            f"  {'Sharpe Ratio':<30} {m.get('sharpe',0):>.3f}",
            f"  {'Sortino Ratio':<30} {m.get('sortino',0):>.3f}",
            f"  {'Calmar Ratio':<30} {m.get('calmar',0):>.3f}",
            sep,
            f"  {'Max Drawdown':<30} {m.get('max_drawdown_pct',0):>.2f}%",
            f"  {'Avg Drawdown':<30} {m.get('avg_drawdown_pct',0):>.2f}%",
            sep,
            f"  {'Total Trades':<30} {m.get('total_trades',0):>}",
            f"  {'Win Rate':<30} {m.get('win_rate_pct',0):>.1f}%",
            f"  {'Profit Factor':<30} {m.get('profit_factor',0):>.3f}",
            f"  {'Avg Win':<30} {m.get('avg_win_pct',0):>+.2f}%",
            f"  {'Avg Loss':<30} {m.get('avg_loss_pct',0):>+.2f}%",
            f"  {'Expectancy':<30} {m.get('expectancy_pct',0):>+.4f}%",
            sep,
            f"  {'Time in Market':<30} {m.get('time_in_market_pct',0):>.1f}%",
            f"{'═'*52}\n",
        ]
        return "\n".join(lines)


class SimpleBacktester:
    """
    Pure pandas/numpy vectorized backtester.
    Produces an equity curve, trade log, and performance metrics.
    """

    def __init__(
        self,
        initial_capital: float = cfg.INITIAL_CAPITAL,
        commission:      float = cfg.COMMISSION,
        slippage:        float = cfg.SLIPPAGE,
    ):
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        df:            pd.DataFrame,
        positions:     pd.Series,      # output of signals_to_positions()
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            df:        Feature DataFrame with OHLCV columns.
            positions: Position series {-1, 0, +1}, same index as df.
            strategy_name: Label for reporting.

        Returns:
            BacktestResult with equity curve, trade log, and metrics.
        """
        assert "open" in df.columns and "close" in df.columns, \
            "DataFrame must contain 'open' and 'close' columns."
        assert positions.index.equals(df.index), \
            "Positions index must match DataFrame index."

        # ── Step 1: Shift positions by 1 bar (execute on next bar's open) ─────
        # Signal generated at close of bar t → execute at open of bar t+1
        exec_positions = positions.shift(1).fillna(0).astype(int)

        # ── Step 2: Compute execution price with slippage ─────────────────────
        exec_price = df["open"].copy()
        long_mask  = exec_positions == 1
        exec_price[long_mask]  *= (1 + self.slippage)   # pay more when buying
        exec_price[~long_mask] *= (1 - self.slippage)   # receive less when selling

        # ── Step 3: Bar-by-bar returns ────────────────────────────────────────
        # Raw price return for each bar
        price_return = df["close"] / df["close"].shift(1) - 1

        # Strategy return = position × price return
        strat_return = exec_positions * price_return

        # ── Step 4: Apply commission on position changes ──────────────────────
        position_change = exec_positions.diff().fillna(0).abs()
        commission_cost = position_change * self.commission
        strat_return   -= commission_cost

        # ── Step 5: Equity curve ──────────────────────────────────────────────
        equity = self.initial_capital * (1 + strat_return).cumprod()
        equity.iloc[0] = self.initial_capital

        # ── Step 6: Build trade log ───────────────────────────────────────────
        trades = self._build_trade_log(df, exec_positions, exec_price)

        # ── Step 7: Compute metrics ───────────────────────────────────────────
        metrics = self._compute_metrics(
            equity, strat_return, trades, df, exec_positions
        )

        result = BacktestResult(
            strategy_name = strategy_name,
            equity_curve  = equity,
            returns       = strat_return,
            positions     = exec_positions,
            trades        = trades,
            metrics       = metrics,
        )

        logger.info(result.summary())
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def _build_trade_log(
        self,
        df:             pd.DataFrame,
        positions:      pd.Series,
        exec_price:     pd.Series,
    ) -> pd.DataFrame:
        """Extract individual trade entries and exits with PnL."""
        trades = []
        in_trade    = False
        entry_price = 0.0
        entry_time  = None

        for ts, (pos, eprice, close) in enumerate(
            zip(positions, exec_price, df["close"])
        ):
            idx = positions.index[ts]

            if not in_trade and pos == 1:
                in_trade    = True
                entry_price = eprice
                entry_time  = idx

            elif in_trade and pos != 1:
                exit_price = eprice
                pnl_pct    = (exit_price / entry_price - 1) * 100
                pnl_abs    = (exit_price - entry_price) * (
                    self.initial_capital / entry_price
                )
                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   idx,
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(exit_price, 2),
                    "pnl_pct":     round(pnl_pct, 4),
                    "pnl_abs":     round(pnl_abs, 2),
                    "win":         pnl_pct > 0,
                    "duration_h":  int((idx - entry_time).total_seconds() / 3600),
                })
                in_trade = False

        return pd.DataFrame(trades)

    def _compute_metrics(
        self,
        equity:    pd.Series,
        returns:   pd.Series,
        trades:    pd.DataFrame,
        df:        pd.DataFrame,
        positions: pd.Series,
    ) -> dict:
        m   = {}
        ann = 8760   # hours per year (1h bars)

        # ── Returns ───────────────────────────────────────────────────────────
        total_ret    = equity.iloc[-1] / self.initial_capital - 1
        n_years      = len(returns) / ann
        ann_ret      = (1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1

        # Buy & hold
        bh_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1

        m["total_return_pct"] = round(total_ret * 100, 2)
        m["ann_return_pct"]   = round(ann_ret   * 100, 2)
        m["buy_hold_pct"]     = round(bh_ret    * 100, 2)
        m["period"]           = f"{df.index[0].date()} → {df.index[-1].date()}"

        # ── Risk-adjusted ─────────────────────────────────────────────────────
        r    = returns.replace(0, np.nan).dropna()
        mean = r.mean() * ann
        std  = r.std()  * np.sqrt(ann)
        downside = r[r < 0].std() * np.sqrt(ann)

        m["sharpe"]  = round(mean / std       if std       > 0 else 0.0, 4)
        m["sortino"] = round(mean / downside  if downside  > 0 else 0.0, 4)

        # ── Drawdown ──────────────────────────────────────────────────────────
        peak     = equity.cummax()
        dd       = (equity - peak) / peak * 100
        max_dd   = dd.min()
        avg_dd   = dd[dd < 0].mean() if (dd < 0).any() else 0.0

        m["max_drawdown_pct"] = round(max_dd, 2)
        m["avg_drawdown_pct"] = round(avg_dd, 2)
        m["calmar"] = round(
            ann_ret * 100 / abs(max_dd) if max_dd != 0 else 0.0, 4
        )

        # ── Trade stats ───────────────────────────────────────────────────────
        if len(trades) > 0:
            wins     = trades[trades["win"]]
            losses   = trades[~trades["win"]]
            win_rate = len(wins) / len(trades)

            gross_profit = wins["pnl_pct"].sum()   if len(wins)   > 0 else 0
            gross_loss   = losses["pnl_pct"].abs().sum() if len(losses) > 0 else 1e-9

            m["total_trades"]    = len(trades)
            m["win_rate_pct"]    = round(win_rate * 100, 2)
            m["profit_factor"]   = round(gross_profit / gross_loss, 4)
            m["avg_win_pct"]     = round(wins["pnl_pct"].mean(),   4) if len(wins)   > 0 else 0
            m["avg_loss_pct"]    = round(losses["pnl_pct"].mean(), 4) if len(losses) > 0 else 0
            m["avg_duration_h"]  = round(trades["duration_h"].mean(), 1)
            m["expectancy_pct"]  = round(
                win_rate * m["avg_win_pct"] +
                (1 - win_rate) * m["avg_loss_pct"], 4
            )
        else:
            m.update({
                "total_trades": 0, "win_rate_pct": 0,
                "profit_factor": 0, "avg_win_pct": 0,
                "avg_loss_pct": 0, "expectancy_pct": 0,
            })

        # ── Market exposure ───────────────────────────────────────────────────
        m["time_in_market_pct"] = round(
            (positions != 0).sum() / len(positions) * 100, 2
        )

        return m