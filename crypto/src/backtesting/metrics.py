# crypto/src/backtesting/metrics.py
"""
Comprehensive evaluation metrics for trading strategy assessment.

Covers:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)
    - Drawdown analysis (max, avg, duration, recovery)
    - Overfitting detection (in-sample vs out-of-sample degradation)
    - Monte Carlo simulation (bootstrap equity curve confidence bands)
    - Walk-forward consistency scoring
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config.settings as cfg


# ── Core Metrics ──────────────────────────────────────────────────────────────
def compute_full_metrics(
    equity:     pd.Series,
    returns:    pd.Series,
    trades:     pd.DataFrame,
    df:         pd.DataFrame,
    positions:  pd.Series,
    initial_capital: float = cfg.INITIAL_CAPITAL,
    ann_factor: int = 8760,    # hours per year for 1h data
) -> dict:
    """
    Compute the full suite of trading metrics.

    Args:
        equity:    Portfolio value over time.
        returns:   Per-bar strategy returns.
        trades:    Trade log DataFrame from SimpleBacktester.
        df:        Feature DataFrame with 'close' column.
        positions: Position series {-1, 0, 1}.
        initial_capital: Starting capital.
        ann_factor: Bars per year for annualisation.

    Returns:
        dict of metric name → value.
    """
    m = {}

    # ── Returns ───────────────────────────────────────────────────────────────
    total_ret  = equity.iloc[-1] / initial_capital - 1
    n_years    = len(returns) / ann_factor
    ann_ret    = (1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1
    bh_ret     = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    alpha      = total_ret - bh_ret

    m["total_return_pct"]    = round(total_ret * 100, 4)
    m["ann_return_pct"]      = round(ann_ret   * 100, 4)
    m["buy_hold_pct"]        = round(bh_ret    * 100, 4)
    m["alpha_pct"]           = round(alpha     * 100, 4)
    m["period_start"]        = str(df.index[0].date())
    m["period_end"]          = str(df.index[-1].date())

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    r_clean = returns.replace(0, np.nan).dropna()
    mu      = r_clean.mean() * ann_factor
    sigma   = r_clean.std()  * np.sqrt(ann_factor)
    down_sd = r_clean[r_clean < 0].std() * np.sqrt(ann_factor)

    # Omega ratio: probability-weighted gains vs losses
    threshold = 0.0
    gains  = r_clean[r_clean > threshold].sum()
    losses = abs(r_clean[r_clean <= threshold].sum())
    omega  = gains / losses if losses > 0 else np.inf

    m["sharpe"]   = round(mu / sigma    if sigma   > 0 else 0.0, 4)
    m["sortino"]  = round(mu / down_sd  if down_sd > 0 else 0.0, 4)
    m["omega"]    = round(omega, 4)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    # Drawdown duration
    in_dd      = dd < 0
    dd_periods = []
    start      = None
    for i, v in enumerate(in_dd):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dd_periods.append(i - start)
            start = None
    avg_dd_dur = np.mean(dd_periods) if dd_periods else 0

    m["max_drawdown_pct"]    = round(max_dd * 100, 4)
    m["avg_drawdown_pct"]    = round(dd[dd < 0].mean() * 100 if (dd < 0).any() else 0.0, 4)
    m["avg_drawdown_dur_h"]  = round(avg_dd_dur, 1)
    m["calmar"]              = round(
        ann_ret * 100 / abs(max_dd * 100) if max_dd != 0 else 0.0, 4
    )

    # ── Trade statistics ──────────────────────────────────────────────────────
    if len(trades) > 0:
        wins        = trades[trades["win"]]
        losses_df   = trades[~trades["win"]]
        win_rate    = len(wins) / len(trades)
        gross_p     = wins["pnl_pct"].sum()   if len(wins)      > 0 else 0
        gross_l     = losses_df["pnl_pct"].abs().sum() if len(losses_df) > 0 else 1e-9

        m["total_trades"]   = len(trades)
        m["win_rate_pct"]   = round(win_rate * 100, 4)
        m["profit_factor"]  = round(gross_p / gross_l, 4)
        m["avg_win_pct"]    = round(wins["pnl_pct"].mean(),        4) if len(wins)      > 0 else 0
        m["avg_loss_pct"]   = round(losses_df["pnl_pct"].mean(),   4) if len(losses_df) > 0 else 0
        m["avg_duration_h"] = round(trades["duration_h"].mean(),   2)
        m["expectancy_pct"] = round(
            win_rate * m["avg_win_pct"] +
            (1 - win_rate) * m["avg_loss_pct"], 4
        )
        m["largest_win_pct"]  = round(trades["pnl_pct"].max(), 4)
        m["largest_loss_pct"] = round(trades["pnl_pct"].min(), 4)
    else:
        for k in ["total_trades","win_rate_pct","profit_factor",
                  "avg_win_pct","avg_loss_pct","avg_duration_h",
                  "expectancy_pct","largest_win_pct","largest_loss_pct"]:
            m[k] = 0

    m["time_in_market_pct"] = round(
        (positions != 0).sum() / len(positions) * 100, 4
    )

    return m


# ── Overfitting Detection ─────────────────────────────────────────────────────
def check_overfitting(
    train_metrics:  dict,
    test_metrics:   dict,
    threshold:      float = 0.5,
) -> dict:
    """
    Compare in-sample (train) vs out-of-sample (test) performance.
    Flags overfitting when test performance degrades > threshold × train.

    Args:
        train_metrics: Metrics dict from training set backtest.
        test_metrics:  Metrics dict from test set backtest.
        threshold:     Acceptable test/train ratio (default 0.5 = 50% degradation).

    Returns:
        dict of metric → {train, test, ratio, status}
    """
    checks   = {}
    key_metrics = ["sharpe", "win_rate_pct", "profit_factor", "total_return_pct"]

    sep = "─" * 58
    print(f"\n{'═'*58}")
    print(f"  OVERFITTING CHECK")
    print(f"{'═'*58}")
    print(f"  {'Metric':<22} {'Train':>10} {'Test':>10} {'Ratio':>8}  Status")
    print(sep)

    for key in key_metrics:
        train_val = train_metrics.get(key, 0)
        test_val  = test_metrics.get(key,  0)
        ratio     = test_val / (train_val + 1e-9) if train_val != 0 else 0
        ok        = ratio >= threshold

        label = train_metrics.get("strategy_name", key)
        status = "✓ OK" if ok else "✗ OVERFIT"

        print(
            f"  {key:<22} {train_val:>10.3f} {test_val:>10.3f} "
            f"{ratio:>8.3f}  {status}"
        )
        checks[key] = {
            "train": train_val, "test": test_val,
            "ratio": round(ratio, 4), "ok": ok
        }

    n_ok     = sum(1 for v in checks.values() if v["ok"])
    verdict  = "PASS" if n_ok >= 3 else "WARN" if n_ok >= 2 else "FAIL"
    print(sep)
    print(f"  Overall verdict: {verdict}  ({n_ok}/{len(key_metrics)} metrics OK)")
    print(f"{'═'*58}\n")

    return checks


# ── Monte Carlo Bootstrap ─────────────────────────────────────────────────────
def monte_carlo_simulation(
    trade_returns:  pd.Series,
    initial_capital: float = cfg.INITIAL_CAPITAL,
    n_simulations:  int    = 1000,
    n_trades:       int    = None,
    seed:           int    = cfg.RANDOM_SEED,
) -> dict:
    """
    Bootstrap resample trade returns to estimate distribution of outcomes.
    Answers: "if we replayed the strategy with different trade ordering,
    what range of outcomes would we expect?"

    Args:
        trade_returns:   Series of per-trade return percentages.
        initial_capital: Starting capital.
        n_simulations:   Bootstrap samples.
        n_trades:        Trades per simulation (default: same as observed).
        seed:            Random seed.

    Returns:
        dict with percentile outcomes and probability of loss.
    """
    if len(trade_returns) < 5:
        logger.warning(
            f"Monte Carlo: only {len(trade_returns)} trades — "
            f"results unreliable. Need 30+ for statistical significance."
        )

    np.random.seed(seed)
    n    = n_trades or len(trade_returns)
    rets = trade_returns.values / 100    # convert % to decimal

    final_equities = []
    for _ in range(n_simulations):
        sampled   = np.random.choice(rets, size=n, replace=True)
        equity    = initial_capital * np.prod(1 + sampled)
        final_equities.append(equity)

    final_equities = np.array(final_equities)
    final_returns  = (final_equities / initial_capital - 1) * 100

    result = {
        "n_simulations":      n_simulations,
        "n_trades":           n,
        "median_return_pct":  round(float(np.median(final_returns)), 2),
        "p5_return_pct":      round(float(np.percentile(final_returns,  5)), 2),
        "p25_return_pct":     round(float(np.percentile(final_returns, 25)), 2),
        "p75_return_pct":     round(float(np.percentile(final_returns, 75)), 2),
        "p95_return_pct":     round(float(np.percentile(final_returns, 95)), 2),
        "prob_loss_pct":      round(float((final_returns < 0).mean() * 100), 2),
        "prob_beat_bh_pct":   None,    # set by caller if B&H known
        "mean_return_pct":    round(float(final_returns.mean()), 2),
        "std_return_pct":     round(float(final_returns.std()),  2),
    }

    sep = "─" * 48
    print(f"\n{'═'*48}")
    print(f"  MONTE CARLO SIMULATION  ({n_simulations:,} bootstrap samples)")
    print(f"  Based on {n} trades resampled with replacement")
    print(f"{'═'*48}")
    print(f"  {'5th  percentile':<30} {result['p5_return_pct']:>+.1f}%")
    print(f"  {'25th percentile':<30} {result['p25_return_pct']:>+.1f}%")
    print(f"  {'Median outcome':<30} {result['median_return_pct']:>+.1f}%")
    print(f"  {'75th percentile':<30} {result['p75_return_pct']:>+.1f}%")
    print(f"  {'95th percentile':<30} {result['p95_return_pct']:>+.1f}%")
    print(sep)
    print(f"  {'Mean return':<30} {result['mean_return_pct']:>+.1f}%")
    print(f"  {'Std of outcomes':<30} {result['std_return_pct']:>.1f}%")
    print(f"  {'Probability of loss':<30} {result['prob_loss_pct']:>.1f}%")
    print(f"{'═'*48}\n")

    if result["prob_loss_pct"] > 40:
        logger.warning(
            f"Monte Carlo: {result['prob_loss_pct']:.1f}% probability of loss "
            f"— strategy may not be robust."
        )

    return result


# ── WFV Summary Printer ───────────────────────────────────────────────────────
def print_wfv_comparison(reports: dict):
    """
    Print a side-by-side WFV metric comparison across multiple strategies.

    Args:
        reports: dict of strategy_name → WFVReport
    """
    sep    = "═" * (38 + 18 * len(reports))
    names  = list(reports.keys())
    cw     = 18

    print(f"\n{sep}")
    print(f"  WFV STRATEGY COMPARISON  —  {len(next(iter(reports.values())).folds)} Folds")
    print(sep)
    print(f"  {'Metric':<36}" + "".join(f"{n:>{cw}}" for n in names))
    print(f"  {'─'*36}" + "─"*(cw*len(names)))

    rows = [
        ("Mean Sharpe",           lambda r: f"{r.mean_sharpe:>+.3f}"),
        ("Mean Return %",         lambda r: f"{r.mean_return:>+.2f}%"),
        ("Profitable Folds %",    lambda r: f"{r.pct_profitable_folds:>.1f}%"),
        ("Consistency Score",     lambda r: f"{r.consistency_score:>+.3f}"),
        ("N Folds",               lambda r: f"{len(r.folds):>}"),
    ]

    for label, fn in rows:
        row = f"  {label:<36}"
        for name in names:
            try:    row += f"{fn(reports[name]):>{cw}}"
            except: row += f"{'N/A':>{cw}}"
        print(row)

    # Pass/fail per strategy
    print(f"\n  {'─'*36}" + "─"*(cw*len(names)))
    for label, test_fn in [
        ("Mean Sharpe > 1.0",      lambda r: r.mean_sharpe > 1.0),
        ("Profitable Folds > 60%", lambda r: r.pct_profitable_folds > 60),
        ("Consistency > 0.5",      lambda r: r.consistency_score > 0.5),
    ]:
        row = f"  {label:<36}"
        for name in names:
            try:    row += f"{'✓ PASS' if test_fn(reports[name]) else '✗ FAIL':>{cw}}"
            except: row += f"{'N/A':>{cw}}"
        print(row)

    print(f"{sep}\n")