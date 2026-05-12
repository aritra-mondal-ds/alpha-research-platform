# crypto/src/models/rl/trading_env.py
"""
BTC/USDT Gymnasium trading environment for PPO.

State space:
    Concatenation of:
    1. Last `window` bars of normalised feature values  (window × n_features)
    2. Portfolio state vector (3 scalars):
         - position_held:   1.0 if currently long, 0.0 if flat
         - unrealised_pnl:  (current_price - entry_price) / entry_price
         - cash_ratio:      cash / total_portfolio_value

Action space:
    Discrete(3):
        0 → EXIT  (sell if long, do nothing if flat)
        1 → HOLD  (no change)
        2 → BUY   (buy if flat, do nothing if long)

Reward:
    log_return_of_portfolio - trade_penalty
    trade_penalty = 0.001 on any action != HOLD
    This encourages the agent to only trade when confident.

Episode:
    One full pass through the dataset.
    Terminates when portfolio value < 10% of initial (hard stop).

Lookahead contract:
    Observation at step t uses features[t-window+1 .. t] only.
    No future prices or labels in the state.
"""

import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
import config.settings as cfg


class BTCTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df:              pd.DataFrame,
        feature_cols:    list,
        window:          int   = 30,
        initial_capital: float = cfg.INITIAL_CAPITAL,
        commission:      float = cfg.COMMISSION,
        slippage:        float = cfg.SLIPPAGE,
        max_position:    float = cfg.MAX_POSITION_SIZE,
        trade_penalty:   float = 0.001,
        min_capital_pct: float = 0.10,    # terminate if portfolio < 10%
    ):
        super().__init__()
        self.df              = df.reset_index(drop=True)
        self.feature_cols    = feature_cols
        self.n_features      = len(feature_cols)
        self.window          = window
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        self.max_position    = max_position
        self.trade_penalty   = trade_penalty
        self.min_capital_pct = min_capital_pct
        self.n_steps         = len(df)

        # ── Action space: EXIT=0, HOLD=1, BUY=2 ──────────────────────────────
        self.action_space = spaces.Discrete(3)

        # ── Observation space: flattened window + 3 portfolio scalars ─────────
        obs_dim = window * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state (initialised in reset)
        self.current_step  = 0
        self.cash          = 0.0
        self.position      = 0.0
        self.entry_price   = 0.0
        self.portfolio_log = []
        self.trade_log     = []

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window         # start after first window
        self.cash         = self.initial_capital
        self.position     = 0.0
        self.entry_price  = 0.0
        self.portfolio_log = [self.initial_capital]
        self.trade_log     = []
        return self._get_obs(), {}

    def step(self, action: int):
        price     = self._price(self.current_step)
        prev_val  = self._portfolio_value(price)

        # ── Execute action ────────────────────────────────────────────────────
        acted = False

        if action == 2 and self.position == 0:        # BUY
            exec_price   = price * (1 + self.slippage)
            units        = (self.cash * self.max_position) / exec_price
            cost         = units * exec_price * (1 + self.commission)
            if cost <= self.cash:
                self.position    = units
                self.cash       -= cost
                self.entry_price = exec_price
                self.trade_log.append({
                    "step": self.current_step, "type": "buy",
                    "price": exec_price, "units": units,
                })
                acted = True

        elif action == 0 and self.position > 0:        # EXIT
            exec_price = price * (1 - self.slippage)
            proceeds   = self.position * exec_price * (1 - self.commission)
            pnl        = proceeds - self.position * self.entry_price
            self.trade_log.append({
                "step": self.current_step, "type": "sell",
                "price": exec_price, "pnl": pnl,
                "return_pct": pnl / (self.position * self.entry_price) * 100,
            })
            self.cash     += proceeds
            self.position  = 0.0
            acted = True

        # ── Advance step ──────────────────────────────────────────────────────
        self.current_step += 1
        new_price  = self._price(self.current_step)
        new_val    = self._portfolio_value(new_price)
        self.portfolio_log.append(new_val)

        # ── Reward ────────────────────────────────────────────────────────────
        log_ret    = np.log(new_val / (prev_val + 1e-9))
        penalty    = -self.trade_penalty if acted else 0.0
        reward     = float(log_ret + penalty)

        # ── Termination conditions ────────────────────────────────────────────
        terminated = (
            self.current_step >= self.n_steps - 1 or
            new_val < self.initial_capital * self.min_capital_pct
        )
        truncated = False

        obs  = self._get_obs()
        info = {
            "portfolio_value": new_val,
            "position": self.position,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        price = self._price(self.current_step)
        val   = self._portfolio_value(price)
        print(
            f"Step {self.current_step:>5} | "
            f"Price: {price:>10,.2f} | "
            f"Portfolio: {val:>10,.2f} | "
            f"Position: {'LONG' if self.position > 0 else 'FLAT'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        end   = self.current_step

        # Feature window
        window_data = self.df.loc[start:end - 1, self.feature_cols].values.astype(np.float32)

        # Normalise per-window (z-score) to handle non-stationarity
        mean = window_data.mean(axis=0)
        std  = window_data.std(axis=0) + 1e-8
        norm_window = ((window_data - mean) / std).flatten()

        # Portfolio state
        price          = self._price(self.current_step)
        portfolio_val  = self._portfolio_value(price)
        position_held  = float(self.position > 0)
        unrealised_pnl = (
            (price - self.entry_price) / (self.entry_price + 1e-9)
            if self.position > 0 else 0.0
        )
        cash_ratio = self.cash / (portfolio_val + 1e-9)

        portfolio_state = np.array(
            [position_held, unrealised_pnl, cash_ratio], dtype=np.float32
        )

        return np.concatenate([norm_window, portfolio_state])

    def _price(self, step: int) -> float:
        idx = min(step, self.n_steps - 1)
        return float(self.df.loc[idx, "close"])

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.position * price

    def get_portfolio_series(self) -> pd.Series:
        """Return equity curve as a Series for analysis."""
        return pd.Series(self.portfolio_log, name="portfolio_value")

    def get_trade_summary(self) -> dict:
        sells  = [t for t in self.trade_log if t["type"] == "sell"]
        if not sells:
            return {"total_trades": 0}
        returns = [t["return_pct"] for t in sells]
        wins    = [r for r in returns if r > 0]
        return {
            "total_trades": len(sells),
            "win_rate_pct": len(wins) / len(sells) * 100,
            "avg_return_pct": np.mean(returns),
            "total_pnl": sum(t["pnl"] for t in sells),
        }