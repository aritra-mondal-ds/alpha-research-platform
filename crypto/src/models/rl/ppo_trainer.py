# crypto/src/models/rl/ppo_trainer.py
"""
PPO training pipeline using Stable-Baselines3.

Architecture:
    Policy: MlpPolicy — fully connected network
    Net arch: [256, 256] for both actor and critic
    Environment: BTCTradingEnv wrapped in VecNormalize

Training notes:
    - 4 parallel environments for data throughput
    - VecNormalize on observations (not rewards) — financial rewards
      are already on a sensible scale via log returns
    - EvalCallback saves the best model based on mean episode reward
    - Policy network runs on CUDA (RTX 3060)
    - CPU handles the 4 env workers (i5-12400F)

PPO hyperparameters rationale:
    n_steps=2048:      collect 2048 steps per env before each update
    batch_size=256:    mini-batch size for gradient update
    n_epochs=10:       gradient steps per PPO update
    gamma=0.99:        long-horizon discounting for multi-day positions
    ent_coef=0.01:     encourage exploration (avoid early convergence)
    clip_range=0.2:    standard PPO clipping
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env  import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)

sys.path.append(str(Path(__file__).resolve().parents[3]))
import config.settings as cfg
from src.models.rl.trading_env import BTCTradingEnv


# ── Progress callback ─────────────────────────────────────────────────────────
class TrainingProgressCallback(BaseCallback):
    """Logs episode stats every N timesteps."""

    def __init__(self, log_freq: int = 20_000):
        super().__init__()
        self.log_freq   = log_freq
        self.ep_rewards = []

    def _on_step(self) -> bool:
        if "episode" in self.locals.get("infos", [{}])[0]:
            ep = self.locals["infos"][0]["episode"]
            self.ep_rewards.append(ep["r"])

        if self.num_timesteps % self.log_freq == 0 and self.ep_rewards:
            mean_r = np.mean(self.ep_rewards[-20:])
            logger.info(
                f"Timesteps: {self.num_timesteps:>8,} | "
                f"Mean ep reward (last 20): {mean_r:>+.4f}"
            )
        return True


# ── PPO Trainer ───────────────────────────────────────────────────────────────
class PPOTradingAgent:

    def __init__(self):
        self.model      = None
        self.vec_env    = None
        self.eval_env   = None

    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        df_train:       pd.DataFrame,
        df_val:         pd.DataFrame,
        feature_cols:   list,
        total_timesteps: int = 500_000,
        n_envs:         int  = 4,
        window:         int  = 30,
    ) -> "PPOTradingAgent":
        """
        Train PPO agent on training data, evaluate on validation data.

        Args:
            df_train:        Training feature DataFrame (with OHLCV columns).
            df_val:          Validation feature DataFrame.
            feature_cols:    List of feature column names for the state.
            total_timesteps: Total environment steps to train for.
            n_envs:          Number of parallel environments (CPU workers).
            window:          Observation window length in bars.
        """
        out_dir = cfg.MODEL_DIR / "ppo"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PPO training | timesteps: {total_timesteps:,} | "
            f"envs: {n_envs} | window: {window}"
        )
        logger.info(
            f"Train: {len(df_train):,} rows | "
            f"Val: {len(df_val):,} rows | "
            f"Features: {len(feature_cols)}"
        )

        # ── Vectorised training environments ──────────────────────────────────
        def make_train_env():
            return BTCTradingEnv(
                df           = df_train,
                feature_cols = feature_cols,
                window       = window,
            )

        self.vec_env = make_vec_env(make_train_env, n_envs=n_envs)
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs     = True,
            norm_reward  = False,   # keep reward on log-return scale
            clip_obs     = 10.0,
            training     = True,
        )

        # ── Evaluation environment ────────────────────────────────────────────
        def make_val_env():
            return BTCTradingEnv(
                df           = df_val,
                feature_cols = feature_cols,
                window       = window,
            )

        self.eval_env = VecNormalize(
            make_vec_env(make_val_env, n_envs=1),
            norm_obs    = True,
            norm_reward = False,
            training    = False,    # do not update stats during eval
        )
        # Share normalisation stats between train and eval env
        self.eval_env.obs_rms    = self.vec_env.obs_rms
        self.eval_env.ret_rms    = self.vec_env.ret_rms

        # ── PPO model ─────────────────────────────────────────────────────────
        self.model = PPO(
            policy        = "MlpPolicy",
            env           = self.vec_env,
            learning_rate = 3e-4,
            n_steps       = 2048,
            batch_size    = 256,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            policy_kwargs = dict(net_arch=[256, 256]),
            verbose       = 1,
            seed          = cfg.RANDOM_SEED,
            device        = "cuda",    # RTX 3060 for policy network
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals = 5,
            min_evals                = 10,
            verbose                  = 1,
        )
        eval_cb = EvalCallback(
            eval_env          = self.eval_env,
            best_model_save_path = str(out_dir / "best"),
            log_path          = str(out_dir / "eval_logs"),
            eval_freq         = max(10_000 // n_envs, 1),
            n_eval_episodes   = 3,
            deterministic     = True,
            callback_after_eval = stop_cb,
            verbose           = 1,
        )
        progress_cb = TrainingProgressCallback(log_freq=20_000)

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info("Starting PPO training ...")
        self.model.learn(
            total_timesteps = total_timesteps,
            callback        = [eval_cb, progress_cb],
            progress_bar    = True,
        )

        # ── Save model + normalisation stats ──────────────────────────────────
        self.save(out_dir)
        return self

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        df_test:      pd.DataFrame,
        feature_cols: list,
        window:       int = 30,
        n_episodes:   int = 1,
    ) -> dict:
        """
        Run the trained agent on the test set deterministically.
        Returns performance metrics dict.
        """
        assert self.model is not None, "Call .train() or .load() first."

        all_metrics = []
        for ep in range(n_episodes):
            env = BTCTradingEnv(
                df           = df_test,
                feature_cols = feature_cols,
                window       = window,
            )
            obs, _    = env.reset()
            done      = False
            positions = []

            while not done:
                # Normalise obs using training stats if vec_env available
                if self.vec_env is not None:
                    obs_norm = self.vec_env.normalize_obs(obs)
                else:
                    obs_norm = obs
                action, _ = self.model.predict(obs_norm, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                positions.append(info.get("position", 0))

            equity  = env.get_portfolio_series()
            summary = env.get_trade_summary()
            metrics = self._compute_metrics(equity, df_test, summary)
            all_metrics.append(metrics)

        # Average over episodes (for n_episodes > 1)
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        self._print_metrics(avg)
        return avg

    def _compute_metrics(
        self,
        equity:   pd.Series,
        df:       pd.DataFrame,
        summary:  dict,
    ) -> dict:
        ann       = 8760
        returns   = equity.pct_change().dropna()
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1
        n_years   = len(returns) / ann
        ann_ret   = (1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1

        bh_ret   = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        mean_r   = returns.mean() * ann
        std_r    = returns.std() * np.sqrt(ann)
        down_r   = returns[returns < 0].std() * np.sqrt(ann)

        peak   = equity.cummax()
        dd     = (equity - peak) / peak * 100
        max_dd = dd.min()

        return {
            "total_return_pct": round(total_ret * 100, 2),
            "ann_return_pct":   round(ann_ret * 100, 2),
            "buy_hold_pct":     round(bh_ret * 100, 2),
            "sharpe":           round(mean_r / std_r if std_r > 0 else 0, 4),
            "sortino":          round(mean_r / down_r if down_r > 0 else 0, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar":           round(ann_ret * 100 / abs(max_dd) if max_dd != 0 else 0, 4),
            "total_trades":     summary.get("total_trades", 0),
            "win_rate_pct":     round(summary.get("win_rate_pct", 0), 2),
            "avg_return_pct":   round(summary.get("avg_return_pct", 0), 4),
        }

    def _print_metrics(self, m: dict):
        sep = "─" * 50
        print(f"\n{'═'*50}")
        print(f"  PPO Agent Evaluation")
        print(f"{'═'*50}")
        print(f"  {'Total Return':<28} {m['total_return_pct']:>+.2f}%")
        print(f"  {'Ann. Return':<28} {m['ann_return_pct']:>+.2f}%")
        print(f"  {'Buy & Hold':<28} {m['buy_hold_pct']:>+.2f}%")
        print(sep)
        print(f"  {'Sharpe Ratio':<28} {m['sharpe']:>.3f}")
        print(f"  {'Sortino Ratio':<28} {m['sortino']:>.3f}")
        print(f"  {'Calmar Ratio':<28} {m['calmar']:>.3f}")
        print(sep)
        print(f"  {'Max Drawdown':<28} {m['max_drawdown_pct']:>.2f}%")
        print(f"  {'Total Trades':<28} {m['total_trades']:>}")
        print(f"  {'Win Rate':<28} {m['win_rate_pct']:>.1f}%")
        print(f"  {'Avg Trade Return':<28} {m['avg_return_pct']:>+.4f}%")
        print(f"{'═'*50}\n")

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, out_dir: Path):
        self.model.save(str(out_dir / "ppo_model"))
        if self.vec_env is not None:
            self.vec_env.save(str(out_dir / "vec_normalize.pkl"))
        logger.info(f"PPO model saved → {out_dir}")

    def load(self, feature_cols: list, window: int = 30) -> "PPOTradingAgent":
        out_dir   = cfg.MODEL_DIR / "ppo"
        model_path = out_dir / "best" / "best_model.zip"
        norm_path  = out_dir / "vec_normalize.pkl"

        if not model_path.exists():
            # Fall back to last saved model
            model_path = out_dir / "ppo_model.zip"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No PPO model at {out_dir}. Run --mode train-rl first."
            )

        # Rebuild a dummy env to satisfy SB3 load requirement
        dummy_df = pd.DataFrame({
            **{c: np.zeros(window + 10) for c in feature_cols},
            "close": np.ones(window + 10) * 50000.0,
            "open":  np.ones(window + 10) * 50000.0,
        })
        dummy_env = make_vec_env(
            lambda: BTCTradingEnv(dummy_df, feature_cols, window), n_envs=1
        )
        if norm_path.exists():
            dummy_env = VecNormalize.load(str(norm_path), dummy_env)
            dummy_env.training = False
        self.vec_env = dummy_env
        self.model   = PPO.load(str(model_path), env=dummy_env)
        logger.info(f"PPO model loaded ← {model_path}")
        return self