"""
Pong DQN 訓練主程式

使用 Stable-Baselines3 DQN + Gymnasium ALE/Pong-v5
預處理流程（由 make_atari_env 自動套用）：
  1. NoopResetEnv   — 開局隨機 noop 避免學到固定策略
  2. MaxAndSkipEnv  — 每 4 幀做一次決策（frame skip）
  3. EpisodicLifeEnv — 失去一條命視為 episode 結束
  4. WarpFrame      — 灰階化 + 縮放至 84×84
  5. ClipRewardEnv  — 獎勵截斷至 {-1, 0, +1}
VecFrameStack 在此之上疊加多幀以提供速度資訊。
"""
import os
import sys

# 將專案根目錄加入 sys.path 以便匯入 utils
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from utils.callbacks import EpisodeRewardCallback

# 向 gymnasium 註冊 ALE Atari 環境
gym.register_envs(ale_py)


def make_pong_env(n_stack: int = 4, seed: int = 0):
    """
    建立 Pong 向量化環境。

    Args:
        n_stack: frame stacking 幀數（1 = 無疊幀，4 = 標準疊幀）
        seed:    隨機種子

    Returns:
        VecFrameStack 包裝後的 VecEnv
    """
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=n_stack)
    return env


def train_pong_dqn(
    n_stack: int = 4,
    exploration_fraction: float = 0.1,
    total_timesteps: int = 500_000,
    seed: int = 42,
    save_path: str | None = None,
) -> tuple:
    """
    訓練 Pong DQN agent。

    Args:
        n_stack:              Frame stacking 幀數
        exploration_fraction: ε 從 1.0 衰減至 exploration_final_eps
                              所花步數佔 total_timesteps 的比例
                              （0.1 → 50k steps，0.4 → 200k steps）
        total_timesteps:      訓練總步數
        seed:                 隨機種子
        save_path:            模型儲存路徑（None 則不儲存）

    Returns:
        (model, episode_rewards)
        model:           訓練完成的 SB3 DQN 模型
        episode_rewards: 每個 episode 的總獎勵列表
    """
    env = make_pong_env(n_stack=n_stack, seed=seed)

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=50_000,   # 先收集 50k 步再開始訓練
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,             # 每 4 步更新一次網路
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        verbose=1,
        seed=seed,
        device="cuda",
    )

    callback = EpisodeRewardCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_path)
        print(f"[Model saved] {save_path}")

    env.close()
    return model, callback.episode_rewards


# ── 單獨執行：快速驗證訓練流程是否正常 ────────────────────────────────────────
if __name__ == "__main__":
    model, rewards = train_pong_dqn(
        n_stack=4,
        exploration_fraction=0.1,
        total_timesteps=500_000,
        save_path=os.path.join(os.path.dirname(__file__), "results", "default", "pong_dqn"),
    )
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    print(f"\n訓練完成 | Episodes: {len(rewards)} | 最後 100 ep 平均: {np.mean(last_100):.2f}")
