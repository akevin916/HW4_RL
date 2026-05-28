"""
LunarLander DQN 訓練主程式

使用 Stable-Baselines3 DQN + Gymnasium LunarLander-v3（離散動作）

動作空間：Discrete(4)
  0: 不噴射
  1: 噴射左引擎
  2: 噴射主引擎（向上）
  3: 噴射右引擎

觀測空間：Box(8,) — 位置(x,y)、速度(vx,vy)、角度、角速度、左腳/右腳接觸
成功標準：episode 總獎勵 ≥ 200
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from utils.callbacks import EpisodeRewardCallback


def make_lunar_env(seed: int = 0):
    """建立 LunarLander-v3（離散版）環境。"""
    env = gym.make("LunarLander-v3")
    env = Monitor(env)
    return env


def train_lunar_dqn(
    learning_rate: float = 1e-4,
    policy_kwargs: dict | None = None,
    total_timesteps: int = 300_000,
    seed: int = 42,
    save_path: str | None = None,
) -> tuple:
    """
    訓練 LunarLander DQN agent。

    Args:
        learning_rate:   學習率
        policy_kwargs:   網路結構（e.g. dict(net_arch=[64, 64])）
        total_timesteps: 訓練總步數
        seed:            隨機種子
        save_path:       模型儲存路徑

    Returns:
        (model, episode_rewards)
    """
    env = make_lunar_env(seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=50_000,
        learning_starts=10_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
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


if __name__ == "__main__":
    model, rewards = train_lunar_dqn(
        total_timesteps=300_000,
        save_path=os.path.join(os.path.dirname(__file__), "results", "default", "lunar_dqn"),
    )
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    print(f"\n訓練完成 | Episodes: {len(rewards)} | 最後 100 ep 平均: {np.mean(last_100):.2f}")
