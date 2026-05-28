"""
LunarLander PPO 訓練主程式

使用 Stable-Baselines3 PPO + Gymnasium LunarLander-v3
支援離散（continuous=False）與連續動作空間（continuous=True）。

離散動作（Discrete(4)）：
  PPO 使用 CategoricalDistribution → actor 輸出 softmax 機率

連續動作（Box(-1,1,(2,))）：
  PPO 使用 DiagGaussianDistribution → actor 輸出 μ 與 log σ
  action[0]: 主引擎推力（-1 關閉，>0 開啟）
  action[1]: 側向引擎推力（-1 左，+1 右）
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from utils.callbacks import EpisodeRewardCallback


def make_lunar_env(continuous: bool = False, seed: int = 0):
    """
    建立 LunarLander-v3 環境。

    Args:
        continuous: True → 連續動作空間 Box(-1,1,(2,))
                    False → 離散動作空間 Discrete(4)
        seed:       隨機種子（gymnasium 透過 env.reset(seed=...) 設定）
    """
    env = gym.make("LunarLander-v3", continuous=continuous)
    env = Monitor(env)
    return env


def train_lunar_ppo(
    learning_rate: float = 3e-4,
    policy_kwargs: dict | None = None,
    total_timesteps: int = 300_000,
    continuous: bool = False,
    seed: int = 42,
    save_path: str | None = None,
) -> tuple:
    """
    訓練 LunarLander PPO agent。

    Args:
        learning_rate:   學習率
        policy_kwargs:   網路結構（e.g. dict(net_arch=[64, 64])）
        total_timesteps: 訓練總步數
        continuous:      是否使用連續動作空間
        seed:            隨機種子
        save_path:       模型儲存路徑

    Returns:
        (model, episode_rewards)
    """
    env = make_lunar_env(continuous=continuous, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,          # 每次 rollout 收集的步數
        batch_size=64,
        n_epochs=10,           # 每個 rollout 更新網路的次數
        gamma=0.99,
        gae_lambda=0.95,       # Generalized Advantage Estimation λ
        clip_range=0.2,        # PPO clip 係數
        ent_coef=0.01,         # entropy bonus（鼓勵探索）
        vf_coef=0.5,           # value function loss 權重
        max_grad_norm=0.5,
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
    model, rewards = train_lunar_ppo(
        total_timesteps=300_000,
        save_path=os.path.join(os.path.dirname(__file__), "results", "default", "lunar_ppo"),
    )
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    print(f"\n訓練完成 | Episodes: {len(rewards)} | 最後 100 ep 平均: {np.mean(last_100):.2f}")
