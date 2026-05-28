"""
實驗 L5：Reward Shaping 對 LunarLander-v3（PPO）的影響

研究問題：reward 函數的設計如何影響 agent 的學習速度與最終表現？

設計（全部使用 PPO，lr=3e-4，[64,64] 網路）：
  - Default:          原始 reward（dense）
  - Custom Shaped:    原始 reward + 角度懲罰 + 低空速度懲罰
  - Vertical Landing: 原始 reward + 角度獎勵（鼓勵直立）+ 側引擎懲罰減免

預期發現：
  - Default          → 穩定學習，作為基準
  - Custom Shaped    → 收斂更快，著陸動作更穩定（懲罰傾斜）
  - Vertical Landing → 更積極用側引擎調整姿態，角度更精準

理論基礎：
  Ng et al. (1999) 證明：若 shaping term F 滿足 F = γΦ(s') - Φ(s)（位能差），
  則最優策略不變。本實驗採用啟發式設計，觀察實際效果。
  角度獎勵採「獎勵正確行為」而非「懲罰錯誤」，兩者梯度方向等價，
  但獎勵版在好狀態下有更明確的正向訊號。
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_wrappers import VerticalLandingWrapper, PrecisionLandingWrapper
from utils.callbacks import EpisodeRewardCallback
from utils.plot import plot_comparison
from utils.evaluate import eval_full

# ── 實驗參數 ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 300_000
SEED = 42
RESULTS_DIR = os.path.join(_DIR, "results", "L5_reward_shaping")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_env(reward_type: str) -> gym.Env:
    """
    根據 reward_type 建立對應的 LunarLander 環境。

    Args:
        reward_type: "default" | "shaped" | "vertical"
    """
    base_env = gym.make("LunarLander-v3")
    if reward_type == "shaped":
        base_env = PrecisionLandingWrapper(
            base_env,
            center_coef=0.2,
            velocity_coef=0.1,
            height_threshold=0.5,
        )
    elif reward_type == "vertical":
        base_env = VerticalLandingWrapper(
            base_env,
            angle_reward_coef=0.1,
            engine_relief_coef=0.02,
        )
    # Monitor 必須包在最外層以正確記錄 episode reward
    return Monitor(base_env)


def train_ppo(reward_type: str, save_path: str | None = None) -> tuple:
    """使用 PPO 訓練指定 reward 類型的 LunarLander agent。"""
    env = make_env(reward_type)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=SEED,
        device="cuda",
    )

    callback = EpisodeRewardCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_path)
        print(f"[Model saved] {save_path}")

    env.close()
    return model, callback.episode_rewards


def run():
    configs = {
        "Default (Dense)":    "default",
        "Custom Shaped":      "shaped",
        "Vertical Landing":   "vertical",
    }

    all_rewards = {}
    summary_rows = []

    for label, reward_type in configs.items():
        print(f"\n{'='*60}")
        print(f"  L-5 | {label}  (reward_type='{reward_type}')")
        print(f"{'='*60}")

        model, rewards = train_ppo(
            reward_type=reward_type,
            save_path=os.path.join(RESULTS_DIR, f"model_{reward_type}"),
        )

        all_rewards[label] = rewards
        np.save(os.path.join(RESULTS_DIR, f"rewards_{reward_type}.npy"), np.array(rewards))

        # 評估時一律用原始 default 環境（公平比較實際表現）
        eval_env = Monitor(gym.make("LunarLander-v3"))
        mean, std, sr = eval_full(model, eval_env, n_episodes=20, threshold=200.0)
        eval_env.close()

        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary_rows.append({
            "Reward Design":        label,
            "Total Episodes":       len(rewards),
            "Last-100 Mean":        round(float(np.mean(tail)), 1),
            "Last-100 Std":         round(float(np.std(tail)), 1),
            "Eval Mean ± Std":      f"{mean:.1f} ± {std:.1f}",
            "Success Rate (≥200)":  f"{sr*100:.1f}%",
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment L-5: Reward Shaping Effect on PPO LunarLander-v3",
        save_path=os.path.join(RESULTS_DIR, "comparison_L5.png"),
        window=20,
    )

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_L5.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment L-5 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n注意：Eval 均使用原始 Default 環境，確保三組比較公平。")
    print(f"結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
