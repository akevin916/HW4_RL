"""
實驗 L-1：DQN vs PPO 在 LunarLander-v3（離散版）的比較

研究問題：Value-based 方法（DQN）與 Policy-based 方法（PPO）在此環境下
哪個學習更快、最終表現更佳？

設計：
  - DQN：Off-policy，使用 replay buffer，Q-value 估計
  - PPO：On-policy，使用 surrogate objective，更新更穩定

評估指標：
  - 訓練曲線（episode reward vs episode）
  - 最終 20 回合平均 ± 標準差
  - 成功著陸率（reward ≥ 200）
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from train_dqn import train_lunar_dqn
from train_ppo import train_lunar_ppo
from utils.plot import plot_comparison
from utils.evaluate import eval_full

# ── 實驗參數 ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 300_000
SEED = 42
RESULTS_DIR = os.path.join(_DIR, "results", "L1_algo_compare")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    # ── 訓練 DQN ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  L-1 | Training DQN on LunarLander-v3\n{'='*60}")
    dqn_model, dqn_rewards = train_lunar_dqn(
        total_timesteps=TOTAL_TIMESTEPS,
        seed=SEED,
        save_path=os.path.join(RESULTS_DIR, "model_dqn"),
    )

    # ── 訓練 PPO ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  L-1 | Training PPO on LunarLander-v3\n{'='*60}")
    ppo_model, ppo_rewards = train_lunar_ppo(
        total_timesteps=TOTAL_TIMESTEPS,
        seed=SEED,
        save_path=os.path.join(RESULTS_DIR, "model_ppo"),
    )

    # ── 儲存訓練曲線資料 ──────────────────────────────────────────────────────
    np.save(os.path.join(RESULTS_DIR, "rewards_dqn.npy"), np.array(dqn_rewards))
    np.save(os.path.join(RESULTS_DIR, "rewards_ppo.npy"), np.array(ppo_rewards))

    # ── 最終評估（各 20 回合）────────────────────────────────────────────────
    eval_env = Monitor(gym.make("LunarLander-v3"))
    dqn_mean, dqn_std, dqn_sr = eval_full(dqn_model, eval_env, n_episodes=20, threshold=200.0)
    eval_env.close()

    eval_env = Monitor(gym.make("LunarLander-v3"))
    ppo_mean, ppo_std, ppo_sr = eval_full(ppo_model, eval_env, n_episodes=20, threshold=200.0)
    eval_env.close()

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results={"DQN": dqn_rewards, "PPO": ppo_rewards},
        title="Experiment L-1: DQN vs PPO on LunarLander-v3",
        save_path=os.path.join(RESULTS_DIR, "comparison_L1.png"),
        window=20,
    )

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    dqn_tail = dqn_rewards[-100:] if len(dqn_rewards) >= 100 else dqn_rewards
    ppo_tail = ppo_rewards[-100:] if len(ppo_rewards) >= 100 else ppo_rewards

    df = pd.DataFrame([
        {
            "Algorithm":            "DQN",
            "Total Episodes":       len(dqn_rewards),
            "Last-100 Mean":        round(float(np.mean(dqn_tail)), 1),
            "Eval Mean ± Std":      f"{dqn_mean:.1f} ± {dqn_std:.1f}",
            "Success Rate (≥200)":  f"{dqn_sr*100:.1f}%",
        },
        {
            "Algorithm":            "PPO",
            "Total Episodes":       len(ppo_rewards),
            "Last-100 Mean":        round(float(np.mean(ppo_tail)), 1),
            "Eval Mean ± Std":      f"{ppo_mean:.1f} ± {ppo_std:.1f}",
            "Success Rate (≥200)":  f"{ppo_sr*100:.1f}%",
        },
    ])
    df.to_csv(os.path.join(RESULTS_DIR, "summary_L1.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment L-1 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
