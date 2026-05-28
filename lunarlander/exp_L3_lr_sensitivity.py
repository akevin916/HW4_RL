"""
實驗 L-3：學習率對 PPO 訓練穩定性與收斂速度的影響

研究問題：PPO 的學習率大小如何影響訓練穩定度與最終表現？

設計（全部使用離散 LunarLander-v3，相同架構）：
  - lr = 1e-4（小）：更新步伐保守，收斂慢但穩定
  - lr = 3e-4（中，PPO 預設值）：通常表現最好的甜蜜點
  - lr = 1e-3（大）：更新激進，可能不穩定或振盪

超參數敏感度分析是實務 RL 訓練中非常重要的一環。
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

from train_ppo import train_lunar_ppo
from utils.plot import plot_comparison
from utils.evaluate import eval_full

# ── 實驗參數 ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 300_000
SEED = 42
RESULTS_DIR = os.path.join(_DIR, "results", "L3_lr_sensitivity")
os.makedirs(RESULTS_DIR, exist_ok=True)

LEARNING_RATES = [1e-4, 3e-4, 1e-3]


def run():
    all_rewards = {}
    summary_rows = []

    for lr in LEARNING_RATES:
        label = f"lr = {lr:.0e}"
        print(f"\n{'='*60}")
        print(f"  L-3 | {label}")
        print(f"{'='*60}")

        safe = f"lr_{str(lr).replace('.', '').replace('-', 'm').replace('+', '')}"
        model, rewards = train_lunar_ppo(
            learning_rate=lr,
            total_timesteps=TOTAL_TIMESTEPS,
            continuous=False,
            seed=SEED,
            save_path=os.path.join(RESULTS_DIR, f"model_{safe}"),
        )

        all_rewards[label] = rewards
        np.save(os.path.join(RESULTS_DIR, f"rewards_{safe}.npy"), np.array(rewards))

        eval_env = Monitor(gym.make("LunarLander-v3"))
        mean, std, sr = eval_full(model, eval_env, n_episodes=20, threshold=200.0)
        eval_env.close()

        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary_rows.append({
            "Learning Rate":        lr,
            "Total Episodes":       len(rewards),
            "Last-100 Mean":        round(float(np.mean(tail)), 1),
            "Last-100 Std":         round(float(np.std(tail)), 1),
            "Eval Mean ± Std":      f"{mean:.1f} ± {std:.1f}",
            "Success Rate (≥200)":  f"{sr*100:.1f}%",
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment L-3: Learning Rate Sensitivity for PPO on LunarLander-v3",
        save_path=os.path.join(RESULTS_DIR, "comparison_L3.png"),
        window=20,
    )

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_L3.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment L-3 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
