"""
實驗 L-2：PPO 在離散 vs 連續動作空間的比較

研究問題：動作空間的設計（離散 vs 連續）如何影響 PPO 的學習難度與最終表現？

設計：
  - 離散（Discrete(4)）：直接選擇 4 個動作之一，決策邊界清晰
  - 連續（Box(-1,1,(2,))）：輸出連續推力，需學習更精細的控制

預期：
  - 離散版通常更容易學習（探索空間較小）
  - 連續版若能收斂，最終控制精度可能更高
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
RESULTS_DIR = os.path.join(_DIR, "results", "L2_action_space")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = {
    "PPO Discrete  (Discrete(4))":          {"continuous": False},
    "PPO Continuous (Box(-1,1,(2,)))":      {"continuous": True},
}


def run():
    all_rewards = {}
    summary_rows = []

    for label, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  L-2 | {label}")
        print(f"{'='*60}")

        safe = "discrete" if not cfg["continuous"] else "continuous"
        model, rewards = train_lunar_ppo(
            total_timesteps=TOTAL_TIMESTEPS,
            continuous=cfg["continuous"],
            seed=SEED,
            save_path=os.path.join(RESULTS_DIR, f"model_{safe}"),
        )

        all_rewards[label] = rewards
        np.save(os.path.join(RESULTS_DIR, f"rewards_{safe}.npy"), np.array(rewards))

        # 對應的 eval 環境需使用相同的 continuous 設定
        eval_env = Monitor(gym.make("LunarLander-v3", continuous=cfg["continuous"]))
        mean, std, sr = eval_full(model, eval_env, n_episodes=20, threshold=200.0)
        eval_env.close()

        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary_rows.append({
            "Configuration":        label,
            "Action Space":         "Continuous" if cfg["continuous"] else "Discrete",
            "Total Episodes":       len(rewards),
            "Last-100 Mean":        round(float(np.mean(tail)), 1),
            "Eval Mean ± Std":      f"{mean:.1f} ± {std:.1f}",
            "Success Rate (≥200)":  f"{sr*100:.1f}%",
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment L-2: PPO Discrete vs Continuous Action Space",
        save_path=os.path.join(RESULTS_DIR, "comparison_L2.png"),
        window=20,
    )

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_L2.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment L-2 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
