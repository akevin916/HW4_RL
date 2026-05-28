"""
實驗 L-4：神經網路隱藏層大小對 PPO 表現的影響

研究問題：更大的網路容量是否帶來更好的 LunarLander 表現？

設計（全部使用 PPO 離散 LunarLander-v3，lr=3e-4）：
  - 小型網路 [64, 64]：2 個隱藏層各 64 個神經元，快速訓練
  - 大型網路 [256, 256]：2 個隱藏層各 256 個神經元，更高容量

討論重點：
  - 容量過小 → underfitting，無法表達複雜策略
  - 容量過大 → 可能 overfitting 或收斂慢，但對 LunarLander 此問題不大
  - sample efficiency 的差異（大網路每步計算更慢但可能需要更少 episode）
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
RESULTS_DIR = os.path.join(_DIR, "results", "L4_network_size")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = {
    "Small Network [64, 64]":   {"net_arch": [64, 64]},
    "Large Network [256, 256]": {"net_arch": [256, 256]},
}


def run():
    all_rewards = {}
    summary_rows = []

    for label, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  L-4 | {label}")
        print(f"{'='*60}")

        arch = cfg["net_arch"]
        safe = f"net_{'x'.join(map(str, arch))}"

        # SB3 policy_kwargs: net_arch 指定 actor 與 critic 共享的隱藏層
        model, rewards = train_lunar_ppo(
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=arch),
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

        # 計算模型參數量
        n_params = sum(p.numel() for p in model.policy.parameters())

        summary_rows.append({
            "Network Arch":         str(arch),
            "# Parameters":         f"{n_params:,}",
            "Total Episodes":       len(rewards),
            "Last-100 Mean":        round(float(np.mean(tail)), 1),
            "Last-100 Std":         round(float(np.std(tail)), 1),
            "Eval Mean ± Std":      f"{mean:.1f} ± {std:.1f}",
            "Success Rate (≥200)":  f"{sr*100:.1f}%",
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment L-4: Network Size Effect on PPO for LunarLander-v3",
        save_path=os.path.join(RESULTS_DIR, "comparison_L4.png"),
        window=20,
    )

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_L4.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment L-4 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
