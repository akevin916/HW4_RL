"""
實驗 P-4：ε-greedy 衰減速度對 Pong DQN 的影響

研究問題：探索（exploration）持續時間的長短如何影響最終策略品質？

設計（total_timesteps = 500k）：
  - 快速衰減：exploration_fraction=0.10 → ε 在前 50k 步完成衰減
  - 慢速衰減：exploration_fraction=0.40 → ε 在前 200k 步完成衰減

直覺：
  - 衰減太快 → 過早剝削，容易陷入局部最優
  - 衰減太慢 → 探索太久，收斂速度較慢但最終策略可能更穩定
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import pandas as pd

from train_dqn import train_pong_dqn
from utils.plot import plot_comparison

# ── 實驗參數 ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
SEED = 42
RESULTS_DIR = os.path.join(_DIR, "results", "P4_epsilon")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = {
    "Fast Decay  (ε→0.01 in  50k steps)": {"exploration_fraction": 0.10},
    "Slow Decay  (ε→0.01 in 200k steps)": {"exploration_fraction": 0.40},
}


def run():
    all_rewards = {}
    summary_rows = []

    for label, cfg in CONFIGS.items():
        frac = cfg["exploration_fraction"]
        decay_steps = int(frac * TOTAL_TIMESTEPS)

        print(f"\n{'='*60}")
        print(f"  P-4 | {label}")
        print(f"  exploration_fraction={frac}  (decay steps={decay_steps:,})")
        print(f"{'='*60}")

        safe = f"eps_{str(frac).replace('.', '')}"
        model, rewards = train_pong_dqn(
            n_stack=4,
            exploration_fraction=frac,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=SEED,
            save_path=os.path.join(RESULTS_DIR, f"model_{safe}"),
        )

        all_rewards[label] = rewards
        np.save(os.path.join(RESULTS_DIR, f"rewards_{safe}.npy"), np.array(rewards))

        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary_rows.append({
            "Configuration":        label,
            "exploration_fraction": frac,
            "Decay Steps":          decay_steps,
            "Total Episodes":       len(rewards),
            "Last-100 Mean":        round(float(np.mean(tail)), 2),
            "Last-100 Std":         round(float(np.std(tail)), 2),
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment P-4: Effect of ε-Greedy Decay Rate on Pong DQN",
        save_path=os.path.join(RESULTS_DIR, "comparison_P4.png"),
        window=20,
    )

    # ── 結果表 ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_P4.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment P-4 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
