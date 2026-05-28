"""
實驗 P-1：Frame Stacking 對 Pong DQN 的影響

研究問題：提供多幀歷史資訊（疊幀）是否讓 agent 更容易學到有效策略？

設計：
  - 對照組：n_stack=1（單幀，agent 看不出球的移動方向）
  - 實驗組：n_stack=4（標準疊幀，agent 可從幀差推算速度）

預期：n_stack=4 收斂更快、最終分數更高，因為球的速度資訊是 Pong 的關鍵狀態。
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)                          # pong/ 本地匯入
sys.path.insert(0, os.path.dirname(_DIR))         # 專案根目錄（utils）

import numpy as np
import pandas as pd

from train_dqn import train_pong_dqn
from utils.plot import plot_comparison

# ── 實驗參數 ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
SEED = 42
RESULTS_DIR = os.path.join(_DIR, "results", "P1_frame_stack")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = {
    "1 Frame (No Stacking)": {"n_stack": 1},
    "4 Frames (Standard)":   {"n_stack": 4},
}


def run():
    all_rewards = {}
    summary_rows = []

    for label, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  P-1 | {label}")
        print(f"{'='*60}")

        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        model, rewards = train_pong_dqn(
            n_stack=cfg["n_stack"],
            exploration_fraction=0.1,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=SEED,
            save_path=os.path.join(RESULTS_DIR, f"model_{safe}"),
        )

        all_rewards[label] = rewards
        np.save(os.path.join(RESULTS_DIR, f"rewards_{safe}.npy"), np.array(rewards))

        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary_rows.append({
            "Configuration":       label,
            "n_stack":             cfg["n_stack"],
            "Total Episodes":      len(rewards),
            "Last-100 Mean":       round(float(np.mean(tail)), 2),
            "Last-100 Std":        round(float(np.std(tail)), 2),
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment P-1: Effect of Frame Stacking on Pong DQN",
        save_path=os.path.join(RESULTS_DIR, "comparison_P1.png"),
        window=20,
    )

    # ── 結果表 ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_P1.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment P-1 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
