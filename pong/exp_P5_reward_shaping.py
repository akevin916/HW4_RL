"""
實驗 P-5：Reward 設計對 Pong DQN 的影響

研究問題：改變 Atari Pong 的 reward 設計是否影響 agent 的學習行為？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pong Reward 機制與設計說明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原始 ALE Reward（無任何處理）：
  +1   → 對手未接到球（我方得分）
  -1   → 己方未接到球（對手得分）
   0   → 其他所有 step（約佔 99% 以上的 step）

問題：reward 極度稀疏。每個 episode 約 3000–5000 steps，
      得分事件只有 ~21 次（Pong 打到 21 分結束），
      其餘 step 的 gradient 更新信號為 0。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三種 Reward 設計（本實驗比較對象）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① Default（Clipped）—— 對照組
   步驟：ALE → AtariWrapper(clip_reward=True) → {-1, 0, +1}
   理論：Mnih et al. (2013) 使用此設計，讓不同遊戲的梯度尺度一致。
   特色：所有「得分事件」權重相等，不區分大贏小贏。

② No Clip（原始分數）
   步驟：ALE → AtariWrapper(clip_reward=False) → 原始 ±1
   對 Pong 而言數值與 Default 相同（ALE 本就是 ±1），
   但此設計在其他遊戲中會有顯著差異（如 Space Invaders ±50）。
   用途：說明 Clip 在 Pong 這個特例中「不影響結果」，
         反過來驗證：Clip 的主要目的是跨遊戲統一尺度而非針對 Pong。

③ Step Penalty（每步懲罰）
   步驟：ALE → AtariWrapper(clip_reward=True) → StepPenaltyWrapper(-0.001)
   設計動機：對每個非 terminal step 扣 0.001，
             讓 agent 傾向「快速得分」而非消極拖延。
   潛在問題（Reward Hacking）：
             若 agent 發現「快速輸掉」（縮短 episode）也能減少總懲罰，
             可能出現反直覺的學習行為。這正是本實驗要觀察的現象。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
預期發現
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Default ≈ No Clip   → Pong 原本就是 ±1，驗證 clip 對此遊戲影響有限
  Step Penalty 可能：
    (a) 學更快（主動進攻）→ reward shaping 成功
    (b) 表現更差（reward hacking）→ 提供 Discussion 素材
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from reward_wrappers import AggressiveWrapper, DefensiveWrapper, BalancedWrapper
from utils.callbacks import EpisodeRewardCallback
from utils.plot import plot_comparison

gym.register_envs(ale_py)

# ── 實驗參數 ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
N_STACK = 4
SEED = 42
RESULTS_DIR = os.path.join(_DIR, "results", "P5_reward_shaping")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_pong_custom(reward_type: str = "default", n_stack: int = 4, seed: int = 0):
    """
    建立 Pong 環境，支援三種策略的 reward 設計。

    Args:
        reward_type: "aggressive" | "defensive" | "balanced"
        n_stack:     frame stacking 幀數
        seed:        隨機種子
    """
    def _make():
        base_env = gym.make("ALE/Pong-v5")
        env = AtariWrapper(base_env, clip_reward=False)  # 不 clip，讓 wrapper 自己控制屬度

        if reward_type == "aggressive":
            env = AggressiveWrapper(env, streak_bonus=0.3, advantage_bonus=0.5)
        elif reward_type == "defensive":
            env = DefensiveWrapper(env, survival_bonus=0.001, hit_bonus=0.5)
        elif reward_type == "balanced":
            env = BalancedWrapper(env, hit_bonus=0.5, advantage_bonus=0.5, lose_penalty=1.2)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        return Monitor(env)

    vec_env = DummyVecEnv([_make])
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)
    return vec_env


def train_pong_reward(reward_type: str, save_path: str | None = None) -> tuple:
    """使用 DQN 訓練指定 reward 設計的 Pong agent。"""
    env = make_pong_custom(reward_type=reward_type, n_stack=N_STACK, seed=SEED)

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=50_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
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
        "Aggressive (進攻型)": "aggressive",
        "Defensive (防守型)": "defensive",
        "Balanced (平衡型)":   "balanced",
    }

    all_rewards = {}
    summary_rows = []

    for label, reward_type in configs.items():
        print(f"\n{'='*60}")
        print(f"  P-5 | {label}  (reward_type='{reward_type}')")
        print(f"{'='*60}")

        model, rewards = train_pong_reward(
            reward_type=reward_type,
            save_path=os.path.join(RESULTS_DIR, f"model_{reward_type}"),
        )

        all_rewards[label] = rewards
        np.save(os.path.join(RESULTS_DIR, f"rewards_{reward_type}.npy"), np.array(rewards))

        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary_rows.append({
            "Reward Design":  label,
            "Total Episodes": len(rewards),
            "Last-100 Mean":  round(float(np.mean(tail)), 2),
            "Last-100 Std":   round(float(np.std(tail)), 2),
        })

    # ── 比較圖 ────────────────────────────────────────────────────────────────
    plot_comparison(
        results=all_rewards,
        title="Experiment P-5: Reward Design Effect on Pong DQN",
        save_path=os.path.join(RESULTS_DIR, "comparison_P5.png"),
        window=20,
    )

    # ── 結果摘要 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_P5.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Experiment P-5 Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n注意：No Clip 在 Pong 中數值與 Default 幾乎相同（ALE 本就是±1），")
    print(f"      此設計旨在說明 ClipRewardEnv 的功能而非改變 Pong 本身的分數。")
    print(f"結果已儲存至 {RESULTS_DIR}")


if __name__ == "__main__":
    run()
