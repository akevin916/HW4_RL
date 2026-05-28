"""
評估工具：對訓練好的 agent 執行多回合測試，回傳統計指標。

使用 SB3 內建 evaluate_policy，支援 VecEnv 與一般 Env。
"""
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


def eval_full(
    model,
    eval_env,
    n_episodes: int = 20,
    threshold: float = 200.0,
    deterministic: bool = True,
) -> tuple:
    """
    一次呼叫同時計算平均獎勵、標準差、成功率。

    Args:
        model:        訓練好的 SB3 模型
        eval_env:     評估用環境（建議用 Monitor 包裝）
        n_episodes:   評估回合數
        threshold:    成功判斷門檻（LunarLander 為 200，Pong 可設 0）
        deterministic: 是否使用確定性策略（eval 通常設 True）

    Returns:
        (mean_reward, std_reward, success_rate)
        mean_reward:  n_episodes 的平均總獎勵
        std_reward:   標準差
        success_rate: 達到 threshold 的回合比例 [0.0, 1.0]
    """
    ep_rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        return_episode_rewards=True,
        deterministic=deterministic,
    )
    ep_rewards = np.array(ep_rewards, dtype=float)
    return (
        float(np.mean(ep_rewards)),
        float(np.std(ep_rewards)),
        float(np.mean(ep_rewards >= threshold)),
    )
