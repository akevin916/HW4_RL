"""
watch_game.py — 直接看 LunarLander agent 降落
===============================================
這支腳本會開一個視窗，讓你「肉眼看到」agent 在降落。
你可以選擇：
  - 看「完全隨機」的 agent（不需要預訓練）
  - 看「已存的訓練模型」降落（需要先跑過 L1–L5 之類的實驗）

用法：
  python lunarlander/watch_game.py                              # 隨機 agent
  python lunarlander/watch_game.py --model lunarlander/results/L1_algo_compare/model_ppo
  python lunarlander/watch_game.py --model lunarlander/results/L3_lr_sensitivity/model_lr_0.001 --algo ppo
  python lunarlander/watch_game.py --algo dqn --model lunarlander/results/L1_algo_compare/model_dqn

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
可修改的地方：
  NUM_EPISODES  — 看幾局
  FPS           — 播放速度（30 正常速，10 慢動作，0 盡量快）
  CONTINUOUS    — 連續動作空間（預設 False）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys, time, argparse

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import gymnasium as gym

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可修改的地方
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUM_EPISODES = 5      # 看幾局
FPS          = 30     # 播放速度；10 = 慢動作好看清楚，30 = 正常
CONTINUOUS   = False  # True = 連續動作（L2 實驗用的那種），False = 離散


def watch_random(n_episodes: int, continuous: bool):
    """看完全隨機 agent 降落（無需預訓練）"""
    env = gym.make("LunarLander-v3", continuous=continuous, render_mode="human")
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if FPS > 0:
                time.sleep(1.0 / FPS)
        total_rewards.append(total_reward)
        landed = "✓ 成功" if total_reward >= 200 else "✗ 失敗"
        print(f"[Random] Episode {ep+1}/{n_episodes} | steps={steps:4d} | reward={total_reward:7.1f} | {landed}")
    env.close()
    print(f"\n平均分：{np.mean(total_rewards):.1f}  (成功率：{np.mean(np.array(total_rewards)>=200)*100:.0f}%)")


def watch_trained(model_path: str, algo: str, n_episodes: int, continuous: bool):
    """看訓練好的模型降落"""
    from stable_baselines3 import DQN, PPO

    print(f"載入模型：{model_path}  (algo={algo.upper()})")
    env = gym.make("LunarLander-v3", continuous=continuous, render_mode="human")

    if algo.lower() == "ppo":
        model = PPO.load(model_path, device="cpu")
    elif algo.lower() == "dqn":
        model = DQN.load(model_path, device="cpu")
    else:
        raise ValueError(f"未知 algo：{algo}，請用 ppo 或 dqn")

    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if FPS > 0:
                time.sleep(1.0 / FPS)
        total_rewards.append(total_reward)
        landed = "✓ 成功" if total_reward >= 200 else "✗ 失敗"
        print(f"[{algo.upper()}] Episode {ep+1}/{n_episodes} | steps={steps:4d} | reward={total_reward:7.1f} | {landed}")

    env.close()
    print(f"\n平均分：{np.mean(total_rewards):.1f}  (成功率：{np.mean(np.array(total_rewards)>=200)*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="觀看 LunarLander agent 降落")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路徑（不含 .zip），省略則用隨機 agent")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn"], help="模型演算法（預設 ppo）")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--continuous", action="store_true", default=CONTINUOUS,
                        help="使用連續動作空間")
    args = parser.parse_args()

    if args.model:
        watch_trained(args.model, args.algo, args.episodes, args.continuous)
    else:
        print("沒有指定模型，使用隨機 agent（亂按）...")
        watch_random(args.episodes, args.continuous)
