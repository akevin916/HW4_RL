"""
watch_game.py — 直接看 Pong agent 打球
========================================
這支腳本會開一個視窗，讓你「肉眼看到」agent 在打球。
你可以選擇：
  - 看「完全隨機」的 agent（不需要預訓練）
  - 看「已存的訓練模型」打球（需要先跑過 P1/P5 之類的實驗）

用法：
  python pong/watch_game.py              # 看隨機 agent
  python pong/watch_game.py --model pong/results/P5_reward_shaping/model_default  # 看訓練模型

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
可修改的地方：
  NUM_EPISODES     — 看幾局
  FPS              — 播放速度（越大越快，0 = 盡量快）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys, time, argparse

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可修改的地方
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUM_EPISODES = 3      # 看幾局
FPS          = 30     # 幀率；調低可以看慢動作，調高可以快速略看


def watch_random(n_episodes: int = NUM_EPISODES):
    """看完全隨機 agent 打球（無需預訓練）"""
    env = gym.make("ALE/Pong-v5", render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()   # 亂按
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if FPS > 0:
                time.sleep(1.0 / FPS)
        print(f"[Random] Episode {ep+1}/{n_episodes} | steps={steps} | reward={total_reward:.1f}")
    env.close()


def watch_trained(model_path: str, n_episodes: int = NUM_EPISODES):
    """看訓練好的模型打球"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.atari_wrappers import AtariWrapper

    print(f"載入模型：{model_path}")
    # 評估用環境（render_mode=human）
    base_env = gym.make("ALE/Pong-v5", render_mode="human")
    env = AtariWrapper(base_env, clip_reward=True)

    model = DQN.load(model_path, device="cpu")   # 觀看用 CPU 就夠

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            # SB3 需要 (1, H, W, C) 格式，但 AtariWrapper 輸出 (H, W) 灰階
            # 直接用 predict 需要對應格式，這裡轉成 numpy array
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if FPS > 0:
                time.sleep(1.0 / FPS)
        print(f"[Trained] Episode {ep+1}/{n_episodes} | steps={steps} | reward={total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="觀看 Pong agent 打球")
    parser.add_argument(
        "--model", type=str, default=None,
        help="訓練模型路徑（不含 .zip），省略則使用隨機 agent"
    )
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = parser.parse_args()

    if args.model:
        watch_trained(args.model, args.episodes)
    else:
        print("沒有指定模型，使用隨機 agent（亂按）...")
        watch_random(args.episodes)
