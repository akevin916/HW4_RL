"""
watch_game.py — 直接看 Pong agent 打球
========================================
這支腳本會開一個視窗，讓你「肉眼看到」agent 在打球。
你可以選擇：
  - 看「完全隨機」的 agent（不需要預訓練）
  - 看「已存的訓練模型」打球（需要先跑過 P1/P4/P5 之類的實驗）

用法：
  python pong/watch_game.py                                               # 隨機 agent
  python pong/watch_game.py --model pong/results/P1_frame_stack/model_1_Frame_No_Stacking --nstack 1
  python pong/watch_game.py --model pong/results/P1_frame_stack/model_4_Frames_Standard   --nstack 4
  python pong/watch_game.py --model pong/results/P5_reward_shaping/model_default          --nstack 4

  ⚠  --nstack 必須和訓練時的 n_stack 一致（P1 有 1 和 4 兩種；P4/P5 都是 4）

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


def watch_trained(model_path: str, n_stack: int = 4, n_episodes: int = NUM_EPISODES):
    """看訓練好的模型打球（n_stack 需與訓練時相同）"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    print(f"載入模型：{model_path}  (n_stack={n_stack})")

    # 使用與訓練完全一致的 make_atari_env（內含 VecTransposeImage），
    # 只多傳 render_mode="human" 讓畫面顯示出來
    vec_env = make_atari_env(
        "ALE/Pong-v5",
        n_envs=1,
        env_kwargs={"render_mode": "human"},
    )
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    model = DQN.load(model_path, device="cpu")   # 觀看用 CPU 就夠

    for ep in range(n_episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = vec_env.step(action)
            total_reward += float(reward[0])
            steps += 1
            done = dones[0]
            if FPS > 0:
                time.sleep(1.0 / FPS)
        print(f"[Trained] Episode {ep+1}/{n_episodes} | steps={steps} | reward={total_reward:.1f}")
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="觀看 Pong agent 打球")
    parser.add_argument(
        "--model", type=str, default=None,
        help="訓練模型路徑（不含 .zip），省略則使用隨機 agent"
    )
    parser.add_argument("--nstack", type=int, default=4,
                        help="frame stack 數（P1 的 1-frame 模型用 1；其餘用 4）")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = parser.parse_args()

    if args.model:
        watch_trained(args.model, args.nstack, args.episodes)
    else:
        print("沒有指定模型，使用隨機 agent（亂按）...")
        watch_random(args.episodes)
