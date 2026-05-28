"""
自訂 Reward Shaping Wrappers for LunarLander-v3

LunarLander-v3 原始 reward 組成：
  - 每步根據與降落墊距離給予正/負 reward
  - 每條腿接觸地面：+10
  - 主引擎點火：-0.3/step
  - 側向引擎點火：-0.03/step
  - 成功降落（靜止）：+100
  - 墜毀：-100

觀測向量 obs（共 8 維）：
  [0] x      — 水平位置（0 = 中心）
  [1] y      — 垂直位置（1.4 = 起始高度）
  [2] vx     — 水平速度
  [3] vy     — 垂直速度（負 = 下降）
  [4] angle  — 傾斜角（0 = 直立）
  [5] omega  — 角速度
  [6] leg_l  — 左腳是否接觸地面（0/1）
  [7] leg_r  — 右腳是否接觸地面（0/1）
"""
import gymnasium as gym
import numpy as np


class VerticalLandingWrapper(gym.Wrapper):
    """
    垂直降落獎勵：鼓勵 agent 保持機體直立，同時減輕側引擎的點火懲罰。

    設計動機：
      原始環境對側引擎點火施加 -0.03/step 的懲罰，
      有時會讓 agent 不敢用側引擎調整角度，導致斜著降落。
      本 wrapper 做兩件事：
        1. 角度獎勵：角度越接近垂直（0）給越多正獎勵
           → reward += angle_reward_coef × (1 - |angle| / π)
           → 完全垂直時 +angle_reward_coef，完全翻轉時 +0
        2. 側引擎懲罰減免：把原始 -0.03 的懲罰用補償抵消部分
           → 讓 agent 更願意用側引擎修正姿態

    Args:
        angle_reward_coef:   角度獎勵的最大值（建議 0.05～0.2）
        engine_relief_coef:  每步對側引擎懲罰的補償量（建議 0.01～0.03）
    """

    def __init__(
        self,
        env: gym.Env,
        angle_reward_coef: float = 0.1,
        engine_relief_coef: float = 0.02,
    ):
        super().__init__(env)
        self.angle_reward_coef = angle_reward_coef
        self.engine_relief_coef = engine_relief_coef

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        angle = float(obs[4])  # obs[4] = 傾斜角

        # 1. 角度獎勵：垂直時最大，翻轉時為 0
        angle_reward = self.angle_reward_coef * (1.0 - abs(angle) / 3.14159)

        # 2. 側引擎懲罰補償：每步固定補回一點，降低不敢噴的顧慮
        #    （原始懲罰 -0.03 由底層環境扣，這裡補回 engine_relief_coef）
        engine_relief = self.engine_relief_coef

        shaped_reward = reward + angle_reward + engine_relief
        return obs, shaped_reward, terminated, truncated, info


class PrecisionLandingWrapper(gym.Wrapper):
    """
    自訂 Reward Shaping：置中獎勵 + 低空減速懲罰。

    附加項目：
      1. 置中獎勵（center_reward）：
         x 座標越靠近 0（降落台中心）給越多獎勵。
         reward += center_coef × (1 - |x| / x_max)
         → 正中心 +center_coef，偏到邊緣趨近 +0

      2. 低空速度懲罰（velocity_penalty）：
         當高度 y < height_threshold 時，
         懲罰過快的水平與垂直速度，鼓勵輕柔著陸。
         penalty = -velocity_coef × (|vx| + |vy|)  （僅在低空啟用）

    設計邏輯：
      原始 reward 已包含靠近降落台的資訊，但訊號較模糊。
      置中獎勵直接強化「x → 0」的導引，讓 agent 更快學會
      先把水平位置對準，再執行垂直降落。
      低空速度懲罰確保最後接觸地面時速度夠慢，避免墜毀。
    """

    def __init__(
        self,
        env: gym.Env,
        center_coef: float = 0.2,
        x_max: float = 1.0,
        velocity_coef: float = 0.1,
        height_threshold: float = 0.5,
    ):
        """
        Args:
            center_coef:       置中獎勵最大值（建議 0.1～0.3）
            x_max:             x 偏移的正規化基準（超過此值獎勵為 0）
            velocity_coef:     低空速度懲罰係數（建議 0.05～0.15）
            height_threshold:  啟用速度懲罰的高度門檻（y < threshold）
        """
        super().__init__(env)
        self.center_coef = center_coef
        self.x_max = x_max
        self.velocity_coef = velocity_coef
        self.height_threshold = height_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x, y, vx, vy, angle, omega, leg_l, leg_r = obs

        # 1. 置中獎勵：x 越靠近 0 越高，偏出 x_max 後夾在 0
        center_reward = self.center_coef * max(0.0, 1.0 - abs(float(x)) / self.x_max)

        # 2. 低空速度懲罰：只在快碰地時啟用，避免干擾高空機動
        if float(y) < self.height_threshold:
            velocity_penalty = -self.velocity_coef * (abs(float(vx)) + abs(float(vy)))
        else:
            velocity_penalty = 0.0

        shaped_reward = reward + center_reward + velocity_penalty
        return obs, shaped_reward, terminated, truncated, info
