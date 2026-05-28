"""
Pong Reward Wrappers

Pong 原始 reward 機制說明
========================
ALE 對 Pong 的原始獎勵：
  +1  當對手未能接到球（我方得分）
  -1  當己方未能接到球（對手得分）
   0  其餘所有 step（約佔 99%）

本模組提供三種策略導向的 reward 設計：

1. AggressiveWrapper（進攻型）
   核心邏輯：雪球效應，領先越多進攻動力越強
   設計項目：
     - 連勝加成：連續得分 reward 遞增（streak bonus）
     - 優勢獎勵：領先分數越多，得分獎勵越高
     - 無 step penalty（不催促，讓 agent 找到最佳時機）
     - 失分清零連勝計數

2. DefensiveWrapper（防守型）
   核心邏輯：先學「打球」，再學怎麼贏
   設計項目：
     - 生存獎勵（survival_bonus）：每步存活給小 bonus
     - 打球獎勵（hit_bonus）：偵測到玩家擊球時額外給分（RAM 偵測）
     - 失分維持原始 -1.0（不加重，依靠正向引導）
     - 得分不加碼：維持 +1，重點是持續打球

3. BalancedWrapper（平衡型）
   核心邏輯：打球獎勵 ＋ 優勢獎勵的結合
   設計項目：
     - 打球獎勵（hit_bonus）：偵測到玩家擊球時給分（RAM 偵測）
     - 優勢獎勵：領先時得分加成（與進攻型相同機制）
     - 失分略重 -1.5（適度壓力）
"""
import gymnasium as gym


class _BallHitMixin:
    """
    Mixin：從 ALE RAM 偵測玩家擊球事件（Pong 專用）。

    Pong RAM[49] = 球的 x 座標（0–160）。
    玩家控制右側球拍。當球 x 方向從「向右（接近玩家）」
    反轉為「向左（離開玩家）」時，代表玩家剛擊球。

    對方（左側 AI）擊球時方向是「向左→向右」，不會觸發。
    """

    def _init_hit_tracking(self):
        self._prev_ball_x: int | None = None
        self._ball_going_right: bool | None = None

    def _check_player_hit(self) -> bool:
        """偵測玩家是否剛打到球，並更新追蹤狀態。"""
        try:
            curr_x = int(self.unwrapped.ale.getRAM()[49])
        except Exception:
            return False

        hit = False
        if self._prev_ball_x is not None and curr_x != self._prev_ball_x:
            going_right = curr_x > self._prev_ball_x
            # 球方向：右→左 = 玩家（右拍）剛擊球
            if self._ball_going_right is True and not going_right:
                hit = True
            self._ball_going_right = going_right

        self._prev_ball_x = curr_x
        return hit

    def _reset_hit_tracking(self):
        """得分 / 失分後重置追蹤（球位置重置，避免誤判）。"""
        self._prev_ball_x = None
        self._ball_going_right = None


class AggressiveWrapper(gym.Wrapper):
    """
    進攻型策略：不管怎樣，我就是要贏。

    我得分：
      reward = 1.0
             + streak_bonus × streak       （連勝加成）
             + advantage_bonus × lead / 21  （優勢獎勵：領先越多加越多）
      streak += 1
    我失分：
      streak = 0                           （連勝清零）
    每步：
      reward = 0                           （無額外懲罰）

    設計邏輯：
      領先時每次得分都更值錢 → 正向雪球效應。
      不加 step penalty，讓 agent 自由選擇進攻時機。
    """

    def __init__(self, env: gym.Env, streak_bonus: float = 0.3,
                 advantage_bonus: float = 0.5):
        super().__init__(env)
        self.streak_bonus = streak_bonus
        self.advantage_bonus = advantage_bonus
        self._streak = 0
        self._my_score = 0
        self._opp_score = 0

    def reset(self, **kwargs):
        self._streak = 0
        self._my_score = 0
        self._opp_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward > 0.5:          # 我得分
            lead = max(0, self._my_score - self._opp_score)
            reward = (1.0
                      + self.streak_bonus * self._streak
                      + self.advantage_bonus * lead / 21.0)
            self._my_score += 1
            self._streak += 1
        elif reward < -0.5:       # 我失分
            self._opp_score += 1
            self._streak = 0
            # 失分保持 -1.0（原始值），不額外懲罰

        return obs, reward, terminated, truncated, info


class DefensiveWrapper(_BallHitMixin, gym.Wrapper):
    """
    防守型策略：活下來，球有碰到就好。

    每步（沒有得失分）：
      reward = survival_bonus                （生存獎勵：活著就有小分）
               + hit_bonus  （若偵測到擊球）  （打球獎勵：碰到球有額外獎勵）
    我得分：
      reward = +1.0                          （維持原始，不加碼）
    我失分：
      reward = -1.0                          （維持原始，不加重）

    設計邏輯：
      生存獎勵讓 agent 學到「活著有意義」，
      打球獎勵（碰球）讓 agent 學到「移動去接球有意義」，
      兩者都是純正向訊號，不依賴額外懲罰。
    """

    def __init__(self, env: gym.Env, survival_bonus: float = 0.001,
                 hit_bonus: float = 0.5):
        super().__init__(env)
        self.survival_bonus = survival_bonus
        self.hit_bonus = hit_bonus
        self._init_hit_tracking()

    def reset(self, **kwargs):
        self._reset_hit_tracking()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hit = self._check_player_hit()

        if reward > 0.5:          # 我得分
            self._reset_hit_tracking()
            reward = 1.0
        elif reward < -0.5:       # 我失分
            self._reset_hit_tracking()
            reward = -1.0
        else:                     # 一般 step
            reward = self.survival_bonus + (self.hit_bonus if hit else 0.0)

        return obs, reward, terminated, truncated, info


class BalancedWrapper(_BallHitMixin, gym.Wrapper):
    """
    平衡型策略：打球獎勵 ＋ 優勢獎勵的結合。

    每步（沒有得失分）：
      hit = 碰球偵測（RAM）
      reward = hit_bonus  （若偵測到擊球，否則 0）

    我得分時（優勢獎勵）：
      reward = 1.0 + advantage_bonus × max(0, lead) / 21
      → 領先越多得分獎勵越高（與進攻型相同機制）

    我失分時：
      reward = -lose_penalty                （略重於原始，提供適度壓力）

    設計邏輯：
      打球獎勵提供密集的早期學習信號（來自 Defensive），
      優勢獎勵在建立領先後強化進攻動機（來自 Aggressive），
      兩者結合讓 agent 先學會打球，再學會怎麼贏。
    """

    def __init__(self, env: gym.Env, hit_bonus: float = 0.5,
                 advantage_bonus: float = 0.5, lose_penalty: float = 1.2):
        super().__init__(env)
        self.hit_bonus = hit_bonus
        self.advantage_bonus = advantage_bonus
        self.lose_penalty = lose_penalty
        self._my_score = 0
        self._opp_score = 0
        self._init_hit_tracking()

    def reset(self, **kwargs):
        self._my_score = 0
        self._opp_score = 0
        self._reset_hit_tracking()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hit = self._check_player_hit()

        if reward > 0.5:          # 我得分：優勢獎勵
            lead = max(0, self._my_score - self._opp_score)
            reward = 1.0 + self.advantage_bonus * lead / 21.0
            self._my_score += 1
            self._reset_hit_tracking()
        elif reward < -0.5:       # 我失分
            self._opp_score += 1
            reward = -self.lose_penalty
            self._reset_hit_tracking()
        else:                     # 一般 step：打球獎勵
            reward = self.hit_bonus if hit else 0.0

        return obs, reward, terminated, truncated, info
