"""
共用 Callback：記錄每個 episode 的獎勵與步數。

搭配 SB3 Monitor 包裝的環境使用，
Monitor 在 episode 結束時會在 info["episode"] 中放入 {"r": reward, "l": length}。
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardCallback(BaseCallback):
    """
    在訓練過程中蒐集每個 episode 的獎勵與長度。
    適用於 VecEnv（多環境）或單環境，需搭配 Monitor wrapper。
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.episode_lengths: list = []

    def _on_step(self) -> bool:
        # SB3 在每個 step 後將 infos 放入 locals
        # Monitor wrapper 會在 episode 結束時注入 "episode" 鍵
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True

    @property
    def last_100_mean(self) -> float:
        """最近 100 個 episode 的平均獎勵。"""
        if not self.episode_rewards:
            return float("nan")
        return float(np.mean(self.episode_rewards[-100:]))

    @property
    def last_100_std(self) -> float:
        """最近 100 個 episode 的標準差。"""
        if not self.episode_rewards:
            return float("nan")
        return float(np.std(self.episode_rewards[-100:]))
