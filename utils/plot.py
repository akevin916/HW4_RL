"""
共用繪圖工具：訓練曲線、多組比較圖。

所有圖片使用 Agg backend（無需顯示器），儲存為 PNG。
"""
import os

import matplotlib
matplotlib.use("Agg")  # 無螢幕環境必須設定
import matplotlib.pyplot as plt
import numpy as np


# ── 顏色方案 ──────────────────────────────────────────────────────────────────
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def smooth(data: list | np.ndarray, window: int = 20) -> np.ndarray:
    """
    移動平均平滑化。

    Args:
        data:   原始數列
        window: 滑動窗口大小

    Returns:
        平滑後的數列（長度為 max(1, len(data) - window + 1)）
    """
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_single(
    rewards: list,
    title: str,
    save_path: str,
    window: int = 20,
    color: str = COLORS[0],
    ylabel: str = "Episode Reward",
) -> None:
    """
    繪製單一訓練曲線（含平滑）並存檔。

    Args:
        rewards:   每個 episode 的獎勵列表
        title:     圖標題
        save_path: 儲存路徑（含檔名，.png）
        window:    平滑窗口大小
        color:     曲線顏色
        ylabel:    Y 軸標籤
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.25, color=color, linewidth=0.8, label="Raw")

    smoothed = smooth(rewards, window)
    offset = len(rewards) - len(smoothed)
    ax.plot(
        np.arange(offset, len(rewards)),
        smoothed,
        color=color,
        linewidth=2,
        label=f"Moving avg (window={window})",
    )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot saved] {save_path}")


def plot_comparison(
    results: dict,
    title: str,
    save_path: str,
    window: int = 20,
    ylabel: str = "Episode Reward",
) -> None:
    """
    在同一張圖中比較多組訓練曲線。

    Args:
        results:   {label: list_of_rewards}
        title:     圖標題
        save_path: 儲存路徑（含檔名，.png）
        window:    平滑窗口大小
        ylabel:    Y 軸標籤
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, rewards) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(
            np.arange(len(rewards)), rewards,
            alpha=0.2, color=color, linewidth=0.6,
        )
        smoothed = smooth(rewards, window)
        offset = len(rewards) - len(smoothed)
        ax.plot(
            np.arange(offset, len(rewards)),
            smoothed,
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot saved] {save_path}")
