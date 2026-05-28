# Reinforcement Learning Assignment — 2026

## 專案概述

本專案針對 Reinforcement Learning 作業，使用 [Gymnasium](https://gymnasium.farama.org/) 框架，
在兩個環境上訓練 RL 代理人並進行系統性實驗：

| 任務 | 環境 ID | 演算法 | 類型 |
|------|---------|--------|------|
| Atari Pong | `ALE/Pong-v5` | DQN | Atari |
| Lunar Lander | `LunarLander-v3` / `LunarLanderContinuous-v2` | DQN、PPO | Classic Control |

---

## 研究問題

> **訓練配置（演算法選擇、超參數設定、輸入前處理方式）如何影響 RL 代理人在不同類型環境下的學習效率與最終表現？**

---

## 環境設置

### 1. 建立 Conda 虛擬環境

```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env
```

### 2. 安裝套件

```bash
pip install "gymnasium[atari,box2d,other]" "stable-baselines3>=2.0" \
    ale-py AutoROM matplotlib pandas seaborn tqdm \
    ipykernel jupyter tensorboard opencv-python torch torchvision
```

### 3. 安裝 Atari ROMs

```bash
AutoROM --accept-license
```

### 4. 驗證安裝

```bash
python -c "
import gymnasium as gym, ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5')
print('Pong OK:', env.reset()[0].shape)
env.close()
env = gym.make('LunarLander-v3')
print('LunarLander OK:', env.reset()[0].shape)
env.close()
"
```

---

## 專案結構

```
RL/
├── README.md                  # 本文件
├── requirements.txt           # 套件版本清單
│
├── pong/                      # Task 1: Atari Pong
│   ├── train_dqn.py           # DQN 訓練主程式
│   ├── exp_P1_frame_stack.py  # 實驗 P-1: Frame Stacking 比較
│   ├── exp_P4_epsilon.py      # 實驗 P-4: ε-greedy 衰減速度比較
│   └── results/               # 訓練曲線、模型權重輸出
│
├── lunarlander/               # Task 2: LunarLander
│   ├── train_dqn.py           # DQN 訓練主程式
│   ├── train_ppo.py           # PPO 訓練主程式
│   ├── exp_L1_algo_compare.py # 實驗 L-1: DQN vs PPO
│   ├── exp_L2_action_space.py # 實驗 L-2: 離散 vs 連續動作
│   ├── exp_L3_lr_sensitivity.py   # 實驗 L-3: 學習率敏感度
│   ├── exp_L4_network_size.py     # 實驗 L-4: 網路結構大小
│   └── results/               # 訓練曲線、模型權重輸出
│
└── utils/
    ├── plot.py                # 統一繪圖工具（訓練曲線、比較圖）
    └── evaluate.py            # 評估工具（計算平均分數、成功率）
```

---

## 實驗設計摘要

### Task 1 — Pong (DQN)

| 實驗 | 操控變因 | 對照 vs 實驗 |
|------|---------|------------|
| P-1 | Frame Stacking | 1 frame vs 4 frames |
| P-4 | ε 衰減速度 | 快速 (50k steps) vs 慢速 (200k steps) |

- 訓練步數：500k steps / 組
- 評估指標：每 10 episode 平均分數、收斂 episode 數

### Task 2 — LunarLander (DQN / PPO)

| 實驗 | 操控變因 | 說明 |
|------|---------|------|
| L-1 | 演算法 | DQN vs PPO（離散版） |
| L-2 | 動作空間 | PPO 離散 vs PPO 連續 |
| L-3 | 學習率 | lr ∈ {1e-4, 3e-4, 1e-3} |
| L-4 | 網路大小 | [64,64] vs [256,256] |

- 訓練步數：200k steps / 組
- 評估指標：平均累積獎勵、成功著陸率（reward ≥ 200）

---

## 執行方式

```bash
conda activate rl_env

# Pong 實驗
python pong/exp_P1_frame_stack.py
python pong/exp_P4_epsilon.py

# LunarLander 實驗
python lunarlander/exp_L1_algo_compare.py
python lunarlander/exp_L2_action_space.py
python lunarlander/exp_L3_lr_sensitivity.py
python lunarlander/exp_L4_network_size.py
```

結果（圖表與模型）會儲存至各任務的 `results/` 資料夾。

---

## 參考資料

1. Farama Foundation, *Gymnasium Documentation*, https://gymnasium.farama.org/
2. Stable-Baselines3, https://stable-baselines3.readthedocs.io/
3. Mnih et al., "Playing Atari with Deep Reinforcement Learning", DeepMind, 2013.
4. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.
5. Brockman et al., "OpenAI Gym", arXiv:1606.01540, 2016.

---

## 硬體環境

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA RTX 4090 (24GB) |
| CUDA | 12.4 |
| Python | 3.10 |
| PyTorch | 2.3.1 |
| Gymnasium | 1.2.3 |
| Stable-Baselines3 | 2.8.0 |
