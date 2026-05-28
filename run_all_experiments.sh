#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# 依序執行所有 RL 實驗（Pong × 2 + LunarLander × 4）
# 使用方式：bash run_all_experiments.sh
# =============================================================================

set -e  # 任何指令失敗即中止

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── conda 環境啟動 ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl_env

echo "======================================================"
echo "  RL Experiments — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python : $(python --version)"
echo "  Device : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "======================================================"
echo ""

# ── 實驗執行函式 ──────────────────────────────────────────────────────────────
run_exp() {
    local name="$1"
    local script="$2"
    local logfile="$LOG_DIR/${name}.log"

    echo "------------------------------------------------------"
    echo "  ▶  $name"
    echo "     Script : $script"
    echo "     Log    : $logfile"
    echo "     Start  : $(date '+%H:%M:%S')"
    echo "------------------------------------------------------"

    python "$SCRIPT_DIR/$script" 2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "  ✔  $name 完成  ($(date '+%H:%M:%S'))"
    else
        echo "  ✘  $name 失敗（exit code $exit_code），請查看 $logfile"
        exit $exit_code
    fi
    echo ""
}

# ── Pong 實驗 ─────────────────────────────────────────────────────────────────
run_exp "P1_Frame_Stacking"   "pong/exp_P1_frame_stack.py"
run_exp "P4_Epsilon_Decay"    "pong/exp_P4_epsilon.py"

# ── LunarLander 實驗 ──────────────────────────────────────────────────────────
run_exp "L1_Algo_Compare"     "lunarlander/exp_L1_algo_compare.py"
run_exp "L2_Action_Space"     "lunarlander/exp_L2_action_space.py"
run_exp "L3_LR_Sensitivity"   "lunarlander/exp_L3_lr_sensitivity.py"
run_exp "L4_Network_Size"     "lunarlander/exp_L4_network_size.py"

# ── 完成摘要 ──────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  全部實驗完成！  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo ""
echo "結果位置："
echo "  pong/results/       — P1, P4 圖表與 CSV"
echo "  lunarlander/results/ — L1–L4 圖表與 CSV"
echo "  logs/               — 各實驗完整終端輸出"
