#!/usr/bin/env bash
# =============================================================================
# run_exp5.sh
# 只執行第 5 組實驗：P5 Reward Shaping + L5 Reward Shaping
# 使用方式：bash run_exp5.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl_env

echo "======================================================"
echo "  Exp-5 Only — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python : $(python --version)"
echo "  Device : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "======================================================"
echo ""

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

run_exp "P5_Reward_Shaping"  "pong/exp_P5_reward_shaping.py"
run_exp "L5_Reward_Shaping"  "lunarlander/exp_L5_reward_shaping.py"

echo "======================================================"
echo "  完成！  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo "結果位置："
echo "  pong/results/P5_reward_shaping/"
echo "  lunarlander/results/L5_reward_shaping/"
echo "  logs/P5_Reward_Shaping.log"
echo "  logs/L5_Reward_Shaping.log"
