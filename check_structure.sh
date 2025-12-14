#!/bin/bash
# check_structure.sh
# 快速检查 modular_reconf 项目目录结构

WORKDIR="/home/yhzhu/myWorks/UBot/modular_reconf"
cd "$WORKDIR" || { echo "工作目录不存在: $WORKDIR"; exit 1; }

echo "===== 顶层目录和子目录 ====="
ls -d */

echo -e "\n===== reconfiguration 目录文件 ====="
RECONF_FILES=("core_simulation.py" "modular_reconfig.py" "multi_module_state.py" "verifier.py")
for f in "${RECONF_FILES[@]}"; do
    if [[ -f "reconfiguration/$f" ]]; then
        echo "[OK] $f"
    else
        echo "[MISSING] $f"
    fi
done

echo -e "\n===== visualization 目录文件 ====="
VIS_FILES=("main_visualize.py" "main_visualize_improved.py" "main_visualize_improved_scaled.py" "main_mujoco_visualize.py" "script.py")
for f in "${VIS_FILES[@]}"; do
    if [[ -f "visualization/$f" ]]; then
        echo "[OK] $f"
    else
        echo "[MISSING] $f"
    fi
done

echo -e "\n===== 顶层多余 Python 文件 ====="
TOP_PY_FILES=("env_check.py" "main.py")
for f in "${TOP_PY_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "[EXISTS] $f"
    else
        echo "[OK] $f not present"
    fi
done

echo -e "\n===== 完整目录树（2级） ====="
if command -v tree >/dev/null 2>&1; then
    tree -L 2
else
    echo "tree 命令未安装，可用: sudo apt install tree"
fi

echo -e "\n===== 检查完成 ====="
