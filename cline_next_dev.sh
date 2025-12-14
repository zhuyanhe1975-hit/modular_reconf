#!/bin/bash
# 文件名: cline_next_dev.sh
# 功能: 下一步开发批处理指令，统一接口、增强 ExecutionTrace、增加测试和可视化、版本管理

# 进入项目工作目录
cd /home/yhzhu/myWorks/UBot/modular_reconf || exit 1

echo "=== Step 1: 确保统一入口接口 ==="
# 检查 main.py 调用 modular_reconfig.py + verifier.py + visualization
# 由 Cline 处理检查逻辑，输出日志

echo "=== Step 2: 增强 ExecutionTrace 结构 ==="
# 在 reconfiguration/modular_reconfig.py 中：
# - ExecutionStep dataclass 添加字段: timestamp, collision_type
# - ExecutionTrace.summarize() 返回总步数、冲突步数、事件统计
# 由 Cline 修改代码并验证输出一致

echo "=== Step 3: 添加自动测试用例 ==="
# 在 tests/ 下创建 test_execution_trace.py
# - 验证 ExecutionTrace 结构完整性
# - 验证 verifier.py 输出正确
# - 使用 2-3 模块的快速场景
touch tests/test_execution_trace.py
echo "# 自动生成测试文件 - 由 Cline 填充测试内容" > tests/test_execution_trace.py

echo "=== Step 4: 升级可视化模块 ==="
# 确保 visualization 下脚本独立运行，支持 ExecutionTrace 输入
# 可选择播放步数或全部，并输出模块状态与轨迹
# 由 Cline 修改 visualization/ 下所有脚本

echo "=== Step 5: Git 版本管理 ==="
git add .
git commit -m "Enhance ExecutionTrace, unify interface, add tests and visualization options"
git tag execution-layer-v1.2-2025-12-14
git push origin main --tags

echo "=== Step 6: 快速检查目录结构 ==="
echo "顶层目录和子目录："
ls -d */

echo -e "\nreconfiguration 目录文件："
ls -l reconfiguration/

echo -e "\nvisualization 目录文件："
ls -l visualization/

echo -e "\ntests 目录文件："
ls -l tests/

echo -e "\n顶层多余的 Python 文件："
ls -l *.py

echo -e "\n=== 执行完毕 ==="
