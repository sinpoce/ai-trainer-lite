#!/bin/bash
# AI Trainer Lite - macOS 打包脚本
# 用法: bash scripts/build_mac.sh

set -e

echo "============================================================"
echo "  AI Trainer Lite - macOS 应用打包"
echo "============================================================"
echo ""

cd "$(dirname "$0")/.."

# 检查依赖
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "[安装] 正在安装 PyInstaller..."
    pip3 install pyinstaller
fi

echo "[构建] 开始打包..."
python3 build.py

echo ""
echo "✅ 完成！应用位于 dist/AI-Trainer-Lite/"
