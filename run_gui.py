#!/usr/bin/env python3
"""AI Trainer Lite — 桌面 GUI 客户端入口"""

import sys
import os

# 确保从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import run_app

if __name__ == "__main__":
    run_app()
