#!/usr/bin/env python3
"""
AI Trainer Lite — 一键打包脚本
用法: python build.py

生成结果在 dist/AI-Trainer-Lite/ 目录下
"""

import subprocess
import sys
import os
import shutil


def check_pyinstaller():
    try:
        import PyInstaller
        print(f"[OK] PyInstaller {PyInstaller.__version__}")
        return True
    except ImportError:
        print("[!] 未安装 PyInstaller，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True


def clean():
    for d in ["build", "dist"]:
        if os.path.exists(d):
            print(f"[清理] 删除 {d}/")
            shutil.rmtree(d)


def build():
    print("\n" + "=" * 60)
    print("  AI Trainer Lite — 开始打包")
    print("=" * 60 + "\n")

    # 检查依赖
    check_pyinstaller()

    # 清理旧构建
    clean()

    # 运行 PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "ai_trainer.spec",
        "--noconfirm",
    ]

    print(f"[构建] 运行: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        dist_dir = os.path.join("dist", "AI-Trainer-Lite")
        if os.path.exists(dist_dir):
            # 计算大小
            total_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(dist_dir) for f in fns
            )
            size_mb = total_size / (1024 * 1024)

            print("\n" + "=" * 60)
            print("  ✅ 打包成功！")
            print("=" * 60)
            print(f"\n  输出目录：{os.path.abspath(dist_dir)}")
            print(f"  总大小：{size_mb:.0f} MB")
            print(f"\n  运行方式：")
            if sys.platform == "win32":
                print(f"    dist/AI-Trainer-Lite/AI-Trainer-Lite.exe")
            else:
                print(f"    dist/AI-Trainer-Lite/AI-Trainer-Lite")
            print()
        else:
            print("\n[!] 构建完成但未找到输出目录")
    else:
        print(f"\n[❌] 构建失败（退出码 {result.returncode}）")
        sys.exit(1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build()
