@echo off
REM AI Trainer Lite - Windows EXE 打包脚本
REM 用法: 双击运行 或 在命令行中执行 scripts\build_exe.bat

echo ============================================================
echo   AI Trainer Lite - Windows EXE 打包
echo ============================================================
echo.

REM 切换到项目根目录
cd /d "%~dp0\.."

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 检查 PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 PyInstaller...
    pip install pyinstaller
)

REM 运行打包
echo [构建] 开始打包...
echo.
python build.py

echo.
echo 按任意键退出...
pause
