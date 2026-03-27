# -*- mode: python ; coding: utf-8 -*-
"""
AI Trainer Lite - PyInstaller 打包配置
生成单目录 EXE 应用
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── 隐式导入 ──────────────────────────────────────────────────────────────────
hidden_imports = [
    # scikit-learn
    'sklearn.tree._classes',
    'sklearn.ensemble._gb',
    'sklearn.ensemble._forest',
    'sklearn.ensemble._gb_losses',
    'sklearn.linear_model._logistic',
    'sklearn.svm._classes',
    'sklearn.neighbors._classification',
    'sklearn.naive_bayes',
    'sklearn.preprocessing._label',
    'sklearn.preprocessing._encoders',
    'sklearn.preprocessing._data',
    'sklearn.impute._base',
    'sklearn.compose._column_transformer',
    'sklearn.pipeline',
    'sklearn.model_selection._split',
    'sklearn.metrics._classification',
    'sklearn.metrics._regression',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    # xgboost
    'xgboost',
    'xgboost.sklearn',
    'xgboost.core',
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    # matplotlib
    'matplotlib',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.figure',
    # numpy / pandas
    'numpy',
    'pandas',
    'pandas._libs.tslibs.np_datetime',
    # 项目内部模块
    'trainers',
    'trainers.text_trainer',
    'trainers.tabular_trainer',
    'trainers.image_trainer',
    'trainers.audio_trainer',
    'gui',
    'gui.main_window',
    'gui.styles',
    'gui.widgets.sidebar',
    'gui.widgets.chart_widget',
    'gui.pages.dashboard',
    'gui.pages.text_page',
    'gui.pages.tabular_page',
    'gui.pages.image_page',
    'gui.pages.audio_page',
    'gui.pages.predict_page',
    'gui.pages.history_page',
    'gui.pages.settings_page',
    'utils',
    'utils.export',
]

# ── 数据文件 ──────────────────────────────────────────────────────────────────
datas = [
    ('examples', 'examples'),
]

# ── 分析 ──────────────────────────────────────────────────────────────────────
a = Analysis(
    ['run_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'gradio',           # Web UI 不打包进桌面版
        'uvicorn',
        'fastapi',
        'httpx',
        'websockets',
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── 打包 ──────────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI-Trainer-Lite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,      # 无控制台窗口
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI-Trainer-Lite',
)
