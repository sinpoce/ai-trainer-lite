<div align="center">

# AI Trainer Lite

**3 步训练你的专属 AI 模型 · 无需 GPU · 无需机器学习背景**

[![Stars](https://img.shields.io/github/stars/sinpoce/ai-trainer-lite?style=for-the-badge&color=violet)](https://github.com/sinpoce/ai-trainer-lite/stargazers)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-Desktop-green?style=for-the-badge)](https://doc.qt.io/qtforpython-6/)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-orange?style=for-the-badge)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[English](#english) · [快速开始](#-快速开始) · [桌面客户端](#-桌面客户端) · [使用文档](#-使用文档)

</div>

---

**AI Trainer Lite** 是一个开箱即用的 AI 模型训练工具，提供 **桌面 GUI 客户端** 和 **Web 界面** 两种使用方式，让任何人都能训练自己的 AI 模型——不需要写代码，不需要 GPU，不需要懂机器学习。

---

## v2.1 新功能

- **EXE 打包** — 一键打包为 Windows / macOS 桌面应用，无需安装 Python
- **ONNX 导出** — 支持将所有类型模型导出为 ONNX 格式，跨平台部署
- **Gradio 音频分类** — Web UI 新增音频分类和批量预测 Tab
- **PyQt6 桌面客户端** — 原生窗口体验，深色主题，侧边栏导航
- **音频分类** — 基于 Mel 频谱图 + CNN，支持语音命令、环境音识别
- **批量预测** — 加载已训练模型，对新数据批量推理并导出
- **训练历史管理** — 查看、对比、清理所有已训练模型
- **实时训练曲线** — 嵌入式 Matplotlib 图表，训练过程可视化
- **算法对比图** — 表格 AutoML 自动生成算法性能对比柱状图

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 📝 **文本分类** | 基于 DistilBERT / BERT 微调，支持情感分析、主题分类、意图识别等 |
| 📊 **表格数据 AutoML** | 自动尝试 7 种算法，找到最优模型，展示特征重要性 |
| 🖼️ **图像分类** | 迁移学习（MobileNet / ResNet / EfficientNet），支持数据增强 |
| 🎵 **音频分类** | Mel 频谱图 / MFCC + CNN，支持 WAV/MP3/FLAC/OGG |
| 🔮 **批量预测** | 加载模型对新数据批量推理，支持导出 CSV |
| 📋 **训练历史** | 管理已训练模型，查看详情，一键清理 |
| 📈 **可视化** | 训练曲线、算法对比、特征重要性、混淆矩阵 |
| 💾 **代码导出** | 自动生成推理代码，一键复制使用 |
| ⚡ **CPU 友好** | 所有模型均可在 CPU 上运行，无 GPU 也能用 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/sinpoce/ai-trainer-lite
cd ai-trainer-lite
pip install -r requirements.txt
```

### 桌面客户端（推荐）

```bash
python run_gui.py
```

### Web 界面

```bash
python app.py
# 浏览器访问 http://localhost:7860
```

---

## 桌面客户端

v2.0 新增原生桌面 GUI 客户端，基于 PyQt6 构建：

- **仪表盘** — 系统概览、已训练模型统计、功能入口
- **文本分类** — 上传 CSV，选择模型，一键训练
- **表格 AutoML** — 自动算法对比，实时柱状图
- **图像分类** — 文件夹浏览器，训练曲线实时绘制
- **音频分类** — 支持多种音频格式，Mel/MFCC 特征
- **批量预测** — 单条和批量推理，结果导出 CSV
- **训练历史** — 模型管理，查看/删除/对比
- **设置** — 默认参数、模型目录配置

---

## 使用文档

### 📝 文本分类

适合场景：情感分析、评论分类、邮件分类、新闻分类、意图识别...

**数据格式（CSV）：**

```csv
text,label
"这个产品非常好用！",positive
"质量太差了，完全不推荐",negative
"还可以，一般般",neutral
```

**支持模型：**

| 模型 | 语言 | 参数量 | 推荐场景 |
|------|------|--------|----------|
| distilbert-base-uncased | 英文 | 66M | 速度优先 |
| bert-base-uncased | 英文 | 110M | 精度优先 |
| bert-base-chinese | 中文 | 110M | 中文文本 |
| distilbert-base-multilingual-cased | 多语言 | 135M | 多语言 |
| roberta-base | 英文 | 125M | 最高精度 |

### 📊 表格数据 AutoML

适合场景：房价预测、客户流失、疾病预测、销量预测...

自动尝试以下算法：

| 算法 | 分类 | 回归 | 特点 |
|------|------|------|------|
| Logistic/Linear Regression | ✓ | ✓ | 基准线 |
| Random Forest | ✓ | ✓ | 通常最优 |
| Gradient Boosting | ✓ | ✓ | 高精度 |
| XGBoost | ✓ | ✓ | 高效 |
| SVM/SVR | ✓ | ✓ | 小数据集 |
| KNN | ✓ | - | 简单 |
| Naive Bayes | ✓ | - | 文本 |

### 🖼️ 图像分类

文件夹结构：
```
my_dataset/
├── 猫/
│   ├── cat001.jpg
│   └── ...
├── 狗/
│   └── ...
└── 鸟/
    └── ...
```

| 架构 | 参数量 | CPU 速度 | 精度 |
|------|--------|---------|------|
| MobileNet V3 Small | 1.5M | 快 | 良好 |
| EfficientNet B0 | 5.3M | 中等 | 优秀 |
| ResNet18 | 11M | 中等 | 良好 |
| VGG11 | 128M | 慢 | 优秀 |

### 🎵 音频分类

文件夹结构：
```
audio_dataset/
├── 语音命令_开灯/
│   ├── sample001.wav
│   └── ...
├── 语音命令_关灯/
│   └── ...
└── 背景噪音/
    └── ...
```

支持格式：WAV, MP3, FLAC, OGG, M4A

特征类型：
- **Mel Spectrogram** — 适合通用音频分类
- **MFCC** — 适合语音识别

---

## Python API

### 文本分类

```python
import pandas as pd
from trainers import TextClassifierTrainer

df = pd.read_csv("my_data.csv")
trainer = TextClassifierTrainer(model_name="bert-base-chinese", epochs=3)
result = trainer.train(df, text_col="text", label_col="label")
print(f"准确率: {result['accuracy']:.2%}")
```

### 表格数据

```python
from trainers import TabularTrainer

df = pd.read_csv("my_data.csv")
trainer = TabularTrainer(task_type="classification")
result = trainer.train(df, target_col="target")
print(f"最佳: {result['best_algorithm']} ({result['best_score']:.4f})")
```

### 图像分类

```python
from trainers import ImageClassifierTrainer

trainer = ImageClassifierTrainer(epochs=10)
result = trainer.train(folder="./my_dataset")
print(f"准确率: {result['best_acc']:.2%}")
```

### 音频分类

```python
from trainers.audio_trainer import AudioClassifierTrainer

trainer = AudioClassifierTrainer(feature_type="mel_spectrogram", epochs=20)
result = trainer.train(folder="./audio_dataset")
print(f"准确率: {result['best_acc']:.2%}")
```

---

## 项目结构

```
ai-trainer-lite/
├── run_gui.py              # 桌面 GUI 入口
├── app.py                  # Web UI 入口 (Gradio)
├── build.py                # 一键打包脚本
├── ai_trainer.spec         # PyInstaller 配置
├── gui/                    # PyQt6 桌面客户端
│   ├── main_window.py      # 主窗口
│   ├── styles.py           # 深色主题样式表
│   ├── pages/              # 各功能页面
│   │   ├── dashboard.py    # 仪表盘
│   │   ├── text_page.py    # 文本分类
│   │   ├── tabular_page.py # 表格 AutoML
│   │   ├── image_page.py   # 图像分类
│   │   ├── audio_page.py   # 音频分类
│   │   ├── predict_page.py # 批量预测
│   │   ├── history_page.py # 训练历史
│   │   └── settings_page.py# 设置
│   └── widgets/            # 可复用组件
│       ├── sidebar.py      # 侧边栏导航
│       └── chart_widget.py # 图表组件
├── trainers/               # 训练器核心
│   ├── text_trainer.py     # 文本分类 (BERT)
│   ├── tabular_trainer.py  # 表格 AutoML
│   ├── image_trainer.py    # 图像分类 (CNN)
│   └── audio_trainer.py    # 音频分类 (Mel+CNN)
├── utils/                  # 工具模块
│   └── export.py           # ONNX 模型导出
├── scripts/                # 打包脚本
│   ├── build_exe.bat       # Windows 打包
│   └── build_mac.sh        # macOS 打包
├── examples/               # 示例数据集
│   ├── sentiment.csv       # 情感分析示例
│   └── iris.csv            # 鸢尾花分类示例
├── requirements.txt        # 运行依赖
└── requirements-build.txt  # 构建依赖
```

---

## 系统要求

- Python 3.9+
- RAM：4GB+（文本模型需要 4-8GB）
- 存储空间：5GB+（模型权重）
- GPU：可选（有 GPU 训练更快，无 GPU 也能运行）

---

## 打包为 EXE

### Windows

```bash
# 方式一：运行打包脚本
scripts\build_exe.bat

# 方式二：命令行
pip install pyinstaller
python build.py
```

### macOS

```bash
bash scripts/build_mac.sh
```

打包后的应用位于 `dist/AI-Trainer-Lite/` 目录。

---

## ONNX 导出

将训练好的模型导出为 ONNX 格式，支持在 C++ / Java / JavaScript / C# 等环境中运行推理：

```bash
# 安装 ONNX 依赖
pip install onnx onnxruntime skl2onnx
```

```python
from utils.export import auto_export

# 自动检测模型类型并导出
onnx_path = auto_export("./models/my-model")
print(f"ONNX 模型已保存至: {onnx_path}")
```

支持导出：
- 文本分类模型（BERT/DistilBERT → ONNX）
- 表格数据模型（scikit-learn → ONNX，需要 skl2onnx）
- 图像分类模型（PyTorch → ONNX）
- 音频分类模型（PyTorch → ONNX）

---

## 贡献

欢迎 PR！可以添加：
- 新的模型架构支持
- 更多示例数据集
- 模型量化/压缩
- 更多训练任务（目标检测、NER...）

---

## 许可证

MIT License

---

<div align="center">

如果这个项目对你有帮助，请点 **Star** 支持！

**[GitHub Issues](https://github.com/sinpoce/ai-trainer-lite/issues)** · **[Discussions](https://github.com/sinpoce/ai-trainer-lite/discussions)**

</div>
