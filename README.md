<div align="center">

# 🤖 AI Trainer Lite

**3 步训练你的专属 AI 模型 · 无需 GPU · 无需机器学习背景**

[![Stars](https://img.shields.io/github/stars/sinpoce/ai-trainer-lite?style=for-the-badge&color=violet)](https://github.com/sinpoce/ai-trainer-lite/stargazers)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[English](#english) · [快速开始](#-快速开始) · [功能演示](#-功能演示) · [使用文档](#-使用文档)

<img src="https://github.com/sinpoce/ai-trainer-lite/assets/preview.png" alt="AI Trainer Lite Screenshot" width="800"/>

</div>

---

**AI Trainer Lite** 是一个开箱即用的 AI 模型训练工具，提供可视化 Web 界面，让任何人都能训练自己的 AI 模型——不需要写代码，不需要 GPU，不需要懂机器学习。

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 📝 **文本分类** | 基于 DistilBERT / BERT 微调，支持情感分析、主题分类、意图识别等 |
| 📊 **表格数据 AutoML** | 自动尝试 7 种算法，找到最优模型，展示特征重要性 |
| 🖼️ **图像分类** | 迁移学习（MobileNet / ResNet / EfficientNet），支持自定义类别 |
| 🔮 **推理测试** | 训练完直接测试，无需切换工具 |
| 💾 **代码导出** | 自动生成推理代码，一键复制使用 |
| ⚡ **CPU 友好** | 所有模型均可在 CPU 上运行，无 GPU 也能用 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/sinpoce/ai-trainer-lite
cd ai-trainer-lite
pip install -r requirements.txt
```

### 启动

```bash
python app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

---

## 📖 使用文档

### 📝 文本分类

适合场景：情感分析、评论分类、邮件分类、新闻分类、意图识别...

**数据格式（CSV）：**

```csv
text,label
"这个产品非常好用！",positive
"质量太差了，完全不推荐",negative
"还可以，一般般",neutral
```

**步骤：**
1. 上传 CSV 文件
2. 填写文本列名和标签列名
3. 选择基础模型（中文推荐 `bert-base-chinese`）
4. 点击「开始训练」

**支持模型：**
- `distilbert-base-uncased` - 英文，速度最快（推荐）
- `bert-base-uncased` - 英文，精度更高
- `bert-base-chinese` - 中文专用
- `distilbert-base-multilingual-cased` - 多语言通用

---

### 📊 表格数据 AutoML

适合场景：房价预测、客户流失、疾病预测、销量预测...

**数据格式（CSV）：**

```csv
age,income,education,bought
25,50000,bachelor,yes
42,80000,master,yes
19,20000,high_school,no
```

**步骤：**
1. 上传 CSV 文件
2. 填写目标列名（要预测的列）
3. 选择分类或回归任务
4. 点击「开始训练（自动选最优）」

工具会自动尝试以下算法并选择最优：
- Logistic Regression
- Random Forest 🏆（通常最优）
- Gradient Boosting
- XGBoost
- SVM / KNN / Naive Bayes

---

### 🖼️ 图像分类

适合场景：商品识别、缺陷检测、人脸识别、植物分类...

**文件夹结构：**

```
my_dataset/
├── 猫/
│   ├── cat001.jpg
│   ├── cat002.jpg
│   └── ...
├── 狗/
│   ├── dog001.jpg
│   └── ...
└── 鸟/
    └── ...
```

> 每个类别至少需要 10 张图片，建议 50+ 张效果更好

**步骤：**
1. 按上方结构准备图片（文件夹名即类别名）
2. 填入数据集路径
3. 选择模型架构（CPU 用户推荐 MobileNet V3 Small）
4. 点击「开始训练」

**模型对比：**

| 架构 | 参数量 | CPU 训练速度 | 精度 |
|------|--------|------------|------|
| MobileNet V3 Small | 1.5M | ⚡ 快 | 良好 |
| EfficientNet B0 | 5.3M | 🟡 中等 | 优秀 |
| ResNet18 | 11M | 🟡 中等 | 良好 |
| VGG11 | 128M | 🐢 慢 | 优秀 |

---

## 🐍 Python API

除了 Web UI，也可以直接在代码中使用：

### 文本分类

```python
import pandas as pd
from trainers import TextClassifierTrainer

df = pd.read_csv("my_data.csv")

trainer = TextClassifierTrainer(
    model_name="distilbert-base-uncased",
    epochs=3,
    batch_size=16,
)

result = trainer.train(df, text_col="text", label_col="label")
print(f"验证准确率: {result['accuracy']:.2%}")
print(f"模型路径: {result['model_path']}")
```

### 表格数据

```python
import pandas as pd
from trainers import TabularTrainer

df = pd.read_csv("my_data.csv")

trainer = TabularTrainer(task_type="classification")
result = trainer.train(df, target_col="species")

print(f"最佳算法: {result['best_algorithm']}")
print(f"最佳得分: {result['best_score']:.4f}")
```

### 图像分类

```python
from trainers import ImageClassifierTrainer

trainer = ImageClassifierTrainer(
    model_arch="mobilenet_v3_small (轻量 1.5M, 推荐CPU)",
    epochs=10,
)
result = trainer.train(folder="./my_dataset")
print(f"验证准确率: {result['best_acc']:.2%}")

# 推理
from PIL import Image
loaded = ImageClassifierTrainer.load(result["model_path"])
img = Image.open("test.jpg")
class_idx, confidence = loaded.predict(img)
print(f"预测: {result['classes'][class_idx]}，置信度: {confidence:.2%}")
```

---

## 🗂️ 示例数据集

| 数据集 | 任务 | 说明 |
|--------|------|------|
| `examples/sentiment.csv` | 文本分类 | 英文情感分析（30条，正/负/中性） |
| `examples/iris.csv` | 表格分类 | 经典鸢尾花分类（150条，3类） |

---

## 📋 系统要求

- Python 3.9+
- RAM：4GB+（文本模型需要 4-8GB）
- 存储空间：5GB+（模型权重）
- GPU：可选（有 GPU 训练更快，无 GPU 也能运行）

---

## 🤝 贡献

欢迎 PR！可以添加：
- 新的模型架构支持
- 更多示例数据集
- 训练曲线可视化
- 模型量化/压缩功能
- ONNX 导出支持

---

## 📄 许可证

MIT License

---

<div align="center">

如果这个项目对你有帮助，请点 **⭐ Star** 支持！

**[GitHub Issues](https://github.com/sinpoce/ai-trainer-lite/issues)** · **[Discussions](https://github.com/sinpoce/ai-trainer-lite/discussions)**

</div>
