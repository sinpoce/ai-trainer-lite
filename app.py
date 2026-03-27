"""
AI Trainer Lite - 简易 AI 模型训练工具
支持文本分类、图像分类、表格数据分类，无需 GPU，开箱即用
"""

import gradio as gr
import pandas as pd
import json
import os
from trainers.text_trainer import TextClassifierTrainer
from trainers.tabular_trainer import TabularTrainer
from trainers.image_trainer import ImageClassifierTrainer

# ─── 文本分类 ────────────────────────────────────────────────────────────────

def run_text_training(file, text_col, label_col, model_name, epochs, batch_size, lr, progress=gr.Progress()):
    if file is None:
        return "❌ 请先上传数据集", "", None

    try:
        df = pd.read_csv(file.name)
        if text_col not in df.columns:
            return f"❌ 找不到列 '{text_col}'，可用列：{list(df.columns)}", "", None
        if label_col not in df.columns:
            return f"❌ 找不到列 '{label_col}'，可用列：{list(df.columns)}", "", None

        trainer = TextClassifierTrainer(
            model_name=model_name,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
        )

        progress(0.1, desc="📂 加载数据...")
        result = trainer.train(
            df=df,
            text_col=text_col,
            label_col=label_col,
            progress_callback=progress,
        )

        summary = f"""✅ 训练完成！

📊 数据集统计：
  - 总样本数：{result['total_samples']}
  - 类别数量：{result['num_classes']}（{', '.join(result['classes'])}）
  - 训练集：{result['train_size']}  验证集：{result['eval_size']}

🎯 模型性能：
  - 验证准确率：{result['accuracy']:.2%}
  - F1 Score：{result['f1']:.4f}

💾 模型已保存至：{result['model_path']}"""

        code = f"""# 使用训练好的模型进行推理
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="{result['model_path']}",
    device=-1  # CPU
)

result = classifier("在此输入你的文本")
print(result)
# 输出示例: [{{'label': '{result['classes'][0]}', 'score': 0.98}}]
"""
        return summary, code, result.get("plot")

    except Exception as e:
        import traceback
        return f"❌ 训练失败：{str(e)}\n\n{traceback.format_exc()}", "", None


def preview_text_data(file, text_col, label_col):
    if file is None:
        return "请上传 CSV 文件", gr.update()
    try:
        df = pd.read_csv(file.name)
        cols = list(df.columns)
        preview = df.head(5).to_markdown(index=False)
        stats = f"共 {len(df)} 行，{len(cols)} 列\n\n{preview}"
        return stats, gr.update(choices=cols), gr.update(choices=cols)
    except Exception as e:
        return f"读取失败：{e}", gr.update(), gr.update()


# ─── 表格数据 ────────────────────────────────────────────────────────────────

def run_tabular_training(file, target_col, task_type, algorithms, test_size, progress=gr.Progress()):
    if file is None:
        return "❌ 请先上传数据集", "", None

    try:
        df = pd.read_csv(file.name)
        if target_col not in df.columns:
            return f"❌ 找不到目标列 '{target_col}'", "", None

        trainer = TabularTrainer(task_type=task_type, algorithms=algorithms or ["all"])
        progress(0.1, desc="🔍 分析数据特征...")

        result = trainer.train(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            progress_callback=progress,
        )

        lines = ["✅ 训练完成！\n", "📊 算法对比（从高到低）：\n"]
        for algo, metrics in sorted(result["all_results"].items(), key=lambda x: -x[1]["score"]):
            marker = "🏆 " if algo == result["best_algorithm"] else "   "
            lines.append(f"{marker}{algo:<25} 得分: {metrics['score']:.4f}  耗时: {metrics['time']:.1f}s")

        lines.append(f"\n🏆 最佳算法：{result['best_algorithm']}")
        lines.append(f"📈 最终得分：{result['best_score']:.4f}")
        lines.append(f"\n特征重要性（Top 5）：")
        for feat, imp in list(result.get("feature_importance", {}).items())[:5]:
            bar = "█" * int(imp * 40)
            lines.append(f"  {feat:<20} {bar} {imp:.4f}")
        lines.append(f"\n💾 模型已保存至：{result['model_path']}")

        code = f"""# 使用训练好的模型
import pickle
import pandas as pd

with open("{result['model_path']}", "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
feature_cols = model_bundle["feature_cols"]

# 准备新数据（与训练数据格式相同）
new_data = pd.DataFrame({{
    # 填入你的数据...
}})

X = preprocessor.transform(new_data[feature_cols])
predictions = model.predict(X)
print(predictions)
"""
        return "\n".join(lines), code, result.get("plot")

    except Exception as e:
        import traceback
        return f"❌ 训练失败：{str(e)}\n\n{traceback.format_exc()}", "", None


def preview_tabular_data(file):
    if file is None:
        return "请上传 CSV 文件", gr.update()
    try:
        df = pd.read_csv(file.name)
        cols = list(df.columns)
        info_lines = [f"共 {len(df)} 行，{len(cols)} 列\n"]
        for col in cols:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            null_pct = df[col].isnull().mean()
            info_lines.append(f"  {col:<25} {dtype:<12} {nunique} 唯一值  {null_pct:.1%} 缺失")
        return "\n".join(info_lines), gr.update(choices=cols)
    except Exception as e:
        return f"读取失败：{e}", gr.update()


# ─── 图像分类 ────────────────────────────────────────────────────────────────

def run_image_training(folder, model_arch, epochs, img_size, batch_size, lr, augment, progress=gr.Progress()):
    if not folder:
        return "❌ 请输入数据集路径", ""
    if not os.path.isdir(folder):
        return f"❌ 路径不存在：{folder}", ""

    try:
        trainer = ImageClassifierTrainer(
            model_arch=model_arch,
            epochs=int(epochs),
            img_size=int(img_size),
            batch_size=int(batch_size),
            lr=float(lr),
            augment=augment,
        )
        progress(0.1, desc="📂 扫描图像文件...")
        result = trainer.train(folder=folder, progress_callback=progress)

        summary = f"""✅ 训练完成！

📊 数据集：
  - 类别：{result['num_classes']}（{', '.join(result['classes'])}）
  - 训练图片：{result['train_size']}  验证图片：{result['eval_size']}

🏗️ 模型架构：{model_arch}（参数量: {result['param_count']:,}）

🎯 最佳性能（第 {result['best_epoch']} 轮）：
  - 验证准确率：{result['best_acc']:.2%}
  - 验证损失：{result['best_loss']:.4f}

💾 模型保存至：{result['model_path']}"""

        code = f"""# 使用训练好的图像分类模型
import torch
from torchvision import transforms
from PIL import Image
from trainers.image_trainer import ImageClassifierTrainer

# 加载模型
trainer = ImageClassifierTrainer.load("{result['model_path']}")
classes = {result['classes']}

# 预测单张图片
img = Image.open("your_image.jpg")
pred_class, confidence = trainer.predict(img)
print(f"预测类别: {{classes[pred_class]}}，置信度: {{confidence:.2%}}")
"""
        return summary, code

    except Exception as e:
        import traceback
        return f"❌ 训练失败：{str(e)}\n\n{traceback.format_exc()}", ""


# ─── 推理测试 ────────────────────────────────────────────────────────────────

def run_inference(model_path, input_text):
    if not model_path or not input_text:
        return "请填写模型路径和输入文本"
    try:
        from transformers import pipeline
        classifier = pipeline("text-classification", model=model_path, device=-1)
        results = classifier(input_text, top_k=None)
        output = "\n".join([f"  {r['label']}: {r['score']:.2%}" for r in results])
        return f"预测结果：\n{output}"
    except Exception as e:
        return f"推理失败：{e}"


# ─── UI ──────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="AI Trainer Lite",
        theme=gr.themes.Soft(primary_hue="violet"),
        css=".gr-button-primary { background: linear-gradient(135deg, #667eea, #764ba2); }"
    ) as demo:
        gr.Markdown("""
# 🤖 AI Trainer Lite
**简单几步，训练你自己的 AI 模型 · 无需 GPU · 无需机器学习背景**

---
""")

        with gr.Tabs():
            # ── 文本分类 Tab ──────────────────────────────────────────────
            with gr.Tab("📝 文本分类", id="text"):
                gr.Markdown("上传带有文本和标签列的 CSV，训练文本分类模型（情感分析、主题分类等）")

                with gr.Row():
                    with gr.Column(scale=1):
                        text_file = gr.File(label="上传 CSV 数据集", file_types=[".csv"])
                        text_col = gr.Textbox(label="文本列名", value="text", placeholder="包含文本内容的列")
                        label_col = gr.Textbox(label="标签列名", value="label", placeholder="包含类别标签的列")

                        with gr.Accordion("⚙️ 训练参数", open=False):
                            text_model = gr.Dropdown(
                                label="基础模型",
                                choices=[
                                    "distilbert-base-uncased",
                                    "bert-base-uncased",
                                    "roberta-base",
                                    "distilbert-base-multilingual-cased",
                                    "bert-base-chinese",
                                ],
                                value="distilbert-base-uncased",
                                info="中文数据推荐 bert-base-chinese"
                            )
                            text_epochs = gr.Slider(1, 10, value=3, step=1, label="训练轮数 (Epochs)")
                            text_batch = gr.Slider(4, 64, value=16, step=4, label="批大小 (Batch Size)")
                            text_lr = gr.Number(value=2e-5, label="学习率", precision=8)

                        text_preview_btn = gr.Button("👁️ 预览数据", variant="secondary")
                        text_train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        text_preview = gr.Textbox(label="数据预览", lines=8, interactive=False)
                        text_result = gr.Textbox(label="训练结果", lines=12, interactive=False)
                        text_code = gr.Code(label="📋 推理代码", language="python", lines=12)
                        text_plot = gr.Image(label="训练曲线", visible=False)

                text_preview_btn.click(
                    fn=lambda f, t, l: preview_text_data(f, t, l),
                    inputs=[text_file, text_col, label_col],
                    outputs=[text_preview, text_col, label_col],
                )
                text_train_btn.click(
                    fn=run_text_training,
                    inputs=[text_file, text_col, label_col, text_model, text_epochs, text_batch, text_lr],
                    outputs=[text_result, text_code, text_plot],
                )

                gr.Examples(
                    examples=[["examples/sentiment.csv", "text", "label"]],
                    inputs=[text_file, text_col, label_col],
                    label="示例数据集",
                )

            # ── 表格数据 Tab ──────────────────────────────────────────────
            with gr.Tab("📊 表格/结构化数据", id="tabular"):
                gr.Markdown("上传 CSV，自动尝试多种算法，找到最适合你数据的模型（支持分类和回归）")

                with gr.Row():
                    with gr.Column(scale=1):
                        tab_file = gr.File(label="上传 CSV 数据集", file_types=[".csv"])
                        tab_target = gr.Textbox(label="目标列名（要预测的列）", value="target")
                        tab_task = gr.Radio(
                            label="任务类型",
                            choices=["classification", "regression"],
                            value="classification"
                        )

                        with gr.Accordion("⚙️ 高级设置", open=False):
                            tab_algos = gr.CheckboxGroup(
                                label="选择算法（留空 = 全部尝试）",
                                choices=[
                                    "LogisticRegression", "RandomForest",
                                    "GradientBoosting", "XGBoost",
                                    "SVM", "KNN", "NaiveBayes",
                                ],
                                value=[],
                            )
                            tab_test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="测试集比例")

                        tab_preview_btn = gr.Button("👁️ 预览数据", variant="secondary")
                        tab_train_btn = gr.Button("🚀 开始训练（自动选最优）", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        tab_preview = gr.Textbox(label="数据信息", lines=10, interactive=False)
                        tab_result = gr.Textbox(label="训练结果", lines=15, interactive=False)
                        tab_code = gr.Code(label="📋 推理代码", language="python", lines=12)
                        tab_plot = gr.Image(label="对比图表", visible=False)

                tab_preview_btn.click(
                    fn=preview_tabular_data,
                    inputs=[tab_file],
                    outputs=[tab_preview, tab_target],
                )
                tab_train_btn.click(
                    fn=run_tabular_training,
                    inputs=[tab_file, tab_target, tab_task, tab_algos, tab_test_size],
                    outputs=[tab_result, tab_code, tab_plot],
                )

                gr.Examples(
                    examples=[["examples/iris.csv", "species", "classification"]],
                    inputs=[tab_file, tab_target, tab_task],
                    label="示例：鸢尾花分类",
                )

            # ── 图像分类 Tab ──────────────────────────────────────────────
            with gr.Tab("🖼️ 图像分类", id="image"):
                gr.Markdown("""
按以下结构组织图片文件夹，然后开始训练：
```
my_dataset/
├── 猫/  (文件夹名 = 类别名)
│   ├── 001.jpg
│   └── 002.jpg
└── 狗/
    ├── 001.jpg
    └── 002.jpg
```
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_folder = gr.Textbox(
                            label="数据集路径",
                            placeholder="/path/to/my_dataset"
                        )
                        img_arch = gr.Dropdown(
                            label="模型架构",
                            choices=[
                                "mobilenet_v3_small (轻量 1.5M, 推荐CPU)",
                                "resnet18 (经典 11M)",
                                "efficientnet_b0 (高效 5.3M)",
                                "vgg11 (经典 128M)",
                            ],
                            value="mobilenet_v3_small (轻量 1.5M, 推荐CPU)"
                        )

                        with gr.Accordion("⚙️ 训练参数", open=False):
                            img_epochs = gr.Slider(5, 50, value=10, step=5, label="训练轮数")
                            img_size = gr.Slider(64, 512, value=224, step=32, label="图像尺寸")
                            img_batch = gr.Slider(4, 64, value=16, step=4, label="批大小")
                            img_lr = gr.Number(value=1e-3, label="学习率", precision=8)
                            img_augment = gr.Checkbox(label="开启数据增强（推荐）", value=True)

                        img_train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        img_result = gr.Textbox(label="训练结果", lines=15, interactive=False)
                        img_code = gr.Code(label="📋 推理代码", language="python", lines=12)

                img_train_btn.click(
                    fn=run_image_training,
                    inputs=[img_folder, img_arch, img_epochs, img_size, img_batch, img_lr, img_augment],
                    outputs=[img_result, img_code],
                )

            # ── 推理测试 Tab ──────────────────────────────────────────────
            with gr.Tab("🔮 推理测试", id="inference"):
                gr.Markdown("输入训练好的文本分类模型路径，直接测试")
                with gr.Row():
                    inf_model_path = gr.Textbox(label="模型路径", placeholder="./models/my-text-classifier")
                    inf_input = gr.Textbox(label="输入文本", placeholder="在这里输入你想分类的文本...")
                inf_btn = gr.Button("🔮 预测", variant="primary")
                inf_output = gr.Textbox(label="预测结果", lines=5, interactive=False)

                inf_btn.click(
                    fn=run_inference,
                    inputs=[inf_model_path, inf_input],
                    outputs=[inf_output],
                )

        gr.Markdown("""
---
<div align="center">

**[GitHub](https://github.com/sinpoce/ai-trainer-lite)** · 如果对你有帮助，请点 ⭐ Star！

</div>
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
