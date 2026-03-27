"""仪表盘页面"""

import os
import platform
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QFrame,
)
from PyQt6.QtCore import Qt


class StatCard(QFrame):
    def __init__(self, value: str, label: str, icon: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setStyleSheet("""
            StatCard {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 28px;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        val_lbl = QLabel(value)
        val_lbl.setObjectName("stat_value")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        name_lbl = QLabel(label)
        name_lbl.setObjectName("stat_label")
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon_lbl)
        layout.addWidget(val_lbl)
        layout.addWidget(name_lbl)


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("仪表盘")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel("AI Trainer Lite — 简单几步，训练你自己的 AI 模型")
        subtitle.setObjectName("page_subtitle")
        layout.addWidget(subtitle)

        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        model_count = self._count_models()

        model_size = self._models_disk_usage()

        stats_layout.addWidget(StatCard(str(model_count), "已训练模型", "🧠"))
        stats_layout.addWidget(StatCard("4", "支持任务类型", "📦"))
        stats_layout.addWidget(StatCard("12+", "内置算法", "⚡"))
        stats_layout.addWidget(StatCard(model_size, "模型占用", "💾"))

        layout.addLayout(stats_layout)

        # Feature cards
        features = QGroupBox("支持的训练任务")
        feat_layout = QGridLayout(features)
        feat_layout.setSpacing(15)

        cards_data = [
            ("📝 文本分类", "基于 BERT / DistilBERT 微调\n支持情感分析、主题分类、意图识别",
             "支持模型: DistilBERT, BERT, RoBERTa, 中文BERT"),
            ("📊 表格数据 AutoML", "自动尝试 7+ 种算法，选择最优\n支持分类和回归任务",
             "算法: LR, RF, GBDT, XGBoost, SVM, KNN, NB"),
            ("🖼️ 图像分类", "迁移学习，预训练模型微调\n支持数据增强",
             "架构: MobileNet, ResNet, EfficientNet, VGG"),
            ("🎵 音频分类", "基于 Mel 频谱图 + CNN\n支持环境音、语音命令识别",
             "特征: MFCC, Mel Spectrogram, Chroma"),
        ]

        for i, (title_text, desc, detail) in enumerate(cards_data):
            card = QGroupBox(title_text)
            card_layout = QVBoxLayout(card)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet("color: #c0c0c0; line-height: 1.5;")
            detail_lbl = QLabel(detail)
            detail_lbl.setWordWrap(True)
            detail_lbl.setStyleSheet("color: #6c5ce7; font-size: 11px; margin-top: 5px;")
            card_layout.addWidget(desc_lbl)
            card_layout.addWidget(detail_lbl)
            feat_layout.addWidget(card, i // 2, i % 2)

        layout.addWidget(features)

        # System info
        sys_box = QGroupBox("系统信息")
        sys_layout = QGridLayout(sys_box)

        info = [
            ("操作系统", f"{platform.system()} {platform.release()}"),
            ("Python", platform.python_version()),
            ("CPU", platform.processor() or platform.machine()),
            ("模型目录", os.path.abspath("./models")),
        ]

        try:
            import torch
            info.append(("PyTorch", torch.__version__))
            info.append(("CUDA", "可用 ✓" if torch.cuda.is_available() else "不可用 (CPU 模式)"))
        except ImportError:
            info.append(("PyTorch", "未安装"))

        try:
            import onnx
            info.append(("ONNX 导出", f"✓ v{onnx.__version__}"))
        except ImportError:
            info.append(("ONNX 导出", "未安装"))

        try:
            import transformers
            info.append(("Transformers", transformers.__version__))
        except ImportError:
            info.append(("Transformers", "未安装"))

        for i, (k, v) in enumerate(info):
            key_lbl = QLabel(k)
            key_lbl.setStyleSheet("color: #a0a0c0; font-weight: bold;")
            val_lbl = QLabel(v)
            val_lbl.setStyleSheet("color: #e0e0e0;")
            sys_layout.addWidget(key_lbl, i // 2, (i % 2) * 2)
            sys_layout.addWidget(val_lbl, i // 2, (i % 2) * 2 + 1)

        layout.addWidget(sys_box)
        layout.addStretch()

    def _count_models(self) -> int:
        model_dir = "./models"
        if not os.path.exists(model_dir):
            return 0
        return sum(1 for f in os.listdir(model_dir)
                   if os.path.isdir(os.path.join(model_dir, f)) or f.endswith(".pkl"))

    def _models_disk_usage(self) -> str:
        model_dir = "./models"
        if not os.path.exists(model_dir):
            return "0 MB"
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(model_dir) for f in fns
        )
        if total < 1024 * 1024:
            return f"{total / 1024:.0f} KB"
        elif total < 1024 * 1024 * 1024:
            return f"{total / (1024 * 1024):.0f} MB"
        else:
            return f"{total / (1024 * 1024 * 1024):.1f} GB"
