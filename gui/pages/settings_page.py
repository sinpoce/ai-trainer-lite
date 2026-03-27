"""设置页面"""

import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit, QCheckBox, QComboBox,
    QSpinBox, QMessageBox,
)
from PyQt6.QtCore import Qt

CONFIG_PATH = "./config.json"


def load_config() -> dict:
    defaults = {
        "model_dir": "./models",
        "default_epochs": 3,
        "default_batch_size": 16,
        "auto_save": True,
        "theme": "dark",
        "language": "zh-CN",
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = load_config()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        title = QLabel("设置")
        title.setObjectName("page_title")
        layout.addWidget(title)

        # General
        gen_box = QGroupBox("通用设置")
        gen_layout = QGridLayout(gen_box)

        gen_layout.addWidget(QLabel("模型保存目录:"), 0, 0)
        self.model_dir = QLineEdit(self.config.get("model_dir", "./models"))
        gen_layout.addWidget(self.model_dir, 0, 1)

        gen_layout.addWidget(QLabel("默认训练轮数:"), 1, 0)
        self.default_epochs = QSpinBox()
        self.default_epochs.setRange(1, 100)
        self.default_epochs.setValue(self.config.get("default_epochs", 3))
        gen_layout.addWidget(self.default_epochs, 1, 1)

        gen_layout.addWidget(QLabel("默认批大小:"), 2, 0)
        self.default_batch = QSpinBox()
        self.default_batch.setRange(1, 128)
        self.default_batch.setValue(self.config.get("default_batch_size", 16))
        gen_layout.addWidget(self.default_batch, 2, 1)

        self.auto_save = QCheckBox("训练完成后自动保存模型")
        self.auto_save.setChecked(self.config.get("auto_save", True))
        gen_layout.addWidget(self.auto_save, 3, 0, 1, 2)

        layout.addWidget(gen_box)

        # Export
        export_box = QGroupBox("模型导出")
        export_layout = QVBoxLayout(export_box)

        onnx_info = QLabel(
            "ONNX 导出允许你将训练好的模型转换为跨平台格式，\n"
            "可在 C++/Java/JavaScript 等环境中运行推理。"
        )
        onnx_info.setStyleSheet("color: #a0a0c0;")
        onnx_info.setWordWrap(True)
        export_layout.addWidget(onnx_info)

        onnx_btn = QPushButton("📦 安装 ONNX 导出支持")
        onnx_btn.clicked.connect(self._install_onnx)
        export_layout.addWidget(onnx_btn)

        layout.addWidget(export_box)

        # About
        about_box = QGroupBox("关于")
        about_layout = QVBoxLayout(about_box)
        about_text = QLabel(
            "AI Trainer Lite v2.0\n"
            "简易 AI 模型训练工具\n\n"
            "支持：文本分类、表格 AutoML、图像分类、音频分类\n"
            "GitHub: github.com/sinpoce/ai-trainer-lite"
        )
        about_text.setStyleSheet("color: #a0a0c0;")
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        layout.addWidget(about_box)

        # Save button
        save_btn = QPushButton("💾 保存设置")
        save_btn.setObjectName("primary_btn")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        layout.addStretch()

    def _save(self):
        self.config["model_dir"] = self.model_dir.text()
        self.config["default_epochs"] = self.default_epochs.value()
        self.config["default_batch_size"] = self.default_batch.value()
        self.config["auto_save"] = self.auto_save.isChecked()
        save_config(self.config)
        QMessageBox.information(self, "成功", "设置已保存")

    def _install_onnx(self):
        QMessageBox.information(
            self, "安装 ONNX",
            "请在终端运行：\n\npip install onnx onnxruntime\n\n安装完成后即可在训练页面使用 ONNX 导出功能。"
        )
