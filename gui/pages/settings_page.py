"""设置页面"""

import os
import json
import glob
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit, QCheckBox, QComboBox,
    QSpinBox, QMessageBox, QFileDialog, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

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

        # ONNX Export
        export_box = QGroupBox("ONNX 模型导出")
        export_layout = QVBoxLayout(export_box)

        onnx_info = QLabel(
            "将训练好的模型转换为 ONNX 格式，可在 C++ / Java / JavaScript / C# 等环境中运行推理。"
        )
        onnx_info.setStyleSheet("color: #a0a0c0;")
        onnx_info.setWordWrap(True)
        export_layout.addWidget(onnx_info)

        # 模型选择
        model_select_layout = QHBoxLayout()
        self.export_model_path = QLineEdit()
        self.export_model_path.setPlaceholderText("选择要导出的模型路径...")
        browse_export_btn = QPushButton("📂 浏览")
        browse_export_btn.clicked.connect(self._browse_export_model)
        model_select_layout.addWidget(self.export_model_path)
        model_select_layout.addWidget(browse_export_btn)
        export_layout.addLayout(model_select_layout)

        export_btns = QHBoxLayout()
        onnx_export_btn = QPushButton("📦 导出为 ONNX")
        onnx_export_btn.setObjectName("primary_btn")
        onnx_export_btn.clicked.connect(self._export_onnx)
        export_btns.addWidget(onnx_export_btn)

        onnx_install_btn = QPushButton("⬇️ 安装 ONNX 依赖")
        onnx_install_btn.clicked.connect(self._install_onnx)
        export_btns.addWidget(onnx_install_btn)
        export_layout.addLayout(export_btns)

        self.export_status = QLabel("")
        self.export_status.setStyleSheet("color: #55efc4;")
        export_layout.addWidget(self.export_status)

        layout.addWidget(export_box)

        # About
        about_box = QGroupBox("关于")
        about_layout = QVBoxLayout(about_box)
        about_text = QLabel(
            "AI Trainer Lite v2.1\n"
            "简易 AI 模型训练工具\n\n"
            "支持：文本分类、表格 AutoML、图像分类、音频分类\n"
            "导出：ONNX 跨平台部署 | PyInstaller EXE 打包\n"
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

    def _browse_export_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型目录", "./models")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "./models", "Pickle (*.pkl)")
        if path:
            self.export_model_path.setText(path)

    def _export_onnx(self):
        model_path = self.export_model_path.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择要导出的模型")
            return
        try:
            from utils.export import auto_export
            self.export_status.setText("⏳ 正在导出...")
            self.export_status.repaint()
            onnx_path = auto_export(model_path)
            self.export_status.setText(f"✅ 导出成功：{onnx_path}")
            QMessageBox.information(self, "导出成功", f"ONNX 模型已保存至：\n{onnx_path}")
        except ImportError as e:
            self.export_status.setText("❌ 缺少依赖")
            QMessageBox.warning(self, "缺少依赖",
                f"导出需要额外依赖：\n\n{e}\n\n请运行：pip install onnx onnxruntime skl2onnx")
        except Exception as e:
            self.export_status.setText(f"❌ 导出失败")
            QMessageBox.critical(self, "导出失败", str(e))

    def _install_onnx(self):
        import subprocess, sys
        reply = QMessageBox.question(self, "安装 ONNX",
            "即将安装 onnx, onnxruntime, skl2onnx 三个包。\n确认安装？")
        if reply == QMessageBox.StandardButton.Yes:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install",
                    "onnx", "onnxruntime", "skl2onnx"])
                QMessageBox.information(self, "成功", "ONNX 依赖安装完成！")
            except Exception as e:
                QMessageBox.critical(self, "安装失败", str(e))
