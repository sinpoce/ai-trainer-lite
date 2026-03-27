"""文本分类训练页面"""

import os
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QComboBox, QLineEdit, QTextEdit,
    QProgressBar, QFileDialog, QSpinBox, QDoubleSpinBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from gui.widgets.chart_widget import ChartWidget


class TextTrainWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, df, text_col, label_col, model_name, epochs, batch_size, lr):
        super().__init__()
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self):
        try:
            from trainers.text_trainer import TextClassifierTrainer
            trainer = TextClassifierTrainer(
                model_name=self.model_name,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
            )

            def prog(pct, desc=""):
                self.progress.emit(pct, desc)

            result = trainer.train(
                df=self.df,
                text_col=self.text_col,
                label_col=self.label_col,
                progress_callback=prog,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class TextPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        title = QLabel("文本分类")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel("上传带有文本和标签列的 CSV，训练文本分类模型（情感分析、主题分类等）")
        subtitle.setObjectName("page_subtitle")
        layout.addWidget(subtitle)

        # Main content
        main = QHBoxLayout()
        main.setSpacing(20)

        # Left: Config
        left = QVBoxLayout()

        # Data
        data_box = QGroupBox("数据集")
        data_layout = QGridLayout(data_box)

        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: #a0a0c0;")
        file_btn = QPushButton("📂 选择 CSV 文件")
        file_btn.clicked.connect(self._select_file)

        data_layout.addWidget(file_btn, 0, 0, 1, 2)
        data_layout.addWidget(self.file_label, 1, 0, 1, 2)

        data_layout.addWidget(QLabel("文本列名:"), 2, 0)
        self.text_col = QLineEdit("text")
        data_layout.addWidget(self.text_col, 2, 1)

        data_layout.addWidget(QLabel("标签列名:"), 3, 0)
        self.label_col = QLineEdit("label")
        data_layout.addWidget(self.label_col, 3, 1)

        left.addWidget(data_box)

        # Model config
        model_box = QGroupBox("模型配置")
        model_layout = QGridLayout(model_box)

        model_layout.addWidget(QLabel("基础模型:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
            "distilbert-base-multilingual-cased",
            "bert-base-chinese",
        ])
        model_layout.addWidget(self.model_combo, 0, 1)

        model_layout.addWidget(QLabel("训练轮数:"), 1, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 20)
        self.epochs.setValue(3)
        model_layout.addWidget(self.epochs, 1, 1)

        model_layout.addWidget(QLabel("批大小:"), 2, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(4, 64)
        self.batch_size.setValue(16)
        self.batch_size.setSingleStep(4)
        model_layout.addWidget(self.batch_size, 2, 1)

        model_layout.addWidget(QLabel("学习率:"), 3, 0)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(0.000001, 0.01)
        self.lr.setValue(0.00002)
        self.lr.setSingleStep(0.000005)
        model_layout.addWidget(self.lr, 3, 1)

        left.addWidget(model_box)

        # Buttons
        self.train_btn = QPushButton("🚀 开始训练")
        self.train_btn.setObjectName("primary_btn")
        self.train_btn.clicked.connect(self._start_training)
        left.addWidget(self.train_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #a0a0c0;")
        left.addWidget(self.status_label)

        left.addStretch()

        # Right: Results
        right = QVBoxLayout()

        # Data preview
        preview_box = QGroupBox("数据预览")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setStyleSheet("background-color: #0f3460; border-radius: 6px; font-family: monospace;")
        preview_layout.addWidget(self.preview_text)
        right.addWidget(preview_box)

        # Result
        result_box = QGroupBox("训练结果")
        result_layout = QVBoxLayout(result_box)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setStyleSheet("background-color: #0f3460; border-radius: 6px; font-family: monospace;")
        result_layout.addWidget(self.result_text)
        right.addWidget(result_box)

        # Inference code
        code_box = QGroupBox("推理代码")
        code_layout = QVBoxLayout(code_box)
        self.code_text = QTextEdit()
        self.code_text.setReadOnly(True)
        self.code_text.setMaximumHeight(150)
        self.code_text.setStyleSheet("background-color: #0f3460; border-radius: 6px; font-family: monospace; color: #55efc4;")
        code_layout.addWidget(self.code_text)

        copy_btn = QPushButton("📋 复制代码")
        copy_btn.clicked.connect(lambda: self._copy_code())
        code_layout.addWidget(copy_btn)
        right.addWidget(code_box)

        main.addLayout(left, 1)
        main.addLayout(right, 2)
        layout.addLayout(main)

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV 文件 (*.csv)")
        if path:
            try:
                self.df = pd.read_csv(path)
                name = os.path.basename(path)
                self.file_label.setText(f"✓ {name} ({len(self.df)} 行, {len(self.df.columns)} 列)")
                self.file_label.setStyleSheet("color: #55efc4;")
                self.preview_text.setPlainText(self.df.head(5).to_string(index=False))

                cols = list(self.df.columns)
                if len(cols) >= 2:
                    self.text_col.setText(cols[0])
                    self.label_col.setText(cols[-1])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取文件失败：{e}")

    def _start_training(self):
        if self.df is None:
            QMessageBox.warning(self, "提示", "请先选择数据集")
            return

        tc = self.text_col.text()
        lc = self.label_col.text()
        if tc not in self.df.columns:
            QMessageBox.warning(self, "错误", f"找不到列 '{tc}'，可用列: {list(self.df.columns)}")
            return
        if lc not in self.df.columns:
            QMessageBox.warning(self, "错误", f"找不到列 '{lc}'，可用列: {list(self.df.columns)}")
            return

        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()

        self.worker = TextTrainWorker(
            df=self.df,
            text_col=tc,
            label_col=lc,
            model_name=self.model_combo.currentText(),
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            lr=self.lr.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, desc):
        self.progress_bar.setValue(int(pct * 100))
        self.status_label.setText(desc)

    def _on_finished(self, result):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("✅ 训练完成！")

        summary = (
            f"✅ 训练完成！\n\n"
            f"📊 数据集：{result['total_samples']} 条\n"
            f"   类别：{result['num_classes']} 个 ({', '.join(result['classes'])})\n"
            f"   训练集：{result['train_size']}  验证集：{result['eval_size']}\n\n"
            f"🎯 验证准确率：{result['accuracy']:.2%}\n"
            f"   F1 Score：{result['f1']:.4f}\n\n"
            f"💾 模型路径：{result['model_path']}"
        )
        self.result_text.setPlainText(summary)

        code = (
            f'from transformers import pipeline\n\n'
            f'classifier = pipeline(\n'
            f'    "text-classification",\n'
            f'    model="{result["model_path"]}",\n'
            f'    device=-1\n'
            f')\n\n'
            f'result = classifier("在此输入你的文本")\n'
            f'print(result)'
        )
        self.code_text.setPlainText(code)

    def _on_error(self, err):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ 训练失败")
        self.result_text.setPlainText(f"❌ 训练失败：\n{err}")

    def _copy_code(self):
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code_text.toPlainText())
        self.status_label.setText("✓ 代码已复制到剪贴板")
