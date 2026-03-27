"""音频分类训练页面"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QComboBox, QLineEdit, QTextEdit,
    QProgressBar, QFileDialog, QSpinBox, QDoubleSpinBox,
    QCheckBox, QMessageBox, QScrollArea,
)
from PyQt6.QtCore import QThread, pyqtSignal
from gui.widgets.chart_widget import ChartWidget


class AudioTrainWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder, feature_type, epochs, batch_size, lr, sample_rate):
        super().__init__()
        self.folder = folder
        self.feature_type = feature_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sample_rate = sample_rate

    def run(self):
        try:
            from trainers.audio_trainer import AudioClassifierTrainer
            trainer = AudioClassifierTrainer(
                feature_type=self.feature_type,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                sample_rate=self.sample_rate,
            )

            def prog(pct, desc=""):
                self.progress.emit(pct, desc)

            result = trainer.train(folder=self.folder, progress_callback=prog)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class AudioPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        title = QLabel("音频分类")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel("上传按类别组织的音频文件夹，基于 Mel 频谱图 + CNN 训练分类模型")
        subtitle.setObjectName("page_subtitle")
        layout.addWidget(subtitle)

        main = QHBoxLayout()
        main.setSpacing(20)

        # Left
        left = QVBoxLayout()

        data_box = QGroupBox("数据集")
        data_layout = QGridLayout(data_box)

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("/path/to/audio_dataset")
        browse_btn = QPushButton("📂 浏览")
        browse_btn.clicked.connect(self._browse)
        data_layout.addWidget(QLabel("数据集路径:"), 0, 0)
        data_layout.addWidget(self.folder_input, 0, 1)
        data_layout.addWidget(browse_btn, 0, 2)

        hint = QLabel("结构：dataset/类别名/audio.wav\n支持格式：.wav .mp3 .flac .ogg")
        hint.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        data_layout.addWidget(hint, 1, 0, 1, 3)
        left.addWidget(data_box)

        model_box = QGroupBox("模型配置")
        model_layout = QGridLayout(model_box)

        model_layout.addWidget(QLabel("特征类型:"), 0, 0)
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["mel_spectrogram", "mfcc"])
        model_layout.addWidget(self.feature_combo, 0, 1)

        model_layout.addWidget(QLabel("采样率:"), 1, 0)
        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(8000, 48000)
        self.sample_rate.setValue(16000)
        self.sample_rate.setSingleStep(8000)
        model_layout.addWidget(self.sample_rate, 1, 1)

        model_layout.addWidget(QLabel("训练轮数:"), 2, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(5, 100)
        self.epochs.setValue(20)
        self.epochs.setSingleStep(5)
        model_layout.addWidget(self.epochs, 2, 1)

        model_layout.addWidget(QLabel("批大小:"), 3, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(4, 64)
        self.batch_size.setValue(16)
        model_layout.addWidget(self.batch_size, 3, 1)

        model_layout.addWidget(QLabel("学习率:"), 4, 0)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(0.00001, 0.01)
        self.lr.setValue(0.001)
        model_layout.addWidget(self.lr, 4, 1)

        left.addWidget(model_box)

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

        # Right
        right = QVBoxLayout()

        result_box = QGroupBox("训练结果")
        result_layout = QVBoxLayout(result_box)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: #0f3460; font-family: monospace;")
        result_layout.addWidget(self.result_text)
        right.addWidget(result_box)

        chart_box = QGroupBox("训练曲线")
        chart_layout = QVBoxLayout(chart_box)
        self.chart = ChartWidget(width=6, height=3.5)
        chart_layout.addWidget(self.chart)
        right.addWidget(chart_box)

        main.addLayout(left, 1)
        main.addLayout(right, 2)
        layout.addLayout(main)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音频数据集")
        if folder:
            self.folder_input.setText(folder)

    def _start_training(self):
        folder = self.folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "错误", "请输入有效的数据集路径")
            return

        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = AudioTrainWorker(
            folder=folder, feature_type=self.feature_combo.currentText(),
            epochs=self.epochs.value(), batch_size=self.batch_size.value(),
            lr=self.lr.value(), sample_rate=self.sample_rate.value(),
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
            f"📊 数据集：\n"
            f"  类别：{result['num_classes']} 个 ({', '.join(result['classes'])})\n"
            f"  训练：{result['train_size']}  验证：{result['eval_size']}\n\n"
            f"🎯 最佳（第 {result['best_epoch']} 轮）：\n"
            f"  准确率：{result['best_acc']:.2%}\n\n"
            f"💾 路径：{result['model_path']}"
        )
        self.result_text.setPlainText(summary)

        if result.get("history"):
            self.chart.plot_training_curves(result["history"])

    def _on_error(self, err):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ 训练失败")
        self.result_text.setPlainText(f"❌ 训练失败：\n{err}")
