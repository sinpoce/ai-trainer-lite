"""表格数据 AutoML 训练页面"""

import os
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QComboBox, QLineEdit, QTextEdit,
    QProgressBar, QFileDialog, QCheckBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QMessageBox, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from gui.widgets.chart_widget import ChartWidget


class TabularTrainWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, df, target_col, task_type, algorithms, test_size):
        super().__init__()
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        self.algorithms = algorithms
        self.test_size = test_size

    def run(self):
        try:
            from trainers.tabular_trainer import TabularTrainer
            trainer = TabularTrainer(
                task_type=self.task_type,
                algorithms=self.algorithms or ["all"],
            )

            def prog(pct, desc=""):
                self.progress.emit(pct, desc)

            result = trainer.train(
                df=self.df,
                target_col=self.target_col,
                test_size=self.test_size,
                progress_callback=prog,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class TabularPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
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

        title = QLabel("表格数据 AutoML")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel("上传 CSV，自动尝试多种算法，找到最适合你数据的模型")
        subtitle.setObjectName("page_subtitle")
        layout.addWidget(subtitle)

        main = QHBoxLayout()
        main.setSpacing(20)

        # Left: Config
        left = QVBoxLayout()

        data_box = QGroupBox("数据集")
        data_layout = QGridLayout(data_box)

        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: #a0a0c0;")
        file_btn = QPushButton("📂 选择 CSV 文件")
        file_btn.clicked.connect(self._select_file)
        data_layout.addWidget(file_btn, 0, 0, 1, 2)
        data_layout.addWidget(self.file_label, 1, 0, 1, 2)

        data_layout.addWidget(QLabel("目标列:"), 2, 0)
        self.target_col = QLineEdit("target")
        data_layout.addWidget(self.target_col, 2, 1)

        left.addWidget(data_box)

        # Task type
        task_box = QGroupBox("任务类型")
        task_layout = QVBoxLayout(task_box)
        self.task_group = QButtonGroup()
        self.cls_radio = QRadioButton("分类 (Classification)")
        self.reg_radio = QRadioButton("回归 (Regression)")
        self.cls_radio.setChecked(True)
        self.task_group.addButton(self.cls_radio)
        self.task_group.addButton(self.reg_radio)
        task_layout.addWidget(self.cls_radio)
        task_layout.addWidget(self.reg_radio)
        left.addWidget(task_box)

        # Algorithms
        algo_box = QGroupBox("算法选择（留空 = 全部尝试）")
        algo_layout = QVBoxLayout(algo_box)
        self.algo_checks = {}
        for name in ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "SVM", "KNN", "NaiveBayes"]:
            cb = QCheckBox(name)
            self.algo_checks[name] = cb
            algo_layout.addWidget(cb)
        left.addWidget(algo_box)

        # Test size
        size_box = QGroupBox("测试集比例")
        size_layout = QHBoxLayout(size_box)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.4)
        self.test_size.setValue(0.2)
        self.test_size.setSingleStep(0.05)
        size_layout.addWidget(self.test_size)
        left.addWidget(size_box)

        self.train_btn = QPushButton("🚀 开始训练（自动选最优）")
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

        preview_box = QGroupBox("数据信息")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setStyleSheet("background-color: #0f3460; font-family: monospace;")
        preview_layout.addWidget(self.preview_text)
        right.addWidget(preview_box)

        result_box = QGroupBox("训练结果")
        result_layout = QVBoxLayout(result_box)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setStyleSheet("background-color: #0f3460; font-family: monospace;")
        result_layout.addWidget(self.result_text)
        right.addWidget(result_box)

        # Chart
        chart_box = QGroupBox("算法对比图")
        chart_layout = QVBoxLayout(chart_box)
        self.chart = ChartWidget(width=5, height=3)
        chart_layout.addWidget(self.chart)
        right.addWidget(chart_box)

        main.addLayout(left, 1)
        main.addLayout(right, 2)
        layout.addLayout(main)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV 文件 (*.csv)")
        if path:
            try:
                self.df = pd.read_csv(path)
                name = os.path.basename(path)
                self.file_label.setText(f"✓ {name} ({len(self.df)} 行, {len(self.df.columns)} 列)")
                self.file_label.setStyleSheet("color: #55efc4;")

                lines = [f"共 {len(self.df)} 行，{len(self.df.columns)} 列\n"]
                for col in self.df.columns:
                    dtype = str(self.df[col].dtype)
                    nunique = self.df[col].nunique()
                    null_pct = self.df[col].isnull().mean()
                    lines.append(f"  {col:<25} {dtype:<12} {nunique} 唯一值  {null_pct:.1%} 缺失")
                self.preview_text.setPlainText("\n".join(lines))

                if len(self.df.columns) > 0:
                    self.target_col.setText(self.df.columns[-1])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"读取失败：{e}")

    def _start_training(self):
        if self.df is None:
            QMessageBox.warning(self, "提示", "请先选择数据集")
            return

        tc = self.target_col.text()
        if tc not in self.df.columns:
            QMessageBox.warning(self, "错误", f"找不到目标列 '{tc}'")
            return

        selected_algos = [n for n, cb in self.algo_checks.items() if cb.isChecked()]
        task_type = "classification" if self.cls_radio.isChecked() else "regression"

        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = TabularTrainWorker(
            df=self.df, target_col=tc, task_type=task_type,
            algorithms=selected_algos, test_size=self.test_size.value(),
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

        lines = ["✅ 训练完成！\n", "📊 算法对比：\n"]
        for algo, metrics in sorted(result["all_results"].items(), key=lambda x: -x[1]["score"]):
            marker = "🏆" if algo == result["best_algorithm"] else "  "
            lines.append(f"{marker} {algo:<25} 得分: {metrics['score']:.4f}  耗时: {metrics['time']:.1f}s")
        lines.append(f"\n🏆 最佳：{result['best_algorithm']} ({result['best_score']:.4f})")
        lines.append(f"💾 模型：{result['model_path']}")
        self.result_text.setPlainText("\n".join(lines))

        # Chart
        names = []
        scores = []
        for algo, metrics in sorted(result["all_results"].items(), key=lambda x: x[1]["score"]):
            names.append(algo)
            scores.append(metrics["score"])
        self.chart.plot_comparison(names, scores, "算法得分对比")

    def _on_error(self, err):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ 训练失败")
        self.result_text.setPlainText(f"❌ 训练失败：\n{err}")
