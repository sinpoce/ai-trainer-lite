"""批量预测页面"""

import os
import pickle
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QComboBox, QLineEdit, QTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class PredictPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        title = QLabel("批量预测")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel("加载已训练的模型，对新数据进行批量预测")
        subtitle.setObjectName("page_subtitle")
        layout.addWidget(subtitle)

        # Model type selection
        type_box = QGroupBox("选择模型类型")
        type_layout = QHBoxLayout(type_box)
        self.type_group = QButtonGroup()
        self.text_radio = QRadioButton("📝 文本分类")
        self.tabular_radio = QRadioButton("📊 表格数据")
        self.text_radio.setChecked(True)
        self.type_group.addButton(self.text_radio)
        self.type_group.addButton(self.tabular_radio)
        type_layout.addWidget(self.text_radio)
        type_layout.addWidget(self.tabular_radio)
        layout.addWidget(type_box)

        # Model path
        model_box = QGroupBox("模型路径")
        model_layout = QHBoxLayout(model_box)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("./models/your-model")
        browse_btn = QPushButton("📂 浏览")
        browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(browse_btn)
        layout.addWidget(model_box)

        main = QHBoxLayout()
        main.setSpacing(20)

        # Left: Input
        left = QVBoxLayout()

        # Text input (for single prediction)
        single_box = QGroupBox("单条预测")
        single_layout = QVBoxLayout(single_box)
        self.single_input = QTextEdit()
        self.single_input.setMaximumHeight(80)
        self.single_input.setPlaceholderText("输入文本或逗号分隔的特征值...")
        single_layout.addWidget(self.single_input)

        predict_one_btn = QPushButton("🔮 预测")
        predict_one_btn.setObjectName("primary_btn")
        predict_one_btn.clicked.connect(self._predict_single)
        single_layout.addWidget(predict_one_btn)

        self.single_result = QLabel("")
        self.single_result.setStyleSheet("color: #55efc4; font-size: 14px; padding: 10px;")
        self.single_result.setWordWrap(True)
        single_layout.addWidget(self.single_result)
        left.addWidget(single_box)

        # Batch input
        batch_box = QGroupBox("批量预测")
        batch_layout = QVBoxLayout(batch_box)

        self.batch_file_label = QLabel("未选择文件")
        self.batch_file_label.setStyleSheet("color: #a0a0c0;")
        batch_file_btn = QPushButton("📂 选择 CSV 文件")
        batch_file_btn.clicked.connect(self._select_batch_file)
        batch_layout.addWidget(batch_file_btn)
        batch_layout.addWidget(self.batch_file_label)

        predict_batch_btn = QPushButton("🚀 批量预测")
        predict_batch_btn.setObjectName("primary_btn")
        predict_batch_btn.clicked.connect(self._predict_batch)
        batch_layout.addWidget(predict_batch_btn)

        export_btn = QPushButton("💾 导出预测结果")
        export_btn.clicked.connect(self._export_results)
        batch_layout.addWidget(export_btn)

        left.addWidget(batch_box)
        left.addStretch()

        # Right: Results table
        right = QVBoxLayout()
        results_box = QGroupBox("预测结果")
        results_layout = QVBoxLayout(results_box)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["输入", "预测类别", "置信度"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self.result_table)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #a0a0c0;")
        results_layout.addWidget(self.stats_label)

        right.addWidget(results_box)

        main.addLayout(left, 1)
        main.addLayout(right, 2)
        layout.addLayout(main)

        self.batch_df = None
        self.predictions_df = None

    def _browse_model(self):
        if self.tabular_radio.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "./models", "Pickle (*.pkl)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择模型目录", "./models")
        if path:
            self.model_path.setText(path)

    def _predict_single(self):
        model_path = self.model_path.text().strip()
        text = self.single_input.toPlainText().strip()
        if not model_path or not text:
            QMessageBox.warning(self, "提示", "请填写模型路径和输入内容")
            return

        try:
            if self.text_radio.isChecked():
                from transformers import pipeline
                clf = pipeline("text-classification", model=model_path, device=-1)
                results = clf(text, top_k=5)
                lines = [f"{r['label']}: {r['score']:.2%}" for r in results]
                self.single_result.setText("预测结果：\n" + "\n".join(lines))
            else:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                model = bundle["model"]
                preprocessor = bundle["preprocessor"]
                feature_cols = bundle["feature_cols"]

                values = [v.strip() for v in text.split(",")]
                row = dict(zip(feature_cols, values))
                df = pd.DataFrame([row])
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        pass

                X = preprocessor.transform(df[feature_cols])
                pred = model.predict(X)
                classes = bundle.get("classes", [])
                label = classes[pred[0]] if classes else str(pred[0])
                self.single_result.setText(f"预测结果：{label}")

        except Exception as e:
            self.single_result.setText(f"❌ 预测失败：{e}")

    def _select_batch_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV (*.csv)")
        if path:
            self.batch_df = pd.read_csv(path)
            self.batch_file_label.setText(f"✓ {os.path.basename(path)} ({len(self.batch_df)} 行)")
            self.batch_file_label.setStyleSheet("color: #55efc4;")

    def _predict_batch(self):
        if self.batch_df is None:
            QMessageBox.warning(self, "提示", "请先选择数据文件")
            return

        model_path = self.model_path.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请填写模型路径")
            return

        try:
            if self.text_radio.isChecked():
                from transformers import pipeline
                clf = pipeline("text-classification", model=model_path, device=-1)
                text_col = self.batch_df.columns[0]
                texts = self.batch_df[text_col].astype(str).tolist()
                results = clf(texts)

                self.result_table.setRowCount(len(results))
                for i, (text, res) in enumerate(zip(texts, results)):
                    self.result_table.setItem(i, 0, QTableWidgetItem(text[:80]))
                    self.result_table.setItem(i, 1, QTableWidgetItem(res["label"]))
                    self.result_table.setItem(i, 2, QTableWidgetItem(f"{res['score']:.2%}"))

                self.predictions_df = self.batch_df.copy()
                self.predictions_df["prediction"] = [r["label"] for r in results]
                self.predictions_df["confidence"] = [r["score"] for r in results]

            else:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                model = bundle["model"]
                preprocessor = bundle["preprocessor"]
                feature_cols = bundle["feature_cols"]
                classes = bundle.get("classes", [])

                X = preprocessor.transform(self.batch_df[feature_cols])
                preds = model.predict(X)

                self.result_table.setRowCount(len(preds))
                for i, pred in enumerate(preds):
                    label = classes[pred] if classes else str(pred)
                    input_str = ", ".join(str(self.batch_df.iloc[i][c]) for c in feature_cols[:3])
                    self.result_table.setItem(i, 0, QTableWidgetItem(input_str[:80]))
                    self.result_table.setItem(i, 1, QTableWidgetItem(label))
                    self.result_table.setItem(i, 2, QTableWidgetItem("-"))

                self.predictions_df = self.batch_df.copy()
                self.predictions_df["prediction"] = [classes[p] if classes else p for p in preds]

            self.stats_label.setText(f"完成：共 {len(self.predictions_df)} 条预测")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败：{e}")

    def _export_results(self):
        if self.predictions_df is None:
            QMessageBox.warning(self, "提示", "没有预测结果可导出")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存预测结果", "predictions.csv", "CSV (*.csv)")
        if path:
            self.predictions_df.to_csv(path, index=False)
            self.stats_label.setText(f"✓ 结果已保存至 {path}")
