"""训练历史页面"""

import os
import json
import pickle
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTextEdit,
)
from PyQt6.QtCore import Qt


class HistoryPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        title = QLabel("训练历史")
        title.setObjectName("page_title")
        layout.addWidget(title)

        subtitle = QLabel("查看所有已训练的模型，管理和清理模型文件")
        subtitle.setObjectName("page_subtitle")
        layout.addWidget(subtitle)

        # Toolbar
        toolbar = QHBoxLayout()
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self._refresh)
        toolbar.addWidget(refresh_btn)

        delete_btn = QPushButton("🗑️ 删除选中模型")
        delete_btn.setObjectName("danger_btn")
        delete_btn.clicked.connect(self._delete_selected)
        toolbar.addWidget(delete_btn)

        toolbar.addStretch()

        self.total_label = QLabel("")
        self.total_label.setStyleSheet("color: #a0a0c0;")
        toolbar.addWidget(self.total_label)

        layout.addLayout(toolbar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["模型名称", "类型", "算法/架构", "得分", "创建时间", "大小"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self.table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.table, stretch=2)

        # Detail
        detail_box = QGroupBox("模型详情")
        detail_layout = QVBoxLayout(detail_box)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(200)
        self.detail_text.setStyleSheet("background-color: #0f3460; font-family: monospace;")
        detail_layout.addWidget(self.detail_text)
        layout.addWidget(detail_box)

        self._refresh()

    def _refresh(self):
        model_dir = "./models"
        self.models = []

        if not os.path.exists(model_dir):
            self.table.setRowCount(0)
            self.total_label.setText("共 0 个模型")
            return

        for item in os.listdir(model_dir):
            path = os.path.join(model_dir, item)
            info = {"name": item, "path": path}

            if os.path.isdir(path):
                config_path = os.path.join(path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                    info["type"] = "图像" if "arch" in config else "音频"
                    info["algo"] = config.get("arch", "CNN")
                    info["score"] = config.get("best_acc", 0)
                    info["classes"] = config.get("classes", [])
                else:
                    # Transformers model
                    if os.path.exists(os.path.join(path, "config.json")):
                        info["type"] = "文本"
                        info["algo"] = item.split("-")[0] if "-" in item else "BERT"
                        info["score"] = None
                    else:
                        continue

                info["size"] = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, dn, fns in os.walk(path) for f in fns
                )
                info["time"] = datetime.fromtimestamp(os.path.getmtime(path))

            elif item.endswith(".pkl"):
                info["type"] = "表格"
                try:
                    with open(path, "rb") as f:
                        bundle = pickle.load(f)
                    info["algo"] = bundle.get("best_algorithm", "Unknown")
                    info["score"] = bundle.get("best_score", 0)
                except Exception:
                    info["algo"] = "Unknown"
                    info["score"] = None
                info["size"] = os.path.getsize(path)
                info["time"] = datetime.fromtimestamp(os.path.getmtime(path))
            else:
                continue

            self.models.append(info)

        # Sort by time
        self.models.sort(key=lambda x: x.get("time", datetime.min), reverse=True)

        self.table.setRowCount(len(self.models))
        for i, m in enumerate(self.models):
            self.table.setItem(i, 0, QTableWidgetItem(m["name"]))
            self.table.setItem(i, 1, QTableWidgetItem(m.get("type", "?")))
            self.table.setItem(i, 2, QTableWidgetItem(m.get("algo", "?")))

            score = m.get("score")
            score_str = f"{score:.4f}" if score is not None else "-"
            self.table.setItem(i, 3, QTableWidgetItem(score_str))

            time_str = m.get("time", datetime.min).strftime("%Y-%m-%d %H:%M") if m.get("time") else "-"
            self.table.setItem(i, 4, QTableWidgetItem(time_str))

            size_mb = m.get("size", 0) / (1024 * 1024)
            self.table.setItem(i, 5, QTableWidgetItem(f"{size_mb:.1f} MB"))

        self.total_label.setText(f"共 {len(self.models)} 个模型")

    def _on_select(self):
        rows = set(item.row() for item in self.table.selectedItems())
        if len(rows) == 1:
            idx = list(rows)[0]
            m = self.models[idx]
            lines = [
                f"模型名称: {m['name']}",
                f"类型: {m.get('type', '?')}",
                f"算法: {m.get('algo', '?')}",
                f"得分: {m.get('score', '?')}",
                f"路径: {m['path']}",
                f"大小: {m.get('size', 0) / (1024*1024):.1f} MB",
                f"时间: {m.get('time', '')}",
            ]
            if m.get("classes"):
                lines.append(f"类别: {', '.join(m['classes'])}")
            self.detail_text.setPlainText("\n".join(lines))

    def _delete_selected(self):
        rows = sorted(set(item.row() for item in self.table.selectedItems()), reverse=True)
        if not rows:
            QMessageBox.warning(self, "提示", "请先选择要删除的模型")
            return

        names = [self.models[r]["name"] for r in rows]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除以下 {len(names)} 个模型？\n\n" + "\n".join(names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            for r in rows:
                path = self.models[r]["path"]
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            self._refresh()
