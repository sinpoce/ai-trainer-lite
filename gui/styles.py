"""
全局 QSS 样式表 — 现代深色主题
"""

DARK_THEME = """
/* ── 全局 ─────────────────────────────────────── */
QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

/* ── 侧边栏 ───────────────────────────────────── */
#sidebar {
    background-color: #16213e;
    min-width: 220px;
    max-width: 220px;
    border-right: 1px solid #0f3460;
}

#sidebar QPushButton {
    text-align: left;
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    margin: 2px 8px;
    font-size: 14px;
    color: #a0a0c0;
}

#sidebar QPushButton:hover {
    background-color: #1a1a40;
    color: #ffffff;
}

#sidebar QPushButton:checked,
#sidebar QPushButton[active="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6c5ce7, stop:1 #a29bfe);
    color: #ffffff;
    font-weight: bold;
}

#sidebar_title {
    font-size: 18px;
    font-weight: bold;
    color: #a29bfe;
    padding: 20px 20px 10px 20px;
}

/* ── 主内容区 ──────────────────────────────────── */
#content_area {
    background-color: #1a1a2e;
}

/* ── 卡片 ──────────────────────────────────────── */
.card {
    background-color: #16213e;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #0f3460;
}

QGroupBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
    font-weight: bold;
    font-size: 14px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 8px;
    color: #a29bfe;
}

/* ── 按钮 ──────────────────────────────────────── */
QPushButton {
    padding: 8px 20px;
    border-radius: 8px;
    border: 1px solid #0f3460;
    background-color: #16213e;
    color: #e0e0e0;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #1a1a40;
    border-color: #6c5ce7;
}

QPushButton:pressed {
    background-color: #0f3460;
}

QPushButton#primary_btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6c5ce7, stop:1 #a29bfe);
    border: none;
    color: white;
    font-weight: bold;
    font-size: 14px;
    padding: 10px 30px;
}

QPushButton#primary_btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5b4bd5, stop:1 #918af0);
}

QPushButton#primary_btn:disabled {
    background: #444466;
    color: #888888;
}

QPushButton#danger_btn {
    background-color: #e74c3c;
    border: none;
    color: white;
}

/* ── 输入框 ────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background-color: #0f3460;
    border: 1px solid #1a1a40;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e0e0e0;
    selection-background-color: #6c5ce7;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #6c5ce7;
}

/* ── 下拉框 ────────────────────────────────────── */
QComboBox {
    background-color: #0f3460;
    border: 1px solid #1a1a40;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e0e0e0;
    min-height: 20px;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #16213e;
    border: 1px solid #0f3460;
    selection-background-color: #6c5ce7;
    color: #e0e0e0;
}

/* ── 滑块 ──────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 6px;
    background: #0f3460;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #6c5ce7;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6c5ce7, stop:1 #a29bfe);
    border-radius: 3px;
}

/* ── 进度条 ────────────────────────────────────── */
QProgressBar {
    background-color: #0f3460;
    border: none;
    border-radius: 8px;
    height: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6c5ce7, stop:1 #a29bfe);
    border-radius: 8px;
}

/* ── 表格 ──────────────────────────────────────── */
QTableWidget {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    gridline-color: #0f3460;
    selection-background-color: #6c5ce7;
}

QTableWidget::item {
    padding: 8px;
}

QHeaderView::section {
    background-color: #0f3460;
    color: #a29bfe;
    padding: 8px;
    border: none;
    font-weight: bold;
}

/* ── 标签页 ────────────────────────────────────── */
QTabWidget::pane {
    background-color: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #16213e;
    color: #a0a0c0;
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #6c5ce7;
    color: white;
    font-weight: bold;
}

/* ── 滚动条 ────────────────────────────────────── */
QScrollBar:vertical {
    background: #1a1a2e;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #0f3460;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #6c5ce7;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── Label ─────────────────────────────────────── */
QLabel#page_title {
    font-size: 22px;
    font-weight: bold;
    color: #ffffff;
    padding-bottom: 5px;
}

QLabel#page_subtitle {
    font-size: 13px;
    color: #a0a0c0;
    padding-bottom: 15px;
}

QLabel#stat_value {
    font-size: 28px;
    font-weight: bold;
    color: #a29bfe;
}

QLabel#stat_label {
    font-size: 12px;
    color: #a0a0c0;
}

/* ── CheckBox ──────────────────────────────────── */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #0f3460;
    background: #16213e;
}

QCheckBox::indicator:checked {
    background: #6c5ce7;
    border-color: #6c5ce7;
}

/* ── 提示框 ────────────────────────────────────── */
QToolTip {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #6c5ce7;
    border-radius: 4px;
    padding: 6px;
}
"""
