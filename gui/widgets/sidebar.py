"""侧边栏导航组件"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy
from PyQt6.QtCore import pyqtSignal


class SideBar(QWidget):
    page_changed = pyqtSignal(str)

    PAGES = [
        ("dashboard", "🏠  仪表盘"),
        ("text", "📝  文本分类"),
        ("tabular", "📊  表格数据"),
        ("image", "🖼️  图像分类"),
        ("audio", "🎵  音频分类"),
        ("predict", "🔮  批量预测"),
        ("history", "📋  训练历史"),
        ("settings", "⚙️  设置"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.buttons: dict[str, QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo
        title = QLabel("🤖 AI Trainer")
        title.setObjectName("sidebar_title")
        layout.addWidget(title)

        version = QLabel("  v2.0 Desktop")
        version.setStyleSheet("color: #666; font-size: 11px; padding-left: 20px;")
        layout.addWidget(version)
        layout.addSpacing(20)

        # Nav buttons
        for page_id, label in self.PAGES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setObjectName(f"nav_{page_id}")
            btn.clicked.connect(lambda checked, pid=page_id: self._on_click(pid))
            self.buttons[page_id] = btn
            layout.addWidget(btn)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Version info at bottom
        info = QLabel("  Python + PyQt6")
        info.setStyleSheet("color: #555; font-size: 10px; padding: 10px;")
        layout.addWidget(info)

    def _on_click(self, page_id: str):
        for pid, btn in self.buttons.items():
            btn.setChecked(pid == page_id)
        self.page_changed.emit(page_id)

    def set_active(self, page_id: str):
        for pid, btn in self.buttons.items():
            btn.setChecked(pid == page_id)
