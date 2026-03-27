"""主窗口"""

import sys
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from gui.styles import DARK_THEME
from gui.widgets.sidebar import SideBar
from gui.pages.dashboard import DashboardPage
from gui.pages.text_page import TextPage
from gui.pages.tabular_page import TabularPage
from gui.pages.image_page import ImagePage
from gui.pages.audio_page import AudioPage
from gui.pages.predict_page import PredictPage
from gui.pages.history_page import HistoryPage
from gui.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trainer Lite v2.0")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)
        self.setStyleSheet(DARK_THEME)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        self.sidebar = SideBar()
        self.sidebar.page_changed.connect(self._switch_page)
        layout.addWidget(self.sidebar)

        # Content stack
        self.stack = QStackedWidget()
        self.stack.setObjectName("content_area")

        self.pages = {
            "dashboard": DashboardPage(),
            "text": TextPage(),
            "tabular": TabularPage(),
            "image": ImagePage(),
            "audio": AudioPage(),
            "predict": PredictPage(),
            "history": HistoryPage(),
            "settings": SettingsPage(),
        }

        for page_id, page_widget in self.pages.items():
            self.stack.addWidget(page_widget)

        layout.addWidget(self.stack, stretch=1)

        # Default page
        self.sidebar.set_active("dashboard")
        self.stack.setCurrentWidget(self.pages["dashboard"])

    def _switch_page(self, page_id: str):
        if page_id in self.pages:
            self.stack.setCurrentWidget(self.pages[page_id])


def run_app():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Trainer Lite")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
