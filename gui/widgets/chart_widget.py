"""Matplotlib 图表嵌入组件"""

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class ChartWidget(FigureCanvasQTAgg):
    """可嵌入 PyQt6 的 Matplotlib 图表"""

    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#16213e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._style_ax(self.ax)

    def _style_ax(self, ax):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#a0a0c0", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#0f3460")
        ax.spines["left"].set_color("#0f3460")
        ax.xaxis.label.set_color("#a0a0c0")
        ax.yaxis.label.set_color("#a0a0c0")
        ax.title.set_color("#ffffff")

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._style_ax(self.ax)

    def plot_training_curves(self, history: dict):
        """绘制训练曲线（accuracy + loss 双轴）"""
        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self._style_ax(ax1)
        self._style_ax(ax2)

        epochs = range(1, len(history.get("train_acc", [])) + 1)

        if "train_acc" in history:
            ax1.plot(epochs, history["train_acc"], "-o", color="#6c5ce7", label="训练", markersize=4)
            ax1.plot(epochs, history["val_acc"], "-s", color="#00cec9", label="验证", markersize=4)
            ax1.set_title("准确率", fontsize=13)
            ax1.set_xlabel("Epoch")
            ax1.legend(facecolor="#16213e", edgecolor="#0f3460", labelcolor="#e0e0e0")

        if "train_loss" in history:
            ax2.plot(epochs, history["train_loss"], "-o", color="#fd79a8", label="训练", markersize=4)
            ax2.plot(epochs, history["val_loss"], "-s", color="#fdcb6e", label="验证", markersize=4)
            ax2.set_title("损失", fontsize=13)
            ax2.set_xlabel("Epoch")
            ax2.legend(facecolor="#16213e", edgecolor="#0f3460", labelcolor="#e0e0e0")

        self.fig.tight_layout()
        self.draw()

    def plot_comparison(self, names: list, scores: list, title: str = "算法对比"):
        """绘制算法对比柱状图"""
        self.clear()
        colors = ["#6c5ce7", "#00cec9", "#fd79a8", "#fdcb6e", "#e17055", "#74b9ff", "#55efc4"]
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]

        bars = self.ax.barh(names, scores, color=bar_colors, height=0.6, edgecolor="none")
        self.ax.set_title(title, fontsize=14, pad=10)
        self.ax.set_xlim(0, max(scores) * 1.15 if scores else 1)

        for bar, score in zip(bars, scores):
            self.ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{score:.4f}", va="center", color="#e0e0e0", fontsize=10)

        self.fig.tight_layout()
        self.draw()

    def plot_feature_importance(self, features: dict, top_n: int = 10):
        """绘制特征重要性"""
        self.clear()
        items = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]
        items.reverse()
        names = [x[0] for x in items]
        vals = [x[1] for x in items]

        self.ax.barh(names, vals, color="#6c5ce7", height=0.5)
        self.ax.set_title(f"特征重要性 (Top {top_n})", fontsize=14, pad=10)
        self.fig.tight_layout()
        self.draw()

    def plot_confusion_matrix(self, matrix, labels):
        """绘制混淆矩阵"""
        self.clear()
        im = self.ax.imshow(matrix, cmap="BuPu", aspect="auto")
        self.ax.set_xticks(range(len(labels)))
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        self.ax.set_yticklabels(labels, fontsize=9)
        self.ax.set_title("混淆矩阵", fontsize=14, pad=10)
        self.ax.set_xlabel("预测")
        self.ax.set_ylabel("真实")

        for i in range(len(labels)):
            for j in range(len(labels)):
                self.ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                             color="white" if matrix[i][j] > matrix.max() / 2 else "#a0a0c0")

        self.fig.tight_layout()
        self.draw()

    def plot_data_distribution(self, labels: list, counts: list, title="数据分布"):
        """绘制数据分布饼图"""
        self.clear()
        colors = ["#6c5ce7", "#00cec9", "#fd79a8", "#fdcb6e", "#e17055", "#74b9ff", "#55efc4", "#a29bfe"]
        wedges, texts, autotexts = self.ax.pie(
            counts, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=colors[:len(labels)],
            textprops={"color": "#e0e0e0", "fontsize": 10},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(9)
        self.ax.set_title(title, fontsize=14, pad=10, color="white")
        self.fig.tight_layout()
        self.draw()
