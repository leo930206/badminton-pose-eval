"""
執行方式：python gui_main.py
需要安裝 PyQt5：pip install PyQt5
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.style import APP_STYLESHEET


def _make_emoji_icon(emoji: str, size: int = 64) -> QIcon:
    """用 QPainter 把 emoji 字元繪製成 QIcon（Windows 上使用 Segoe UI Emoji 彩色字型）。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setFont(QFont("Segoe UI Emoji", int(size * 0.65)))
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, emoji)
    painter.end()
    return QIcon(pix)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    app.setWindowIcon(_make_emoji_icon("🏸"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
